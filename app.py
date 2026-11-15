"""Planet X Hunter — ứng dụng Streamlit (SPHEREx / WISE / FITS người dùng).

Luồng: nhập RA/Dec -> tải dữ liệu (SPHEREx.py | ImageGenerate_WISE.py | upload) -> data_io
-> Process_fits_data (đồng trục, PSF, flux, phát hiện) -> tracking (liên kết quỹ đạo)
-> viz (blinker) -> hiển thị ứng viên + vận tốc.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import ImageDraw

st.set_page_config(
    page_title="Planet X Hunter | SPHEREx & WISE",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

import data_io  # noqa: E402
import tracking  # noqa: E402
import viz  # noqa: E402

try:
    import Process_fits_data as pf  # noqa: E402
except Exception as exc:  # thiếu cv2 / photutils / scipy ...
    st.error(f"Không import được Process_fits_data.py: {type(exc).__name__}: {exc}\n\n"
             "Kiểm tra requirements.txt (cần opencv-python-headless, photutils, scipy, astropy).")
    st.stop()
if not hasattr(pf, "prepare_stack_arrays"):
    st.error("Process_fits_data.py là bản cũ (thiếu hàm `prepare_stack_arrays`). Hãy thay bằng bản mới.")
    st.stop()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(APP_DIR, "cache")
WISE_DIR = os.path.join(APP_DIR, data_io.WISE_BASE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==============================================================================
# 1. MỤC TIÊU
# ==============================================================================
CUSTOM = "Tùy chọn (nhập RA/Dec)"
TARGETS = {
    "M101 (Pinwheel Galaxy - test)": dict(ra=210.80227, dec=54.34895, prefix="M101",
                                          desc="Mục tiêu thử nghiệm mặc định"),
    "Barnard's Star (Fast Proper Motion)": dict(ra=269.45210, dec=4.69340, prefix="Barnards_Star",
                                                desc="Chuyển động riêng lớn nhất bầu trời (~10.3″/năm)"),
    "WISE 0855 (Coldest Brown Dwarf)": dict(ra=133.79510, dec=-7.24510, prefix="WISE_0855",
                                            desc="Lùn nâu loại Y, cách ~7.2 năm ánh sáng"),
    "Epsilon Indi B (Binary Brown Dwarf)": dict(ra=331.04380, dec=-56.78280, prefix="Epsilon_Indi_B",
                                                desc="Cặp sao lùn nâu quay quanh Epsilon Indi"),
    "RW Aurigae (Variable Star)": dict(ra=76.95650, dec=30.40140, prefix="RW_Aurigae",
                                       desc="Sao biến quang T Tauri"),
    "R Draconis (Mira Variable Star)": dict(ra=248.16500, dec=66.75500, prefix="R_Draconis",
                                            desc="Sao biến quang chu kỳ dài (Mira)"),
    CUSTOM: dict(ra=None, dec=None, prefix=None, desc="Toạ độ do người dùng nhập"),
}
PALETTE = [(0, 255, 0), (255, 80, 80), (80, 200, 255), (255, 220, 0), (255, 0, 255), (0, 255, 200)]

SRC_SPX = "🛰️ SPHEREx (IRSA)"
SRC_WISE = "📡 WISE / NEOWISE"
SRC_USER = "📁 FITS của bạn"
SRC_DEMO = "🧪 Demo giả lập (offline)"


def _qfloat(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def object_key(ra, dec):
    """Tên thư mục dữ liệu: dùng tên preset nếu toạ độ trùng (<0.01°), không thì mã hoá từ toạ độ."""
    for t in TARGETS.values():
        if t["prefix"] and abs(t["ra"] - ra) < 0.01 and abs(t["dec"] - dec) < 0.01:
            return t["prefix"]
    return "Custom_%x_%x" % (int(round(ra * 1e4)), int(round((dec + 90.0) * 1e4)))


def pixel_scale_arcsec(wcs):
    if wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        return float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    except Exception:
        return None


# ==============================================================================
# 2. TRẠNG THÁI BAN ĐẦU + URL QUERY PARAMS (link chia sẻ ?target=&ra=&dec=)
# ==============================================================================
qp = st.query_params
if "target_sel" not in st.session_state:
    first = list(TARGETS)[0]
    t_name = qp.get("target", first)
    st.session_state.target_sel = t_name if t_name in TARGETS else first
    base = TARGETS[st.session_state.target_sel]
    st.session_state.ra = _qfloat(qp.get("ra"), base["ra"] if base["ra"] is not None else 0.0)
    st.session_state.dec = _qfloat(qp.get("dec"), base["dec"] if base["dec"] is not None else 0.0)


def _on_target_change():
    t = TARGETS[st.session_state.target_sel]
    if t["ra"] is not None:
        st.session_state.ra = float(t["ra"])
        st.session_state.dec = float(t["dec"])


# ==============================================================================
# 3. LOADER CÓ CACHE
# ==============================================================================
@st.cache_resource(show_spinner="Đang đọc tệp FITS…", max_entries=8)
def _load(kind: str, path: str, mtime: float, band: str = "W1"):
    if kind == "spherex":
        return data_io.load_spherex_epochs(path)
    if kind == "wise_mef":
        return data_io.load_wise_mef(path, band)
    if kind == "wise_files":
        return data_io.load_wise_epochs(path)
    raise ValueError(kind)


def make_demo_epochs():
    """Chuỗi ảnh giả lập (có WCS) với 1 vật thể chuyển động ~11.5″/năm, để thử app khi mạng lỗi."""
    from astropy.wcs import WCS
    rng = np.random.default_rng(7)
    size = 200
    sky = np.zeros((size, size), dtype=np.float32)
    for _ in range(45):
        pf.add_gaussian(sky, rng.uniform(15, size - 15), rng.uniform(15, size - 15), 3.0, rng.uniform(8, 300))
    w = WCS(naxis=2)
    w.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    w.wcs.cdelt = [-2.75 / 3600.0, 2.75 / 3600.0]
    w.wcs.crval = [210.8, 54.35]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    years = [2010, 2014, 2016, 2018, 2020, 2022, 2024]
    eps = []
    for k, yr in enumerate(years):
        img = sky + rng.normal(0, 1.0, sky.shape).astype(np.float32)
        t = yr - years[0]
        pf.add_gaussian(img, 40 + 4.0 * t, 70 + 1.2 * t, 3.0, 25.0)
        eps.append(data_io._make_epoch(
            index=k + 1, name=f"DEMO_{yr}", source="demo", image=img, wcs=w,
            mjd=data_io.mjd_from_year(yr), wl=3.4, detector="Demo (giả lập)",
            label=f"{yr} | demo", mjd_approx=True))
    return eps


def spx_cache_path(ra, dec, band, size, max_cut):
    return os.path.join(CACHE_DIR, f"spherex_{ra:.4f}_{dec:+.4f}_{band}_{size:.3f}_{int(max_cut)}.fits")


def wise_paths(name):
    fits_dir = os.path.join(WISE_DIR, name, "fits")
    return fits_dir, os.path.join(fits_dir, f"{name}_WISE_mef.fits")


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


# ==============================================================================
# 4. SIDEBAR
# ==============================================================================
fetch_req = None
ds = None  # {"sig", "epochs", "source"}

with st.sidebar:
    st.title("🔭 Planet X Hunter")
    st.caption("NASA Space Apps Challenge — SPHEREx & WISE")

    st.selectbox("🎯 Mục tiêu", list(TARGETS), key="target_sel", on_change=_on_target_change)
    c1, c2 = st.columns(2)
    c1.number_input("RA (°)", min_value=0.0, max_value=360.0, step=0.001, format="%.5f", key="ra")
    c2.number_input("Dec (°)", min_value=-90.0, max_value=90.0, step=0.001, format="%.5f", key="dec")
    ra, dec = float(st.session_state.ra), float(st.session_state.dec)
    st.query_params.update({"target": st.session_state.target_sel, "ra": f"{ra:.5f}", "dec": f"{dec:.5f}"})
    okey = object_key(ra, dec)

    st.divider()
    source = st.radio("📂 Nguồn dữ liệu", [SRC_SPX, SRC_WISE, SRC_USER, SRC_DEMO], key="source")

    # ---------------- SPHEREx ----------------
    if source == SRC_SPX:
        band = st.selectbox("Bandpass (detector)", [f"SPHEREx-D{i}" for i in range(1, 7)], index=1)
        size_deg = st.slider("Kích thước cutout (°)", 0.02, 0.30, 0.10, 0.01)
        max_cut = st.slider("Số epoch tối đa (rải đều theo thời gian)", 4, 40, 12)
        with st.expander("Tuỳ chọn tải (khi IRSA quá tải)"):
            workers = st.slider("Số luồng tải", 1, 8, 3, help="IRSA đang quá tải: ít luồng + retry thường ổn định hơn.")
            limit_min = st.slider("Giới hạn thời gian tải (phút)", 1, 20, 6)
        path = spx_cache_path(ra, dec, band, size_deg, max_cut)
        b1, b2 = st.columns(2)
        go = b1.button("⬇️ Tải từ IRSA")
        force = b2.button("🔄 Tải lại")
        if go or force:
            fetch_req = dict(kind="spherex", ra=ra, dec=dec, band=band, size=size_deg, max_cut=max_cut,
                             workers=workers, limit=limit_min, path=path, force=force)
        use_local = st.checkbox("Dùng tệp MEF có sẵn / tải lên")
        if use_local:
            default_local = os.path.join(APP_DIR, "spherex_cutouts_mef.fits")
            local_path = st.text_input("Đường dẫn MEF", value=default_local)
            up = st.file_uploader("Hoặc tải lên MEF (.fits)", type=["fits", "fit"], key="mef_upload")
            if up is not None:
                local_path = os.path.join(CACHE_DIR, "upload_" + os.path.basename(up.name))
                with open(local_path, "wb") as f:
                    f.write(up.getbuffer())
            path = local_path
        if os.path.exists(path):
            try:
                ds = dict(sig=f"spx|{path}|{_mtime(path)}", source="spherex",
                          epochs=_load("spherex", path, _mtime(path)))
            except Exception as exc:
                st.error(f"Không đọc được MEF: {exc}")
        else:
            st.caption("Chưa có dữ liệu cho cấu hình này — bấm **Tải từ IRSA**.")

    # ---------------- WISE ----------------
    elif source == SRC_WISE:
        wband = st.radio("Kênh", ["W1", "W2"], horizontal=True, help="W1 = 3.4 µm, W2 = 4.6 µm")
        wsize = st.slider("Kích thước cutout (pixel)", 100, 512, 200, 20)
        wpix = st.slider("Pixel scale (″/px)", 1.0, 6.0, 2.75, 0.25)
        wname = okey if (wsize == 200 and abs(wpix - 2.75) < 1e-6) else f"{okey}_s{wsize}_p{wpix:g}"
        fits_dir, mef = wise_paths(wname)
        b1, b2 = st.columns(2)
        go = b1.button("⬇️ Tải WISE")
        force = b2.button("🔄 Tải lại")
        if go or force:
            fetch_req = dict(kind="wise", ra=ra, dec=dec, name=wname, size=wsize, pix=wpix, force=force)
        if os.path.exists(mef):
            try:
                ds = dict(sig=f"wise|{mef}|{_mtime(mef)}|{wband}", source="wise",
                          epochs=_load("wise_mef", mef, _mtime(mef), wband))
            except Exception as exc:
                st.error(f"Không đọc được MEF WISE: {exc}")
        elif os.path.isdir(fits_dir) and wband == "W1":
            ds = dict(sig=f"wisef|{fits_dir}|{_mtime(fits_dir)}", source="wise",
                      epochs=_load("wise_files", fits_dir, _mtime(fits_dir)))
        else:
            st.caption("Chưa có dữ liệu WISE cho toạ độ này — bấm **Tải WISE**.")

    # ---------------- Upload ----------------
    elif source == SRC_USER:
        ups = st.file_uploader("Tải lên FITS (nhiều file 2D, hoặc 1 MEF của SPHEREx.py)",
                               type=["fits", "fit", "fz"], accept_multiple_files=True)
        if ups:
            sig = "user|" + hashlib.md5("|".join(f"{u.name}:{u.size}" for u in ups).encode()).hexdigest()
            if st.session_state.get("user_sig") != sig:
                tmpdir = tempfile.mkdtemp(prefix="user_fits_")
                paths = []
                for u in ups:
                    p = os.path.join(tmpdir, os.path.basename(u.name))
                    with open(p, "wb") as f:
                        f.write(u.getbuffer())
                    paths.append(p)
                with st.spinner("Đang đọc FITS…"):
                    st.session_state.user_epochs = data_io.load_user_fits(paths)
                st.session_state.user_sig = sig
            ds = dict(sig=sig, source="user", epochs=st.session_state.get("user_epochs", []))

    # ---------------- Demo ----------------
    else:
        if "demo_epochs" not in st.session_state:
            st.session_state.demo_epochs = make_demo_epochs()
        ds = dict(sig="demo", source="demo", epochs=st.session_state.demo_epochs)
        st.caption("7 epoch giả lập (2010–2024) với 1 vật thể chuyển động ~11.5″/năm.")

    # ---------------- Tham số phân tích ----------------
    st.divider()
    with st.expander("⚙️ Tham số phân tích", expanded=False):
        sigma = st.slider("Ngưỡng phát hiện (σ)", 3.0, 9.0, 5.0, 0.5)
        vmax_as = st.number_input("Chuyển động tối đa (″/năm)", 0.5, 600.0, 20.0, 1.0)
        tol_px = st.slider("Dung sai liên kết (px)", 0.5, 4.0, 1.5, 0.25)
        min_ep = st.slider("Số epoch tối thiểu / quỹ đạo", 3, 12, 3)
        max_det = st.slider("Số phát hiện tối đa / khung", 20, 300, 80, 10)
        min_clean = st.slider("Bỏ epoch có < x% pixel sạch", 0, 95, 50, 5)
        do_psf = st.checkbox("Đồng bộ PSF", True)
        do_scale = st.checkbox("Chuẩn hoá thông lượng", True)
        exact = st.checkbox("reproject_exact (chậm, bảo toàn thông lượng)", False)

# ==============================================================================
# 5. TẢI DỮ LIỆU (chạy ở vùng chính để thấy tiến độ)
# ==============================================================================
def run_fetch(req) -> bool:
    kind = req["kind"]
    with st.status("Đang tải dữ liệu " + ("SPHEREx từ IRSA…" if kind == "spherex" else "WISE từ Legacy Survey…"),
                   expanded=True) as status:
        bar = st.progress(0.0, text="Khởi động…")

        def cb(done, total):
            bar.progress(min(1.0, done / max(total, 1)), text=f"Đã xử lý {done}/{total} yêu cầu")

        t0 = time.time()
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            coord = SkyCoord(ra=req["ra"] * u.deg, dec=req["dec"] * u.deg, frame="icrs")
            if kind == "spherex":
                import SPHEREx as spx
                if req["force"] and os.path.exists(req["path"]):
                    os.remove(req["path"])
                st.write("Truy vấn danh sách cutout (IRSA TAP)…")
                out = spx.run_spherex_cutout_pipeline(
                    coord=coord, size=req["size"] * u.deg, bandpass=req["band"],
                    output_filename=req["path"], parallel=True, max_workers=req["workers"],
                    max_cutouts=req["max_cut"], plot=False, time_limit_s=req["limit"] * 60,
                    min_cutouts=2, progress_cb=cb)
            else:
                import ImageGenerate_WISE as wise
                out = wise.download_wise_target(
                    req["name"], coord, out_dir=WISE_DIR, size=req["size"], pixscale=req["pix"],
                    workers=6, retries=3, progress_cb=cb)
        except Exception as exc:
            status.update(label="Tải thất bại", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return False
        if not out or not os.path.exists(out):
            status.update(label="Không tải được dữ liệu", state="error")
            st.error("Không tải được đủ dữ liệu (máy chủ có thể đang quá tải). Thử lại sau, giảm số epoch/luồng, "
                     "hoặc chuyển sang nguồn **WISE** / **Demo**.")
            return False
        status.update(label=f"Hoàn tất sau {time.time() - t0:.0f} giây", state="complete")
        return True


if fetch_req and run_fetch(fetch_req):
    st.rerun()

# ==============================================================================
# 6. TIÊU ĐỀ + KIỂM TRA DỮ LIỆU
# ==============================================================================
tinfo = TARGETS[st.session_state.target_sel]
st.title(f"🌌 {st.session_state.target_sel}")
st.markdown(f"**ICRS:** `RA = {ra:.5f}°`, `Dec = {dec:+.5f}°` — *{tinfo['desc']}*")

if ds is None or not ds["epochs"]:
    st.info("👈 Chọn nguồn dữ liệu ở thanh bên và tải dữ liệu. Nếu SPHEREx/IRSA đang quá tải, dùng **WISE / NEOWISE** "
            "(baseline 2010–2024, rất tốt để thấy chuyển động riêng), tải FITS của bạn, hoặc thử **Demo giả lập**.")
    st.stop()

epochs_all = ds["epochs"]
sig_ds = ds["sig"]

counts: dict[str, int] = {}
for e in epochs_all:
    counts[e["detector"]] = counts.get(e["detector"], 0) + 1
det_opts = ["Tất cả"] + sorted(counts, key=lambda d: -counts[d])
if len(counts) > 1:
    det_choice = st.sidebar.selectbox("📡 Detector dùng để phân tích", det_opts, index=1, key=f"det_{sig_ds}",
                                      help="Mỗi detector có PSF/phông nền khác nhau; nên phân tích từng detector.")
else:
    det_choice = "Tất cả"

epochs = [e for e in epochs_all
          if (det_choice == "Tất cả" or e["detector"] == det_choice) and e["clean_ratio"] >= min_clean]
epochs.sort(key=lambda e: (e["mjd"], e["index"]))

if len(epochs) < 3:
    st.warning(f"Chỉ có {len(epochs)} epoch đạt điều kiện (cần ≥ 3). Hãy giảm 'Bỏ epoch có < x% pixel sạch', "
               "chọn detector khác, hoặc tải thêm epoch.")
    st.dataframe(pd.DataFrame([{"#": e["index"], "MJD": round(e["mjd"], 3), "λ (µm)": round(e["wavelength_um"], 3),
                                "Detector": e["detector"], "Pixel sạch (%)": round(e["clean_ratio"], 1)}
                               for e in epochs_all]))
    st.stop()

if any(e["mjd_approx"] for e in epochs):
    st.caption("⚠️ Một số epoch không có MJD trong header: dùng ngày xấp xỉ (giữa năm theo tên/năm của lớp dữ liệu). "
               "Vận tốc chỉ chính xác ở mức ~±0.5 năm.")

# ==============================================================================
# 7. PHÂN TÍCH (cache)
# ==============================================================================
@st.cache_data(show_spinner=False, max_entries=6)
def run_analysis(sig: str, _epochs, p: dict):
    ref = data_io.best_reference_index(_epochs)
    data, wcss, names, _ = data_io.epochs_to_arrays(_epochs, mask_bad=True)
    stack, names, invalid, fwhm, keep = pf.prepare_stack_arrays(
        data, wcss, names, ref, p["psf"], p["scale"], p["exact"], return_keep=True)
    used = [_epochs[i] for i in keep]
    ref_wcs = _epochs[ref]["wcs"]
    fwhm_used = fwhm or 3.0
    static_sky, base_mask = pf.prepare_reference(stack, invalid)

    dets, snrs = [], []
    for i in range(len(stack)):
        src, _diff, _excl, _med, std_d = pf.detect_candidates(stack, i, static_sky, base_mask, p["sigma"], fwhm_used)
        if src is not None and len(src):
            xy = pf._xy(src)
            pk = np.asarray(src["peak"], dtype=float) if "peak" in src.colnames else np.ones(len(src))
            order = np.argsort(-pk)[: p["max_det"]]
            dets.append(np.asarray(xy)[order])
            snrs.append(pk[order] / max(std_d, 1e-12))
        else:
            dets.append(np.empty((0, 2)))
            snrs.append(np.empty(0))

    mjds = np.array([e["mjd"] for e in used], dtype=float)
    t_yr = (mjds - mjds.min()) / 365.25
    pscale = pixel_scale_arcsec(ref_wcs)
    vmax_px = p["vmax_as"] / (pscale or 1.0)
    tracks = tracking.link_tracklets(dets, t_yr, vmax_px, tol_px=p["tol"], min_epochs=p["min_ep"])

    sky_ok = ref_wcs is not None
    if sky_ok:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
    for tr in tracks:
        mean_snr = float(np.mean([snrs[k][m] for k, m in tr["members"]]))
        tr["snr"] = mean_snr
        tr["pm_as"] = tr["mu_ra"] = tr["mu_dec"] = tr["pa"] = tr["ra"] = tr["dec"] = np.nan
        if pscale:
            tr["pm_as"] = tr["speed"] * pscale
        if sky_ok:
            try:
                ra_w, dec_w = ref_wcs.all_pix2world([tr["x0"], tr["x0"] + tr["vx"]], [tr["y0"], tr["y0"] + tr["vy"]], 0)
                c0 = SkyCoord(ra_w[0] * u.deg, dec_w[0] * u.deg)
                c1 = SkyCoord(ra_w[1] * u.deg, dec_w[1] * u.deg)
                dlon, dlat = c0.spherical_offsets_to(c1)
                tr["mu_ra"], tr["mu_dec"] = float(dlon.arcsec), float(dlat.arcsec)
                tr["pa"] = float(c0.position_angle(c1).deg)
                tr["pm_as"] = float(c0.separation(c1).arcsec)
                tr["ra"], tr["dec"] = float(ra_w[0]), float(dec_w[0])
            except Exception:
                pass

    return dict(stack=stack, names=names, invalid=invalid, fwhm=fwhm_used, static_sky=static_sky, base_mask=base_mask,
                dets=dets, snrs=snrs, t_yr=t_yr, mjds=mjds, tracks=tracks, keep=keep, pscale=pscale,
                ref=keep.index(ref), shape=stack.shape[1:])


P = dict(sigma=float(sigma), vmax_as=float(vmax_as), tol=float(tol_px), min_ep=int(min_ep), max_det=int(max_det),
         psf=bool(do_psf), scale=bool(do_scale), exact=bool(exact))
sig_an = f"{sig_ds}|{det_choice}|{min_clean}"

try:
    with st.spinner("Đang đồng trục, đồng bộ PSF/thông lượng và tìm vật thể chuyển động…"):
        res = run_analysis(sig_an, epochs, P)
except Exception as exc:
    st.error(f"Lỗi khi phân tích: {type(exc).__name__}: {exc}")
    st.caption("Gợi ý: kiểm tra WCS của các khung, giảm 'Bỏ epoch có < x% pixel sạch', hoặc tắt PSF/thông lượng.")
    st.stop()

used = [epochs[i] for i in res["keep"]]
nf = len(used)
tracks = res["tracks"]
moving = [t for t in tracks if t["moving"]]
span_yr = float(res["t_yr"].max())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Số epoch dùng", f"{nf}/{len(epochs_all)}")
m2.metric("Baseline", f"{span_yr:.2f} năm")
m3.metric("Pixel scale", f"{res['pscale']:.2f}″/px" if res["pscale"] else "n/a")
m4.metric("FWHM đồng bộ", f"{res['fwhm']:.2f} px")
m5.metric("Ứng viên chuyển động", f"{len(moving)}")
if span_yr < 0.5:
    st.caption("ℹ️ Baseline < 6 tháng: chỉ phát hiện được vật thể chuyển động rất nhanh. WISE (2010–2024) cho "
               "baseline dài hơn nhiều.")

# ==============================================================================
# 8. HÀM VẼ
# ==============================================================================
def tracklet_marks(res, ids):
    marks = [[] for _ in range(len(res["names"]))]
    by_id = {t["id"]: t for t in res["tracks"]}
    for j, tid in enumerate(ids):
        tr, col = by_id[tid], PALETTE[j % len(PALETTE)]
        for k, tk in enumerate(res["t_yr"]):
            marks[k].append((float(tr["x0"] + tr["vx"] * tk), float(tr["y0"] + tr["vy"] * tk), col))
    return marks


@st.cache_data(show_spinner="Đang dựng khung hình…", max_entries=6)
def build_blink_frames(sig: str, _stack, mode: str, cmap: str, per_frame: bool, crop_pct: int, marks):
    n, h, w = _stack.shape
    nh, nw = max(8, int(h * crop_pct / 100)), max(8, int(w * crop_pct / 100))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    imgs = [_stack[i, y0:y0 + nh, x0:x0 + nw] for i in range(n)]
    scale = int(max(1, min(10, 520 // max(nh, nw))))
    if per_frame:
        units = [viz.to_unit(im, mode) for im in imgs]
    else:
        lim = viz.stretch_limits(np.concatenate([im.ravel() for im in imgs]), mode)
        units = [viz.to_unit(im, mode, lim) for im in imgs]
    uris = []
    for i, un in enumerate(units):
        pil = viz.colorize(un, cmap, scale).convert("RGB")
        d = ImageDraw.Draw(pil)
        for (x, y, col) in marks[i]:
            xx, yy = x - x0, y - y0
            if -0.5 <= xx < nw - 0.5 and -0.5 <= yy < nh - 0.5:
                cx, cy, r = (xx + 0.5) * scale, (nh - yy - 0.5) * scale, max(7.0, 3.5 * scale)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=tuple(col), width=2)
        uris.append(viz.to_data_uri(pil))
    return uris, nh, nw


def cutout(img, x, y, half):
    xi, yi = int(round(x)), int(round(y))
    padded = np.pad(img, half, mode="constant", constant_values=0.0)
    return padded[yi:yi + 2 * half + 1, xi:xi + 2 * half + 1]


def aperture_flux(img, bad, x, y, r):
    h, w = img.shape
    if not (0 <= x < w and 0 <= y < h):
        return np.nan
    r_in, r_out = r + 2.0, r + 5.0
    x0, x1 = int(max(0, np.floor(x - r_out))), int(min(w, np.ceil(x + r_out) + 1))
    y0, y1 = int(max(0, np.floor(y - r_out))), int(min(h, np.ceil(y + r_out) + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    sub, sb = img[y0:y1, x0:x1], bad[y0:y1, x0:x1]
    good = ~sb & np.isfinite(sub)
    ap, ann = rr <= r, (rr >= r_in) & (rr <= r_out) & good
    ok_ap = ap & good
    if ann.sum() < 5 or ap.sum() == 0 or ok_ap.sum() < 0.7 * ap.sum():
        return np.nan
    bkg = float(np.median(sub[ann]))
    return float(np.sum(sub[ok_ap] - bkg) * ap.sum() / ok_ap.sum())


def tracks_df(trs):
    rows = []
    for t in trs:
        rows.append({
            "ID": t["id"], "Chuyển động": "✅" if t["moving"] else "—", "Số epoch": t["n"],
            "Tốc độ (px/năm)": round(t["speed"], 3),
            "μ tổng (″/năm)": None if np.isnan(t["pm_as"]) else round(t["pm_as"], 3),
            "μα* (″/năm)": None if np.isnan(t["mu_ra"]) else round(t["mu_ra"], 3),
            "μδ (″/năm)": None if np.isnan(t["mu_dec"]) else round(t["mu_dec"], 3),
            "PA (° Bắc→Đông)": None if np.isnan(t["pa"]) else round(t["pa"], 1),
            "RA (°) @epoch đầu": None if np.isnan(t["ra"]) else round(t["ra"], 5),
            "Dec (°) @epoch đầu": None if np.isnan(t["dec"]) else round(t["dec"], 5),
            "RMS (px)": round(t["rms"], 3), "SNR TB": round(t["snr"], 1),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 9. TABS
# ==============================================================================
tab_blink, tab_cand, tab_diff, tab_phot, tab_sens, tab_about = st.tabs([
    "🎬 Blinker", "🎯 Ứng viên & vận tốc", "➖ Trừ ảnh & Stack", "📈 Phổ & Đường cong sáng",
    "🔬 Độ nhạy (Injection)", "ℹ️ Thông tin",
])

# ---------------------------------------------------------------- Blinker
with tab_blink:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    mode = c1.selectbox("Kéo giãn", ["ZScale", "Asinh", "Log", "Linear"])
    cmap = c2.selectbox("Bảng màu", ["inferno", "viridis", "magma", "gray", "cividis"])
    per_frame = c3.checkbox("Chuẩn hoá từng khung", True)
    crop = c4.slider("FoV (%)", 30, 100, 100, 10)
    speed = c5.slider("Tốc độ (ms/khung)", 100, 1500, 400, 50)

    show_marks = st.checkbox("🔴 Khoanh vị trí dự đoán của ứng viên chuyển động", True)
    marks = [[] for _ in range(nf)]
    if show_marks and moving:
        n_show = st.slider("Số ứng viên hiển thị", 1, min(6, len(moving)), min(3, len(moving)))
        marks = tracklet_marks(res, [t["id"] for t in moving[:n_show]])
    elif show_marks:
        st.caption("Chưa có ứng viên chuyển động — thử hạ ngưỡng σ hoặc tăng dung sai.")

    marks_key = tuple(tuple((round(x, 2), round(y, 2), tuple(c)) for x, y, c in m) for m in marks)
    uris, nh, nw = build_blink_frames(f"{sig_an}|{hash(str(P))}", res["stack"], mode, cmap, per_frame, crop, marks_key)
    labels = [f"{e['label']}  |  {len(res['dets'][k])} phát hiện" for k, e in enumerate(used)]
    components.html(viz.blink_html(uris, labels, interval_ms=speed, max_width=560),
                    height=int(min(560, 560 * nh / nw)) + 120)
    st.caption("Ảnh đã đồng trục WCS, đồng bộ PSF, chuẩn hoá thông lượng và trừ nền. Vật thể chuyển động sẽ dịch "
               "chuyển thẳng hàng so với nền sao đứng yên.")
    st.dataframe(pd.DataFrame([{
        "#": k + 1, "Ngày (MJD)": round(e["mjd"], 3), "λ (µm)": None if np.isnan(e["wavelength_um"]) else round(e["wavelength_um"], 3),
        "Detector": e["detector"], "Pixel sạch (%)": round(e["clean_ratio"], 1), "Phát hiện": len(res["dets"][k]),
    } for k, e in enumerate(used)]), hide_index=True)

# ---------------------------------------------------------------- Ứng viên
with tab_cand:
    st.subheader("Quỹ đạo chuyển động thẳng đều (tracklet)")
    if not tracks:
        st.info("Không liên kết được quỹ đạo nào. Thử: hạ ngưỡng σ, tăng dung sai (px) hoặc 'Chuyển động tối đa', "
                "giảm số epoch tối thiểu, hoặc bỏ lọc epoch kém.")
    else:
        only_mv = st.checkbox("Chỉ hiện ứng viên chuyển động", True)
        show_tr = [t for t in tracks if t["moving"]] if only_mv else tracks
        df = tracks_df(show_tr)
        if df.empty:
            st.info("Không có ứng viên chuyển động đáng kể (chỉ có quỹ đạo gần như đứng yên / nhiễu).")
        else:
            st.dataframe(df, hide_index=True)
            st.download_button("⬇️ Tải CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"candidates_{okey}.csv", mime="text/csv")
            sel = st.selectbox("Xem chi tiết quỹ đạo", [t["id"] for t in show_tr],
                               format_func=lambda i: f"#{i}")
            tr = next(t for t in show_tr if t["id"] == sel)

            left, right = st.columns([1, 1])
            with left:
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                ax.imshow(viz.to_unit(res["static_sky"], "Asinh"), cmap="gray", origin="lower")
                cm = plt.get_cmap("viridis")
                for k in range(nf):
                    d = res["dets"][k]
                    if len(d):
                        ax.scatter(d[:, 0], d[:, 1], s=22, facecolors="none", edgecolors=cm(k / max(nf - 1, 1)), lw=0.7)
                tt = np.array([0.0, span_yr])
                ax.plot(tr["x0"] + tr["vx"] * tt, tr["y0"] + tr["vy"] * tt, "r-", lw=1.5)
                ax.scatter(*zip(*[res["dets"][k][m] for k, m in tr["members"]]), s=60, c="red", marker="+")
                ax.set_title(f"Quỹ đạo #{sel} trên nền tĩnh (màu = epoch)")
                ax.set_xlabel("x (px, lưới tham chiếu)")
                ax.set_ylabel("y (px)")
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.metric("Tốc độ", f"{tr['speed']:.3f} px/năm" + (f" = {tr['pm_as']:.2f}″/năm" if not np.isnan(tr["pm_as"]) else ""))
                if not np.isnan(tr["pa"]):
                    st.write(f"**Hướng (PA):** {tr['pa']:.1f}°  |  **μα\\*:** {tr['mu_ra']:+.3f}″/năm  |  **μδ:** {tr['mu_dec']:+.3f}″/năm")
                    st.write(f"**Vị trí @epoch đầu:** RA = {tr['ra']:.5f}°, Dec = {tr['dec']:+.5f}°")
                st.write(f"**Xuất hiện ở {tr['n']}/{nf} epoch**, RMS khớp tuyến tính = {tr['rms']:.2f} px, "
                         f"SNR trung bình = {tr['snr']:.1f}")
                st.warning("Đây là *ứng viên* chuyển động đều. Cần đối chiếu catalog (SIMBAD/AllWISE/Gaia) và kiểm tra "
                           "bằng mắt ở tab Blinker trước khi kết luận; thị sai năm cũng gây dịch chuyển nhỏ.")
                fig, ax = plt.subplots(1, 2, figsize=(5.5, 3))
                ks = [k for k, _ in tr["members"]]
                xs = [res["dets"][k][m, 0] for k, m in tr["members"]]
                ys = [res["dets"][k][m, 1] for k, m in tr["members"]]
                tt_m = res["t_yr"][ks]
                for a, v, lab, v0, vv in ((ax[0], xs, "x", tr["x0"], tr["vx"]), (ax[1], ys, "y", tr["y0"], tr["vy"])):
                    a.plot(tt_m, v, "o")
                    a.plot(tt, v0 + vv * tt, "r-")
                    a.set_xlabel("t (năm)")
                    a.set_ylabel(f"{lab} (px)")
                    a.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("**Cutout quanh vị trí dự đoán theo từng epoch**")
            half = 14
            ncol = min(nf, 6)
            nrow = int(np.ceil(nf / ncol))
            fig, axs = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.3 * nrow))
            axs = np.atleast_1d(axs).ravel()
            for k in range(len(axs)):
                if k < nf:
                    px, py = tr["x0"] + tr["vx"] * res["t_yr"][k], tr["y0"] + tr["vy"] * res["t_yr"][k]
                    cu = cutout(res["stack"][k], px, py, half)
                    axs[k].imshow(viz.to_unit(cu, "Asinh"), cmap="inferno", origin="lower")
                    axs[k].add_patch(plt.Circle((half, half), 5, fill=False, ec="lime", lw=1))
                    axs[k].set_title(used[k]["label"].split("|")[0].strip()[:14], fontsize=8)
                axs[k].axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------------------- Trừ ảnh
with tab_diff:
    c1, c2, c3 = st.columns(3)
    i1 = c1.selectbox("Epoch tham chiếu (T1)", range(nf), index=0, format_func=lambda i: used[i]["label"])
    i2 = c2.selectbox("Epoch đích (T2)", range(nf), index=nf - 1, format_func=lambda i: used[i]["label"])
    smode = c3.selectbox("Kéo giãn ", ["Asinh", "ZScale", "Log", "Linear"], key="diff_stretch")
    img1, img2 = res["stack"][i1], res["stack"][i2]
    diff = img2 - img1
    col_a, col_b, col_c, col_d = st.columns(4)
    for col, im, ttl in ((col_a, img1, f"T1: {used[i1]['label'].split('|')[0]}"),
                         (col_b, img2, f"T2: {used[i2]['label'].split('|')[0]}"),
                         (col_c, res["static_sky"], "Nền tĩnh (median stack)")):
        with col:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(viz.to_unit(im, smode), cmap="inferno", origin="lower")
            ax.set_title(ttl, fontsize=10)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
    with col_d:
        fig, ax = plt.subplots(figsize=(4, 4))
        vlim = np.percentile(np.abs(diff), 99) or 1.0
        ax.imshow(diff, cmap="seismic", vmin=-vlim, vmax=vlim, origin="lower")
        for x, y in res["dets"][i2]:
            ax.add_patch(plt.Circle((x, y), 5, fill=False, ec="lime", lw=1.2))
        ax.set_title(f"T2 − T1 | {len(res['dets'][i2])} phát hiện (T2)", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Vòng xanh: nguồn vượt ngưỡng σ trên ảnh (T2 − nền tĩnh) sau khi đã loại sao bão hoà/viền.")

# ---------------------------------------------------------------- Quang trắc
with tab_phot:
    st.subheader("Phổ & đường cong sáng (quang trắc khẩu độ trên ảnh gốc)")
    st.caption("Dùng ảnh gốc (chưa chuẩn hoá thông lượng) và chuyển vị trí qua WCS của từng epoch.")
    h, w = res["shape"]
    follow_opts = ["Cố định (toạ độ pixel)"] + [f"Theo ứng viên #{t['id']}" for t in moving[:10]]
    follow = st.selectbox("Vị trí lấy mẫu", follow_opts)
    sc1, sc2, sc3 = st.columns(3)
    px0 = sc1.number_input("X (lưới tham chiếu)", 0, w - 1, w // 2)
    py0 = sc2.number_input("Y (lưới tham chiếu)", 0, h - 1, h // 2)
    rad = sc3.slider("Bán kính khẩu độ (px)", 1.0, 8.0, 2.5, 0.5)

    ref_epoch = used[res["ref"]]
    ref_wcs = ref_epoch["wcs"]
    if follow == follow_opts[0]:
        pos_ref = [(float(px0), float(py0))] * nf
    else:
        tid = int(follow.split("#")[1])
        tr = next(t for t in tracks if t["id"] == tid)
        pos_ref = [(tr["x0"] + tr["vx"] * tk, tr["y0"] + tr["vy"] * tk) for tk in res["t_yr"]]

    fl = []
    for k, e in enumerate(used):
        x, y = pos_ref[k]
        if ref_wcs is not None and e["wcs"] is not None:
            try:
                ra_s, dec_s = ref_wcs.all_pix2world(x, y, 0)
                x, y = e["wcs"].all_world2pix(ra_s, dec_s, 0)
                x, y = float(x), float(y)
            except Exception:
                x = y = np.nan
        fl.append(aperture_flux(e["image"], e["bad"], x, y, rad) if np.isfinite(x) and np.isfinite(y) else np.nan)
    fl = np.array(fl)
    wl = np.array([e["wavelength_um"] for e in used])
    mj = res["mjds"]
    has_spec = np.isfinite(wl).any() and np.nanmax(wl) - np.nanmin(wl) > 0.05

    cols = st.columns(2 if has_spec else 1)
    if has_spec:
        with cols[0]:
            fig, ax = plt.subplots(figsize=(6, 4))
            o = np.argsort(wl)
            ax.plot(wl[o], fl[o], "o-", color="#ff6b6b")
            ax.set_xlabel("Bước sóng (µm)")
            ax.set_ylabel("Thông lượng khẩu độ (đơn vị ảnh)")
            ax.set_title("Phổ hồng ngoại")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
    with cols[-1]:
        fig, ax = plt.subplots(figsize=(6, 4))
        o = np.argsort(mj)
        ax.plot(mj[o], fl[o], "s--", color="#4ecdc4")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Thông lượng khẩu độ (đơn vị ảnh)")
        ax.set_title("Đường cong sáng")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    if not has_spec:
        st.caption("Nguồn này chỉ có một kênh phổ nên không vẽ phổ.")

# ---------------------------------------------------------------- Độ nhạy
with tab_sens:
    st.subheader("Injection–Recovery: đo độ nhạy thật của pipeline")
    st.caption("Bơm nguồn Gaussian giả (SNR & tốc độ khác nhau) vào chuỗi ảnh rồi đo tỷ lệ phát hiện lại. "
               "Chạy lưới rút gọn để kịp thời gian.")
    inj_key = f"inj|{sig_an}|{hash(str(P))}"
    if st.button("▶️ Chạy mô phỏng"):
        margin_cap = res["shape"][1] / 2 - 25
        motions = tuple(m for m in (0.0, 1.0, 3.0, 6.0) if 20 + m * (nf - 1) < margin_cap) or (0.0,)
        with st.spinner("Đang mô phỏng (có thể mất 0.5–2 phút)…"):
            st.session_state[inj_key] = pf.injection_recovery(
                res["stack"], snr_grid=(4, 6, 8, 12), motion_grid=motions, n_trials=3, n_inject=5,
                fwhm=res["fwhm"], threshold_sigma=P["sigma"], extra_mask=res["invalid"])
    rows = [r for r in st.session_state.get(inj_key, []) if r["n_injected"] > 0]
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for m in sorted({r["motion"] for r in rows}):
            rr = [r for r in rows if r["motion"] == m]
            ax[0].plot([r["snr"] for r in rr], [r["completeness"] for r in rr], "o-", label=f"{m:g} px/epoch")
            ax[1].plot([r["snr"] for r in rr], [r["fp_per_frame"] for r in rr], "s-", label=f"{m:g} px/epoch")
        ax[0].set_xlabel("SNR đỉnh bơm vào")
        ax[0].set_ylabel("Tỷ lệ phát hiện lại")
        ax[0].set_ylim(-0.02, 1.02)
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].set_xlabel("SNR đỉnh bơm vào")
        ax[1].set_ylabel("Dương tính giả / khung")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------- About
with tab_about:
    st.subheader("📜 Ghi nhận dữ liệu")
    st.info("Ứng dụng dùng dữ liệu SPHEREx (JPL/Caltech, NASA) qua NASA/IPAC IRSA; ảnh WISE/NEOWISE qua Legacy Survey "
            "(unWISE co-adds). Khi công bố, hãy dùng đúng câu acknowledgement chính thức của IRSA/SPHEREx và "
            "WISE/NEOWISE.")
    st.markdown("""
**Pipeline:** `SPHEREx.py` / `ImageGenerate_WISE.py` → `data_io.py` → `Process_fits_data.py`
(đồng trục WCS, đồng bộ PSF, chuẩn hoá thông lượng, trừ nền tĩnh, phát hiện) → `tracking.py`
(liên kết quỹ đạo thẳng đều) → `viz.py` (blinker).

**Giới hạn cần nêu rõ:** baseline SPHEREx chỉ ~1–2 năm nên chỉ nhạy với vật thể chuyển động riêng lớn; chuyển động
rất chậm (kiểu Planet X thực sự) cần baseline dài như WISE/NEOWISE; thị sai và quỹ đạo vệ tinh chưa được hiệu chỉnh.
""")