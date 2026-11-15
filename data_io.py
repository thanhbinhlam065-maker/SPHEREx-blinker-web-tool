"""Đọc dữ liệu thống nhất (SPHEREx / WISE / FITS người dùng) thành danh sách 'epoch'.

Mỗi epoch là một dict:
  index, name, label, source, mjd, mjd_approx, wavelength_um, detector, version,
  image  (float32 2D, NaN = không có dữ liệu),
  bad    (bool 2D, True = pixel xấu / không dùng được),
  flags  (int64 2D hoặc None),
  wcs    (astropy WCS 2D hoặc None),
  clean_ratio (% pixel dùng được)

Không import streamlit nên dùng được cả trong app lẫn dòng lệnh.
"""
from __future__ import annotations

import datetime as _dt
import glob
import os
import re
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings("ignore")

# Bit FLAGS của SPHEREx coi là pixel xấu (mặc định của SPXQuery).
# Bit 21 (SOURCE) chỉ là thông tin "có nguồn đã biết" nên KHÔNG bị loại.
SPHEREX_BAD_BITS = (0, 1, 2, 6, 7, 9, 10, 11, 15)
SPHEREX_BAD_MASK = sum(1 << b for b in SPHEREX_BAD_BITS)

DET_LABELS = {
    1: "D1 (0.75 - 1.09 µm)",
    2: "D2 (1.10 - 1.62 µm)",
    3: "D3 (1.63 - 2.41 µm)",
    4: "D4 (2.42 - 3.82 µm)",
    5: "D5 (3.83 - 4.41 µm)",
    6: "D6 (4.42 - 5.00 µm)",
}
_BAND_EDGES = [(1.095, 1), (1.625, 2), (2.415, 3), (3.825, 4), (4.415, 5)]


# --------------------------------------------------------------------------
# Tiện ích nhỏ
# --------------------------------------------------------------------------
def detector_label(det=None, wl_um=None) -> str:
    """Ưu tiên số detector trong header; nếu thiếu thì suy ra từ bước sóng."""
    try:
        if det is not None and int(det) in DET_LABELS:
            return DET_LABELS[int(det)]
    except (TypeError, ValueError):
        pass
    if wl_um is None or not np.isfinite(wl_um):
        return "Unknown detector"
    for edge, d in _BAND_EDGES:
        if wl_um < edge:
            return DET_LABELS[d]
    return DET_LABELS[6]


def mjd_from_year(year: int) -> float:
    """MJD xấp xỉ giữa năm (1/7) - dùng khi ảnh WISE chỉ có năm."""
    return float((_dt.date(int(year), 7, 1) - _dt.date(1858, 11, 17)).days)


def year_from_name(name: str):
    """Lấy năm 4 chữ số cuối cùng trong tên file (vd 'Barnards_Star_2015_WISE.fits')."""
    m = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", name)
    return int(m[-1]) if m else None


def _wcs_from_header(header):
    try:
        w = WCS(header).celestial
        if w.naxis == 2 and w.has_celestial:
            return w
    except Exception:
        pass
    return None


def _read_first_image(path):
    """Đọc ảnh 2D đầu tiên trong file FITS (cube W1/W2 -> lấy lớp đầu = W1)."""
    with fits.open(path, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if hdu.data is None or isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            data = np.squeeze(hdu.data)
            if data.ndim > 2:
                data = data[0]
            if data.ndim != 2:
                continue
            return data.astype(np.float32), hdu.header.copy(), _wcs_from_header(hdu.header)
    raise ValueError(f"Không tìm thấy ảnh 2D hợp lệ trong {path}")


def _make_epoch(index, name, source, image, wcs, mjd, wl, detector,
                version="", flags=None, label=None, mjd_approx=False):
    image = np.asarray(image, dtype=np.float32)
    bad = ~np.isfinite(image)
    if flags is not None and flags.shape == image.shape:
        bad |= (flags & SPHEREX_BAD_MASK) != 0
    else:
        flags = None
    return {
        "index": int(index),
        "name": name,
        "label": label or name,
        "source": source,
        "mjd": float(mjd),
        "mjd_approx": bool(mjd_approx),
        "wavelength_um": float(wl) if wl is not None else float("nan"),
        "detector": detector,
        "version": version,
        "image": image,
        "bad": bad,
        "flags": flags,
        "wcs": wcs,
        "clean_ratio": float(100.0 * (1.0 - bad.mean())),
    }


# --------------------------------------------------------------------------
# 1. SPHEREx MEF (do SPHEREx.py tạo ra)
# --------------------------------------------------------------------------
def load_spherex_epochs(mef_path: str):
    if not os.path.exists(mef_path):
        return []
    epochs = []
    with fits.open(mef_path) as hdul:
        if "CUTOUT_INFO" not in hdul:
            raise ValueError("Tệp không phải MEF của SPHEREx.py (thiếu CUTOUT_INFO).")
        info = hdul["CUTOUT_INFO"].data
        for i in range(len(info)):
            idx = int(info["cutout_index"][i])
            if f"IMAGE{idx}" not in hdul:
                continue
            hdu = hdul[f"IMAGE{idx}"]
            if hdu.data is None:
                continue
            hdr = hdu.header
            flags = None
            if f"FLAGS{idx}" in hdul and hdul[f"FLAGS{idx}"].data is not None:
                flags = np.array(hdul[f"FLAGS{idx}"].data).astype(np.int64) & 0xFFFFFFFF
            mjd = float(info["observation_date"][i])
            wl = float(info["central_wavelength"][i])
            epochs.append(_make_epoch(
                index=idx,
                name=f"SPHEREx_{idx}",
                source="spherex",
                image=np.array(hdu.data, dtype=np.float32),
                wcs=_wcs_from_header(hdr),
                mjd=mjd,
                wl=wl,
                detector=detector_label(hdr.get("DETECTOR"), wl),
                version=str(info["version"][i]).strip(),
                flags=flags,
                label=f"Epoch #{idx} | MJD {mjd:.2f} | λ={wl:.3f}µm",
            ))
    epochs.sort(key=lambda e: (e["mjd"], e["index"]))
    return epochs


# --------------------------------------------------------------------------
# 2. WISE: FITS_DATA/<thiên thể>/*.fits (hoặc .../Starless/*.fits)
# --------------------------------------------------------------------------
def load_wise_epochs(folder: str, variant: str = "original"):
    """variant = 'original' (ảnh gốc) hoặc 'starless' (ảnh đã loại sao)."""
    starless = variant == "starless"
    search = os.path.join(folder, "Starless") if starless else folder
    files = sorted(glob.glob(os.path.join(search, "*.fits")) +
                   glob.glob(os.path.join(search, "*.fit")))
    skip_tags = ["_stars_mask", "_mef"] + ([] if starless else ["_starless"])
    files = [f for f in files if not any(t in os.path.basename(f).lower() for t in skip_tags)]

    epochs = []
    for i, fp in enumerate(files, start=1):
        base = os.path.basename(fp)
        try:
            img, hdr, wcs = _read_first_image(fp)
        except Exception as exc:
            print(f"  ❌ Bỏ qua '{base}': {exc}")
            continue
        year = year_from_name(base)
        if year is None and hdr.get("YEAR") is not None:
            try:
                year = int(hdr["YEAR"])
            except (TypeError, ValueError):
                year = None
        approx = False
        mjd = hdr.get("MJD-OBS", hdr.get("MJD-AVG"))
        if mjd is None:
            if year is not None:
                mjd, approx = mjd_from_year(year), True
            else:
                mjd, approx = 55000.0 + 365.0 * i, True
        epochs.append(_make_epoch(
            index=i, name=base, source="wise", image=img, wcs=wcs,
            mjd=mjd, wl=3.4, detector="WISE W1 (3.4 µm)", version="unWISE",
            label=f"{year if year else base} | {base}", mjd_approx=approx,
        ))
    epochs.sort(key=lambda e: (e["mjd"], e["name"]))
    for k, e in enumerate(epochs, start=1):
        e["index"] = k
    return epochs


# --------------------------------------------------------------------------
# 2b. WISE từ thư mục do ImageGenerate_WISE.py tạo:
#     ImageGenerate_WISE/<tên>/fits/<tên>_WISE_mef.fits  (+ từng file năm riêng lẻ)
# --------------------------------------------------------------------------
WISE_BASE_DIR = "ImageGenerate_WISE"
_WISE_BANDS = {"W1": 3.4, "W2": 4.6}


def load_wise_mef(mef_path: str, band: str = "W1"):
    """Đọc MEF WISE (bảng EPOCH_INFO + các HDU W1_<năm>/W2_<năm>, mỗi HDU có WCS 2D)."""
    if not os.path.exists(mef_path):
        return []
    wl = _WISE_BANDS[band]
    target = os.path.basename(mef_path).replace("_WISE_mef.fits", "")
    epochs = []
    with fits.open(mef_path) as hdul:
        if "EPOCH_INFO" not in hdul:
            raise ValueError("Tệp không phải MEF của ImageGenerate_WISE.py (thiếu EPOCH_INFO).")
        info = hdul["EPOCH_INFO"].data
        for i in range(len(info)):
            year = str(info["year"][i]).strip()
            ext = f"{band}_{year}"
            if ext not in hdul or hdul[ext].data is None:
                continue
            hdu = hdul[ext]
            layer = str(info["layer"][i]).strip()
            epochs.append(_make_epoch(
                index=i + 1,
                name=f"{target}_{year}_{band}",
                source="wise",
                image=np.array(hdu.data, dtype=np.float32),
                wcs=_wcs_from_header(hdu.header),
                mjd=mjd_from_year(int(year)),
                wl=wl,
                detector=f"WISE {band} ({wl} µm)",
                version=layer,
                label=f"{year} | {layer}",
                mjd_approx=True,
            ))
    epochs.sort(key=lambda e: (e["mjd"], e["name"]))
    for k, e in enumerate(epochs, start=1):
        e["index"] = k
    return epochs


def find_wise_base_dir(start: str = ".", max_depth: int = 4):
    """Tìm thư mục gốc chứa <tên>/fits/*_WISE*.fits (thường là ImageGenerate_WISE).
    Thử đường dẫn mặc định trước; nếu không có thì quét các thư mục con của `start`."""
    default = os.path.join(start, WISE_BASE_DIR)
    if os.path.isdir(default):
        return default
    start = os.path.abspath(start)
    skip = {"__pycache__", "site-packages", "venv", ".venv", "node_modules"}
    for root, dirs, files in os.walk(start):
        depth = root[len(start):].count(os.sep)
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        if depth >= max_depth:
            dirs[:] = []
        if os.path.basename(root) == "fits" and any(
            f.endswith("_WISE_mef.fits") or f.endswith("_WISE.fits") for f in files
        ):
            return os.path.dirname(os.path.dirname(root))
    return None


def load_wise_target(name: str, base_dir=None, band: str = "W1"):
    """Nạp dữ liệu WISE của một thiên thể theo tên thư mục (vd 'Barnards_Star').
    Ưu tiên file MEF; nếu thiếu thì đọc các file <tên>_<năm>_WISE.fits riêng lẻ (W1)."""
    if base_dir is None:
        base_dir = find_wise_base_dir(".")
    if base_dir is None:
        return []
    fits_dir = os.path.join(base_dir, name, "fits")
    epochs = load_wise_mef(os.path.join(fits_dir, f"{name}_WISE_mef.fits"), band)
    if not epochs and band == "W1" and os.path.isdir(fits_dir):
        epochs = load_wise_epochs(fits_dir)
    return epochs


# --------------------------------------------------------------------------
# 3. FITS do người dùng tải lên (1 MEF SPHEREx hoặc nhiều file 2D)
# --------------------------------------------------------------------------
def load_user_fits(paths):
    epochs = []
    for p in paths:
        base = os.path.basename(p)
        try:
            with fits.open(p, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
                is_spx = "CUTOUT_INFO" in hdul
            if is_spx:
                epochs.extend(load_spherex_epochs(p))
                continue
            img, hdr, wcs = _read_first_image(p)
        except Exception as exc:
            print(f"  ❌ Bỏ qua '{base}': {exc}")
            continue
        year = year_from_name(base)
        mjd = hdr.get("MJD-OBS", hdr.get("MJD-AVG"))
        approx = False
        if mjd is None:
            if year is not None:
                mjd, approx = mjd_from_year(year), True
            else:
                mjd, approx = 55000.0 + 365.0 * (len(epochs) + 1), True
        epochs.append(_make_epoch(
            index=len(epochs) + 1, name=base, source="user", image=img, wcs=wcs,
            mjd=mjd, wl=None, detector="User FITS", label=base, mjd_approx=approx,
        ))
    epochs.sort(key=lambda e: (e["mjd"], e["name"]))
    for k, e in enumerate(epochs, start=1):
        e["index"] = k
    return epochs


# --------------------------------------------------------------------------
# Cầu nối sang Process_fits_data.prepare_stack_arrays
# --------------------------------------------------------------------------
def best_reference_index(epochs) -> int:
    """Chọn khung có nhiều pixel dùng được nhất làm lưới tham chiếu
    (tránh chọn nhầm cutout bị cắt hẹp sát mép detector)."""
    return int(np.argmax([(~e["bad"]).sum() for e in epochs]))


def epochs_to_arrays(epochs, mask_bad: bool = True):
    """Trả về (data, wcs_list, names, mjds) cho prepare_stack_arrays."""
    data = [np.where(e["bad"], np.nan, e["image"]).astype(np.float32) if mask_bad
            else e["image"] for e in epochs]
    return data, [e["wcs"] for e in epochs], [e["name"] for e in epochs], [e["mjd"] for e in epochs]