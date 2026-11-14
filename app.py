import glob
import os
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, LogStretch, ZScaleInterval
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

# ==============================================================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN THIÊN VĂN (DARK THEME)
# ==============================================================================
st.set_page_config(
    page_title="SPHEREx Blinker | Planet X Discovery Tool",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Danh sách 5 mục tiêu bắt buộc của NASA + M101 test mặc định
TARGETS = {
    "M101 (Pinwheel Galaxy - Test)": {"ra": 210.80227, "dec": 54.34895, "prefix": "M101", "desc": "Default test target from SPHEREx cutout pipeline"},
    "Barnard's Star (Fast Proper Motion)": {"ra": 269.45210, "dec": 4.69340, "prefix": "Barnards_Star", "desc": "Fastest moving star across Earth's sky (10.3 arcsec/yr)"},
    "WISE 0855 (Coldest Brown Dwarf)": {"ra": 133.79510, "dec": -7.24510, "prefix": "WISE_0855", "desc": "Fast-moving Y-type brown dwarf at 7.2 light-years"},
    "Epsilon Indi B (Binary Brown Dwarf)": {"ra": 331.04380, "dec": -56.78280, "prefix": "Epsilon_Indi_B", "desc": "Pair of brown dwarfs orbiting Epsilon Indi"},
    "RW Aurigae (Variable Star)": {"ra": 76.95650, "dec": 30.40140, "prefix": "RW_Aurigae", "desc": "T Tauri variable star with dramatic infrared dimming events"},
    "R Draconis (Mira Variable Star)": {"ra": 248.16500, "dec": 66.75500, "prefix": "R_Draconis", "desc": "Long-period Mira variable red giant star"},
}

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU FITS & ẢNH THIÊN VĂN
# ==============================================================================
def align_to_center(arr: np.ndarray, target_size: int = 56) -> np.ndarray:
    """Căn giữa và đồng nhất kích thước khung hình để tránh lỗi lệch shape khi trừ/gộp ảnh."""
    arr = np.nan_to_num(arr, nan=0.0)
    h, w = arr.shape
    out = np.zeros((target_size, target_size), dtype=float)

    # Tính vùng cắt ở tâm
    ch, cw = min(h, target_size), min(w, target_size)
    y0_src = (h - ch) // 2
    x0_src = (w - cw) // 2
    y0_dst = (target_size - ch) // 2
    x0_dst = (target_size - cw) // 2

    out[y0_dst : y0_dst + ch, x0_dst : x0_dst + cw] = arr[y0_src : y0_src + ch, x0_src : x0_src + cw]
    return out


def get_detector_band(wl_um: float) -> str:
    """Phân loại 6 mảng đầu đọc (Detector Arrays D1-D6) theo bước sóng SPHEREx."""
    if wl_um < 1.1:
        return "D1 (0.75 - 1.1 µm)"
    elif wl_um < 1.6:
        return "D2 (1.1 - 1.6 µm)"
    elif wl_um < 2.4:
        return "D3 (1.6 - 2.4 µm)"
    elif wl_um < 3.8:
        return "D4 (2.4 - 3.8 µm)"
    elif wl_um < 4.4:
        return "D5 (3.8 - 4.4 µm)"
    else:
        return "D6 (4.4 - 5.0 µm)"


@st.cache_data(show_spinner=False)
def load_spherex_mef(mef_path: str, align_size: int = 56):
    """Đọc tệp Multi-Extension FITS chuẩn từ SPHERE.py."""
    if not os.path.exists(mef_path):
        return []

    summary = Table.read(mef_path, hdu=1)
    frames = []
    with fits.open(mef_path) as hdul:
        for i in range(len(summary)):
            idx = int(summary["cutout_index"][i])
            mjd = float(summary["observation_date"][i])
            wl = float(summary["central_wavelength"][i])
            ver = str(summary["version"][i])

            raw_img = hdul[f"IMAGE{idx}"].data
            aligned_img = align_to_center(raw_img, align_size)

            # Đọc FLAGS HDU nếu có để kiểm tra chất lượng pixel
            flag_ext = f"FLAGS{idx}"
            if flag_ext in hdul:
                raw_flags = align_to_center(hdul[flag_ext].data, align_size)
                clean_ratio = float(np.mean(raw_flags == 0) * 100.0)
            else:
                raw_flags = np.zeros_like(aligned_img)
                clean_ratio = 100.0

            frames.append({
                "index": idx,
                "label": f"Epoch #{idx} | MJD {mjd:.2f} | λ={wl:.3f}µm",
                "mjd": mjd,
                "wavelength_um": wl,
                "detector": get_detector_band(wl),
                "version": ver,
                "image": aligned_img,
                "flags": raw_flags,
                "clean_ratio": clean_ratio,
            })
    return frames


@st.cache_data(show_spinner=False)
def load_wise_pngs(folder: str, prefix: str):
    """Đọc chuỗi ảnh PNG từ thư mục wise_timelapse nếu người dùng chọn chế độ WISE Overlay."""
    pattern = os.path.join(folder, f"*{prefix}*.png")
    files = sorted(glob.glob(pattern))
    frames = []
    for i, fp in enumerate(files, start=1):
        img = Image.open(fp).convert("L")
        arr = np.array(img, dtype=float)
        # Lấy năm từ tên file nếu có
        fname = os.path.basename(fp)
        frames.append({
            "index": i,
            "label": f"{fname}",
            "mjd": 57000.0 + i * 365.0,
            "wavelength_um": 3.4,
            "detector": "WISE W1 (3.4 µm)",
            "version": "unWISE",
            "image": arr,
            "flags": np.zeros_like(arr),
            "clean_ratio": 100.0,
        })
    return frames


def detect_moving_candidates(diff_img: np.ndarray, sigma_thresh: float = 4.0):
    """Tự động khoanh vùng các điểm sáng/tối dị thường trên ảnh trừ (Difference Image)."""
    smooth = gaussian_filter(np.abs(diff_img), sigma=1.0)
    med = np.median(smooth)
    std = np.std(smooth)
    thresh = med + sigma_thresh * std

    local_max = (smooth == maximum_filter(smooth, size=7)) & (smooth > thresh)
    ys, xs = np.where(local_max)
    # Giới hạn tối đa 12 ứng viên mạnh nhất
    intensities = smooth[ys, xs]
    top_idx = np.argsort(intensities)[::-1][:12]
    return [(int(xs[k]), int(ys[k]), float(intensities[k])) for k in top_idx]


# ==============================================================================
# 3. XỬ LÝ URL QUERY PARAMETERS (HỖ TRỢ NHÚNG LINK TRỰC TIẾP THEO ĐỀ BÀI)
# ==============================================================================
query_params = st.query_params
default_target_name = list(TARGETS.keys())[0]
if "target" in query_params and query_params["target"] in TARGETS:
    default_target_name = query_params["target"]

# ==============================================================================
# 4. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.title("🛰️ SPHEREx Explorer")
    st.caption("NASA Space Apps Challenge 2026 — Planet X & SPHEREx")

    # Chọn đối tượng hoặc nhập tọa độ RA/Dec
    selected_target = st.selectbox(
        "🎯 Select Target Object",
        options=list(TARGETS.keys()),
        index=list(TARGETS.keys()).index(default_target_name),
    )
    target_info = TARGETS[selected_target]

    col_ra, col_dec = st.columns(2)
    ra_input = col_ra.number_input("RA (deg)", value=float(query_params.get("ra", target_info["ra"])), format="%.5f")
    dec_input = col_dec.number_input("Dec (deg)", value=float(query_params.get("dec", target_info["dec"])), format="%.5f")

    # Cập nhật URL query params để chia sẻ link dễ dàng
    st.query_params["target"] = selected_target
    st.query_params["ra"] = f"{ra_input:.5f}"
    st.query_params["dec"] = f"{dec_input:.5f}"

    st.divider()

    # Chọn nguồn dữ liệu
    data_source = st.radio(
        "📂 Data Source",
        options=[
            "SPHEREx MEF FITS (spherex_cutouts_mef.fits)",
            "WISE Historical Timelapse (wise_timelapse/)",
        ],
    )

    mef_file_path = "spherex_cutouts_mef.fits"
    if "SPHEREx" in data_source:
        mef_file_path = st.text_input("FITS File Path", value="spherex_cutouts_mef.fits")
        uploaded_fits = st.file_uploader("Or Upload SPHEREx MEF (.fits)", type=["fits", "fit"])
        if uploaded_fits is not None:
            with open("temp_uploaded_mef.fits", "wb") as f:
                f.write(uploaded_fits.getbuffer())
            mef_file_path = "temp_uploaded_mef.fits"

    st.divider()

    # Tùy chỉnh hiển thị hình ảnh (FoV Zoom, Stretch, Colormap)
    st.subheader("🎨 Image Display Settings")
    fov_zoom = st.slider("Field of View Zoom (Crop %)", min_value=30, max_value=100, value=100, step=10)
    stretch_mode = st.selectbox("Intensity Stretch", ["ZScale", "Log", "Asinh", "Linear"])
    cmap_choice = st.selectbox("Colormap", ["inferno", "viridis", "magma", "gray", "cividis"])

# ==============================================================================
# 5. TẢI DỮ LIỆU VÀ LỌC THEO DETECTOR ARRAY
# ==============================================================================
if "SPHEREx" in data_source:
    all_frames = load_spherex_mef(mef_file_path)
else:
    all_frames = load_wise_pngs("wise_timelapse", target_info["prefix"])

st.title(f"🌌 {selected_target}")
st.markdown(f"**Coordinates (ICRS):** `RA = {ra_input:.5f}°`, `Dec = {dec_input:.5f}°` — *{target_info['desc']}*")

if not all_frames:
    st.warning(
        f"⚠️ Chưa tìm thấy tệp dữ liệu `{mef_file_path}` (hoặc thư mục `wise_timelapse/`). "
        "Hãy chạy `SPHERE.py` trước hoặc tải tệp `.fits` lên ở thanh bên trái."
    )
    st.stop()

# Bộ lọc theo Detector Array (D1 - D6)
available_detectors = ["All Detectors (102 Bands)"] + sorted(list({f["detector"] for f in all_frames}))
selected_detector = st.sidebar.selectbox("📡 Detector Array Filter", available_detectors)

if selected_detector != "All Detectors (102 Bands)":
    frames = [f for f in all_frames if f["detector"] == selected_detector]
else:
    frames = all_frames

if not frames:
    st.error("Không có khung hình nào khớp với Detector Array đã chọn.")
    st.stop()

# ==============================================================================
# 6. GIAO DIỆN CHÍNH DẠNG TAB (BLINKER | DIFF & STACK | SPECTRUM | ABOUT)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Time-Lapse Blinker",
    "➖ Image Subtraction & Stacking",
    "📈 On-the-Fly Spectrum & Light Curve",
    "ℹ️ Mission & Acknowledgement",
])

# Hàm phụ cắt ảnh theo FoV Zoom
def apply_fov_crop(img: np.ndarray, zoom_pct: int) -> np.ndarray:
    if zoom_pct >= 100:
        return img
    h, w = img.shape
    nh, nw = int(h * zoom_pct / 100), int(w * zoom_pct / 100)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    return img[y0 : y0 + nh, x0 : x0 + nw]


def get_norm(img: np.ndarray, mode: str):
    if mode == "ZScale":
        vmin, vmax = ZScaleInterval().get_limits(img)
        return ImageNormalize(vmin=vmin, vmax=vmax)
    elif mode == "Log":
        positive = img[img > 0]
        vmin = float(np.percentile(positive, 5)) if len(positive) > 0 else 0.1
        vmax = float(np.percentile(img, 99.5))
        return ImageNormalize(vmin=max(vmin, 1e-3), vmax=max(vmax, vmin + 1.0), stretch=LogStretch())
    elif mode == "Asinh":
        vmin, vmax = np.percentile(img, [1, 99.5])
        return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    else:
        vmin, vmax = np.percentile(img, [1, 99.5])
        return ImageNormalize(vmin=vmin, vmax=vmax)


# ------------------------------------------------------------------------------
# TAB 1: TIME-LAPSE BLINKER (CHỚP ẢNH TÌM VẬT THỂ DI CHUYỂN)
# ------------------------------------------------------------------------------
with tab1:
    if "frame_idx" not in st.session_state or st.session_state.frame_idx >= len(frames):
        st.session_state.frame_idx = 0

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 3])
    auto_blink = ctrl_col1.toggle("▶️ Auto-Blink (Loop)", value=False)
    blink_speed = ctrl_col2.slider("Speed (sec/frame)", 0.1, 1.5, 0.35, 0.05)

    if len(frames) > 1:
        selected_idx = ctrl_col3.slider(
            "Timeline Epoch Scrubber",
            min_value=0,
            max_value=len(frames) - 1,
            value=st.session_state.frame_idx,
        )
        st.session_state.frame_idx = selected_idx
    else:
        st.session_state.frame_idx = 0

    cur = frames[st.session_state.frame_idx]
    disp_img = apply_fov_crop(cur["image"], fov_zoom)
    disp_flags = apply_fov_crop(cur["flags"], fov_zoom)

    view_col, meta_col = st.columns([2, 1])
    with view_col:
        show_flag_overlay = st.checkbox("🚩 Overlay Contamination Flags (`FLAGS > 0`)", value=False)
        fig, ax = plt.subplots(figsize=(6, 6))
        norm = get_norm(disp_img, stretch_mode)
        im = ax.imshow(disp_img, cmap=cmap_choice, norm=norm, origin="lower")

        if show_flag_overlay and np.any(disp_flags > 0):
            mask = np.ma.masked_where(disp_flags == 0, disp_flags)
            ax.imshow(mask, cmap="spring", alpha=0.45, origin="lower")

        ax.set_title(f"{cur['label']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("RA Offset (pixels)")
        ax.set_ylabel("Dec Offset (pixels)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Surface Brightness / Flux")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with meta_col:
        st.subheader("📊 Cutout Metadata")
        st.metric("Observation Date (MJD)", f"{cur['mjd']:.4f}")
        st.metric("Central Wavelength", f"{cur['wavelength_um']:.4f} µm")
        st.metric("Detector Array", cur["detector"])
        st.metric("Clean Pixels (FLAGS = 0)", f"{cur['clean_ratio']:.1f}%")
        st.info(
            "💡 **Mẹo phát hiện Planet X / Sao lùn nâu:** Bật **Auto-Blink** để mắt người tự động bắt chuyển động "
            "thẳng hàng của thiên thể gần Hệ Mặt Trời so với nền sao đứng yên phía sau."
        )

    # Vòng lặp chạy Auto-Blink mượt mà
    if auto_blink and len(frames) > 1:
        time.sleep(blink_speed)
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(frames)
        st.rerun()

# ------------------------------------------------------------------------------
# TAB 2: IMAGE SUBTRACTION & STACKING (TRỪ & CỘNG ẢNH)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Difference Imaging & Co-added Stacking")
    st.markdown(
        "Lấy ảnh mốc thời gian sau trừ đi mốc trước (`T2 - T1`) để triệt tiêu các ngôi sao nền cố định "
        "và tự động khoanh tròn các ứng viên di chuyển hoặc biến quang."
    )

    if len(frames) < 2:
        st.warning("Cần ít nhất 2 khung hình để thực hiện phép trừ ảnh.")
    else:
        c1, c2, c3 = st.columns(3)
        idx_t1 = c1.selectbox("Reference Epoch (T1)", range(len(frames)), index=0, format_func=lambda i: frames[i]["label"])
        idx_t2 = c2.selectbox("Target Epoch (T2)", range(len(frames)), index=len(frames) - 1, format_func=lambda i: frames[i]["label"])
        sigma_val = c3.slider("Detection Sensitivity (Sigma)", 2.5, 8.0, 4.5, 0.5)

        img1 = apply_fov_crop(frames[idx_t1]["image"], fov_zoom)
        img2 = apply_fov_crop(frames[idx_t2]["image"], fov_zoom)
        diff = img2 - img1
        candidates = detect_moving_candidates(diff, sigma_thresh=sigma_val)

        col_left, col_mid, col_right = st.columns(3)
        with col_left:
            fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
            ax1.imshow(img1, cmap=cmap_choice, norm=get_norm(img1, stretch_mode), origin="lower")
            ax1.set_title(f"T1: Epoch #{frames[idx_t1]['index']}")
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with col_mid:
            fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
            ax2.imshow(img2, cmap=cmap_choice, norm=get_norm(img2, stretch_mode), origin="lower")
            ax2.set_title(f"T2: Epoch #{frames[idx_t2]['index']}")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        with col_right:
            fig3, ax3 = plt.subplots(figsize=(4.5, 4.5))
            vmax_diff = np.percentile(np.abs(diff), 99)
            ax3.imshow(diff, cmap="seismic", vmin=-vmax_diff, vmax=vmax_diff, origin="lower")
            for cx, cy, _ in candidates:
                circ = patches.Circle((cx, cy), radius=3.5, edgecolor="lime", facecolor="none", linewidth=1.5)
                ax3.add_patch(circ)
            ax3.set_title(f"Diff (T2 - T1) | {len(candidates)} Candidates")
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        if candidates:
            st.success(f"✅ Phát hiện **{len(candidates)}** vùng tín hiệu dị thường (khoanh tròn màu xanh lá trên ảnh Diff).")

# ------------------------------------------------------------------------------
# TAB 3: ON-THE-FLY SPECTRUM & LIGHT CURVE (ĐỒ THỊ PHỔ & ĐỘ SÁNG)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🌈 On-the-Fly Spectrophotometry & Light Curve")
    st.markdown("Chọn tọa độ pixel `(X, Y)` trên khung hình để trích xuất phổ hồng ngoại qua các dải bước sóng SPHEREx.")

    sample_shape = all_frames[0]["image"].shape
    sc1, sc2, sc3 = st.columns(3)
    px_x = sc1.number_input("Pixel X", min_value=0, max_value=sample_shape[1] - 1, value=sample_shape[1] // 2)
    px_y = sc2.number_input("Pixel Y", min_value=0, max_value=sample_shape[0] - 1, value=sample_shape[0] // 2)
    aperture = sc3.slider("Aperture Radius (pixels)", 1, 5, 2)

    # Tính tổng thông lượng (Aperture Photometry) qua toàn bộ frames
    wls, mjds, fluxes = [], [], []
    for f in all_frames:
        im = f["image"]
        y_min, y_max = max(0, px_y - aperture), min(im.shape[0], px_y + aperture + 1)
        x_min, x_max = max(0, px_x - aperture), min(im.shape[1], px_x + aperture + 1)
        flux_val = float(np.sum(im[y_min:y_max, x_min:x_max]))
        wls.append(f["wavelength_um"])
        mjds.append(f["mjd"])
        fluxes.append(flux_val)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
        order_wl = np.argsort(wls)
        ax_spec.plot(np.array(wls)[order_wl], np.array(fluxes)[order_wl], marker="o", linestyle="-", color="#ff6b6b")
        ax_spec.set_xlabel("Wavelength (µm)")
        ax_spec.set_ylabel("Integrated Aperture Flux")
        ax_spec.set_title(f"Infrared Spectrum at Pixel ({px_x}, {px_y})")
        ax_spec.grid(True, alpha=0.3)
        st.pyplot(fig_spec, use_container_width=True)
        plt.close(fig_spec)

    with chart_col2:
        fig_lc, ax_lc = plt.subplots(figsize=(6, 4))
        order_t = np.argsort(mjds)
        ax_lc.plot(np.array(mjds)[order_t], np.array(fluxes)[order_t], marker="s", linestyle="--", color="#4ecdc4")
        ax_lc.set_xlabel("Observation Date (MJD)")
        ax_lc.set_ylabel("Integrated Aperture Flux")
        ax_lc.set_title(f"Temporal Light Curve at Pixel ({px_x}, {px_y})")
        ax_lc.grid(True, alpha=0.3)
        st.pyplot(fig_lc, use_container_width=True)
        plt.close(fig_lc)

# ------------------------------------------------------------------------------
# TAB 4: NASA ACKNOWLEDGEMENT (BẮT BUỘC THEO QUY ĐỊNH IRSA)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📜 NASA / IPAC IRSA Data Acknowledgement")
    st.info(
        '"This publication makes use of data products from the Spectro-Photometer for the History of the Universe, '
        "Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet Propulsion Laboratory "
        "and the California Institute of Technology, and is funded by the National Aeronautics and Space Administration.\""
    )
    st.markdown(
        """
        ### Key Capabilities Built for NASA Space Apps Challenge 2026:
        * **Multi-Epoch Time-Lapse Blinker:** Smoothly animates SPHEREx infrared cutouts to reveal proper motion of brown dwarfs, stars, and Planet X candidates.
        * **Detector Array Filtering (`D1`–`D6`):** Separates SPHEREx's 102 spectral channels across its 6 linear variable filter arrays.
        * **Difference Imaging & Automated Anomaly Detection:** Subtracts epochs (`T2 - T1`) and highlights moving/variable candidates.
        * **On-the-Fly Spectrophotometry:** Extracts infrared spectral energy distributions (0.75–5.0 µm) and light curves at any user-selected pixel.
        * **Shareable Coordinate URLs:** Supports `?ra=...&dec=...` query parameters for direct embedding.
        """
    )