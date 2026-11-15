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

# ==============================================================================
# 0. CẤU HÌNH ĐA NGÔN NGỮ (VI / EN)
# ==============================================================================
qp = st.query_params
if "lang" not in st.session_state:
    _q_lang = qp.get("lang", "vi")
    st.session_state.lang = _q_lang if _q_lang in ("vi", "en") else "vi"

TEXTS = {
    "vi": {
        "lang_label": "🌐 Ngôn ngữ / Language",
        "err_import_pf": (
            "Không import được Process_fits_data.py: {etype}: {exc}\n\n"
            "Kiểm tra requirements.txt (cần opencv-python-headless, photutils, scipy, astropy)."
        ),
        "err_old_pf": "Process_fits_data.py là bản cũ (thiếu hàm `prepare_stack_arrays`). Hãy thay bằng bản mới.",
        "custom_target": "Tùy chọn (nhập RA/Dec)",
        "target_descs": {
            "M101": "Mục tiêu thử nghiệm mặc định",
            "Barnards_Star": "Chuyển động riêng lớn nhất bầu trời (~10.3″/năm)",
            "WISE_0855": "Lùn nâu loại Y, cách ~7.2 năm ánh sáng",
            "Epsilon_Indi_B": "Cặp sao lùn nâu quay quanh Epsilon Indi",
            "RW_Aurigae": "Sao biến quang T Tauri",
            "R_Draconis": "Sao biến quang chu kỳ dài (Mira)",
            "CUSTOM": "Toạ độ do người dùng nhập",
        },
        "spinner_read_fits": "Đang đọc tệp FITS…",
        "target_select": "🎯 Mục tiêu",
        "source_radio": "📂 Nguồn dữ liệu",
        "sources": {
            "spherex": "🛰️ SPHEREx (IRSA)",
            "wise": "📡 WISE / NEOWISE",
            "user": "📁 FITS của bạn",
            "demo": "🧪 Demo giả lập (offline)",
        },
        "spx_band": "Bandpass (detector)",
        "spx_size": "Kích thước cutout (°)",
        "spx_max_cut": "Số epoch tối đa (rải đều theo thời gian)",
        "spx_dl_opt": "Tuỳ chọn tải (khi IRSA quá tải)",
        "spx_workers": "Số luồng tải",
        "spx_workers_help": "IRSA đang quá tải: ít luồng + retry thường ổn định hơn.",
        "spx_limit": "Giới hạn thời gian tải (phút)",
        "spx_btn_fetch": "⬇️ Tải từ IRSA",
        "btn_refetch": "🔄 Tải lại",
        "spx_use_local": "Dùng tệp MEF có sẵn / tải lên",
        "spx_local_path": "Đường dẫn MEF",
        "spx_upload_mef": "Hoặc tải lên MEF (.fits)",
        "spx_err_mef": "Không đọc được MEF: {exc}",
        "spx_no_data": "Chưa có dữ liệu cho cấu hình này — bấm **Tải từ IRSA**.",
        "wise_band": "Kênh",
        "wise_size": "Kích thước cutout (pixel)",
        "wise_pix": "Pixel scale (″/px)",
        "wise_btn_fetch": "⬇️ Tải WISE",
        "wise_err_mef": "Không đọc được MEF WISE: {exc}",
        "wise_no_data": "Chưa có dữ liệu WISE cho toạ độ này — bấm **Tải WISE**.",
        "user_upload": "Tải lên FITS (nhiều file 2D, hoặc 1 MEF của SPHEREx.py)",
        "user_spinner": "Đang đọc FITS…",
        "demo_caption": "7 epoch giả lập (2010–2024) với 1 vật thể chuyển động ~11.5″/năm.",
        "param_expander": "⚙️ Tham số phân tích",
        "param_sigma": "Ngưỡng phát hiện (σ)",
        "param_vmax": "Chuyển động tối đa (″/năm)",
        "param_tol": "Dung sai liên kết (px)",
        "param_min_ep": "Số epoch tối thiểu / quỹ đạo",
        "param_max_det": "Số phát hiện tối đa / khung",
        "param_min_clean": "Bỏ epoch có < x% pixel sạch",
        "param_psf": "Đồng bộ PSF",
        "param_scale": "Chuẩn hoá thông lượng",
        "param_exact": "reproject_exact (chậm, bảo toàn thông lượng)",
        "fetch_status_spx": "Đang tải dữ liệu SPHEREx từ IRSA…",
        "fetch_status_wise": "Đang tải dữ liệu WISE từ Legacy Survey…",
        "fetch_start": "Khởi động…",
        "fetch_prog": "Đã xử lý {done}/{total} yêu cầu",
        "fetch_tap": "Truy vấn danh sách cutout (IRSA TAP)…",
        "fetch_fail": "Tải thất bại",
        "fetch_empty_title": "Không tải được dữ liệu",
        "fetch_empty_msg": (
            "Không tải được đủ dữ liệu (máy chủ có thể đang quá tải). Thử lại sau, giảm số epoch/luồng, "
            "hoặc chuyển sang nguồn **WISE** / **Demo**."
        ),
        "fetch_done": "Hoàn tất sau {sec:.0f} giây",
        "no_data_info": (
            "👈 Chọn nguồn dữ liệu ở thanh bên và tải dữ liệu. Nếu SPHEREx/IRSA đang quá tải, dùng **WISE / NEOWISE** "
            "(baseline 2010–2024, rất tốt để thấy chuyển động riêng), tải FITS của bạn, hoặc thử **Demo giả lập**."
        ),
        "all_detectors": "Tất cả",
        "det_select": "📡 Detector dùng để phân tích",
        "det_help": "Mỗi detector có PSF/phông nền khác nhau; nên phân tích từng detector.",
        "warn_few_epochs": (
            "Chỉ có {n} epoch đạt điều kiện (cần ≥ 3). Hãy giảm 'Bỏ epoch có < x% pixel sạch', "
            "chọn detector khác, hoặc tải thêm epoch."
        ),
        "col_clean_px": "Pixel sạch (%)",
        "warn_mjd_approx": (
            "⚠️ Một số epoch không có MJD trong header: dùng ngày xấp xỉ (giữa năm theo tên/năm của lớp dữ liệu). "
            "Vận tốc chỉ chính xác ở mức ~±0.5 năm."
        ),
        "spinner_analysis": "Đang đồng trục, đồng bộ PSF/thông lượng và tìm vật thể chuyển động…",
        "err_analysis": "Lỗi khi phân tích: {etype}: {exc}",
        "err_analysis_tip": "Gợi ý: kiểm tra WCS của các khung, giảm 'Bỏ epoch có < x% pixel sạch', hoặc tắt PSF/thông lượng.",
        "metric_epochs": "Số epoch dùng",
        "metric_baseline": "Baseline",
        "unit_yr": "năm",
        "metric_pscale": "Pixel scale",
        "metric_fwhm": "FWHM đồng bộ",
        "metric_cand": "Ứng viên chuyển động",
        "cap_short_baseline": (
            "ℹ️ Baseline < 6 tháng: chỉ phát hiện được vật thể chuyển động rất nhanh. WISE (2010–2024) cho "
            "baseline dài hơn nhiều."
        ),
        "spinner_frames": "Đang dựng khung hình…",
        "df_moving": "Chuyển động",
        "df_epochs": "Số epoch",
        "df_speed_px": "Tốc độ (px/năm)",
        "df_pm_tot": "μ tổng (″/năm)",
        "df_mu_ra": "μα* (″/năm)",
        "df_mu_dec": "μδ (″/năm)",
        "df_pa": "PA (° Bắc→Đông)",
        "df_ra0": "RA (°) @epoch đầu",
        "df_dec0": "Dec (°) @epoch đầu",
        "df_snr": "SNR TB",
        "tabs": [
            "🎬 Blinker",
            "🎯 Ứng viên & vận tốc",
            "➖ Trừ ảnh & Stack",
            "📈 Phổ & Đường cong sáng",
            "🔬 Độ nhạy (Injection)",
            "ℹ️ Thông tin",
        ],
        "blink_stretch": "Kéo giãn",
        "blink_cmap": "Bảng màu",
        "blink_per_frame": "Chuẩn hoá từng khung",
        "blink_speed": "Tốc độ (ms/khung)",
        "blink_show_marks": "🔴 Khoanh vị trí dự đoán của ứng viên chuyển động",
        "blink_n_show": "Số ứng viên hiển thị",
        "blink_no_cand": "Chưa có ứng viên chuyển động — thử hạ ngưỡng σ hoặc tăng dung sai.",
        "blink_det_count": "{n} phát hiện",
        "blink_caption": (
            "Ảnh đã đồng trục WCS, đồng bộ PSF, chuẩn hoá thông lượng và trừ nền. Vật thể chuyển động sẽ dịch "
            "chuyển thẳng hàng so với nền sao đứng yên."
        ),
        "col_mjd": "Ngày (MJD)",
        "col_detections": "Phát hiện",
        "cand_subheader": "Quỹ đạo chuyển động thẳng đều (tracklet)",
        "cand_none": (
            "Không liên kết được quỹ đạo nào. Thử: hạ ngưỡng σ, tăng dung sai (px) hoặc 'Chuyển động tối đa', "
            "giảm số epoch tối thiểu, hoặc bỏ lọc epoch kém."
        ),
        "cand_only_mv": "Chỉ hiện ứng viên chuyển động",
        "cand_no_sig": "Không có ứng viên chuyển động đáng kể (chỉ có quỹ đạo gần như đứng yên / nhiễu).",
        "cand_dl_csv": "⬇️ Tải CSV",
        "cand_sel": "Xem chi tiết quỹ đạo",
        "cand_plot_title": "Quỹ đạo #{sel} trên nền tĩnh (màu = epoch)",
        "cand_xlabel": "x (px, lưới tham chiếu)",
        "cand_speed_lbl": "Tốc độ",
        "cand_pa_lbl": "Hướng (PA):",
        "cand_pos0_lbl": "Vị trí @epoch đầu:",
        "cand_stats": "Xuất hiện ở {n}/{nf} epoch, RMS khớp tuyến tính = {rms:.2f} px, SNR trung bình = {snr:.1f}",
        "cand_warn": (
            "Đây là *ứng viên* chuyển động đều. Cần đối chiếu catalog (SIMBAD/AllWISE/Gaia) và kiểm tra "
            "bằng mắt ở tab Blinker trước khi kết luận; thị sai năm cũng gây dịch chuyển nhỏ."
        ),
        "cand_cutout_hdr": "**Cutout quanh vị trí dự đoán theo từng epoch**",
        "diff_t1": "Epoch tham chiếu (T1)",
        "diff_t2": "Epoch đích (T2)",
        "diff_static": "Nền tĩnh (median stack)",
        "diff_title": "T2 − T1 | {n} phát hiện (T2)",
        "diff_caption": "Vòng xanh: nguồn vượt ngưỡng σ trên ảnh (T2 − nền tĩnh) sau khi đã loại sao bão hoà/viền.",
        "phot_subheader": "Phổ & đường cong sáng (quang trắc khẩu độ trên ảnh gốc)",
        "phot_caption": "Dùng ảnh gốc (chưa chuẩn hoá thông lượng) và chuyển vị trí qua WCS của từng epoch.",
        "phot_pos": "Vị trí lấy mẫu",
        "phot_fixed": "Cố định (toạ độ pixel)",
        "phot_follow": "Theo ứng viên #{id}",
        "phot_x": "X (lưới tham chiếu)",
        "phot_y": "Y (lưới tham chiếu)",
        "phot_rad": "Bán kính khẩu độ (px)",
        "phot_wl": "Bước sóng (µm)",
        "phot_flux": "Thông lượng khẩu độ (đơn vị ảnh)",
        "phot_spec_title": "Phổ hồng ngoại",
        "phot_lc_title": "Đường cong sáng",
        "phot_no_spec": "Nguồn này chỉ có một kênh phổ nên không vẽ phổ.",
        "sens_subheader": "Injection–Recovery: đo độ nhạy thật của pipeline",
        "sens_caption": (
            "Bơm nguồn Gaussian giả (SNR & tốc độ khác nhau) vào chuỗi ảnh rồi đo tỷ lệ phát hiện lại. "
            "Chạy lưới rút gọn để kịp thời gian."
        ),
        "sens_btn": "▶️ Chạy mô phỏng",
        "sens_spinner": "Đang mô phỏng (có thể mất 0.5–2 phút)…",
        "sens_snr": "SNR đỉnh bơm vào",
        "sens_comp": "Tỷ lệ phát hiện lại",
        "sens_fp": "Dương tính giả / khung",
        "about_subheader": "📜 Ghi nhận dữ liệu",
        "about_info": (
            "Ứng dụng dùng dữ liệu SPHEREx (JPL/Caltech, NASA) qua NASA/IPAC IRSA; ảnh WISE/NEOWISE qua Legacy Survey "
            "(unWISE co-adds). Khi công bố, hãy dùng đúng câu acknowledgement chính thức của IRSA/SPHEREx và "
            "WISE/NEOWISE."
        ),
        "about_md": """
**Pipeline:** `SPHEREx.py` / `ImageGenerate_WISE.py` → `data_io.py` → `Process_fits_data.py`
(đồng trục WCS, đồng bộ PSF, chuẩn hoá thông lượng, trừ nền tĩnh, phát hiện) → `tracking.py`
(liên kết quỹ đạo thẳng đều) → `viz.py` (blinker).

**Giới hạn cần nêu rõ:** baseline SPHEREx chỉ ~1–2 năm nên chỉ nhạy với vật thể chuyển động riêng lớn; chuyển động
rất chậm (kiểu Planet X thực sự) cần baseline dài như WISE/NEOWISE; thị sai và quỹ đạo vệ tinh chưa được hiệu chỉnh.
""",
    },
    "en": {
        "lang_label": "🌐 Ngôn ngữ / Language",
        "err_import_pf": (
            "Failed to import Process_fits_data.py: {etype}: {exc}\n\n"
            "Check requirements.txt (requires opencv-python-headless, photutils, scipy, astropy)."
        ),
        "err_old_pf": "Process_fits_data.py is outdated (missing `prepare_stack_arrays`). Please replace with the latest version.",
        "custom_target": "Custom (enter RA/Dec)",
        "target_descs": {
            "M101": "Default test target",
            "Barnards_Star": "Highest proper motion in the sky (~10.3″/yr)",
            "WISE_0855": "Y-type brown dwarf, ~7.2 light-years away",
            "Epsilon_Indi_B": "Binary brown dwarf orbiting Epsilon Indi",
            "RW_Aurigae": "T Tauri variable star",
            "R_Draconis": "Long-period Mira variable star",
            "CUSTOM": "User-specified coordinates",
        },
        "spinner_read_fits": "Reading FITS files…",
        "target_select": "🎯 Target",
        "source_radio": "📂 Data source",
        "sources": {
            "spherex": "🛰️ SPHEREx (IRSA)",
            "wise": "📡 WISE / NEOWISE",
            "user": "📁 Your FITS files",
            "demo": "🧪 Simulated Demo (offline)",
        },
        "spx_band": "Bandpass (detector)",
        "spx_size": "Cutout size (°)",
        "spx_max_cut": "Max epochs (evenly spaced in time)",
        "spx_dl_opt": "Download options (when IRSA is busy)",
        "spx_workers": "Download threads",
        "spx_workers_help": "When IRSA is busy: fewer threads + retries are usually more stable.",
        "spx_limit": "Download timeout (minutes)",
        "spx_btn_fetch": "⬇️ Fetch from IRSA",
        "btn_refetch": "🔄 Re-fetch",
        "spx_use_local": "Use existing / uploaded MEF file",
        "spx_local_path": "MEF file path",
        "spx_upload_mef": "Or upload MEF (.fits)",
        "spx_err_mef": "Failed to read MEF: {exc}",
        "spx_no_data": "No data for this configuration yet — click **Fetch from IRSA**.",
        "wise_band": "Band",
        "wise_size": "Cutout size (pixels)",
        "wise_pix": "Pixel scale (″/px)",
        "wise_btn_fetch": "⬇️ Fetch WISE",
        "wise_err_mef": "Failed to read WISE MEF: {exc}",
        "wise_no_data": "No WISE data for these coordinates yet — click **Fetch WISE**.",
        "user_upload": "Upload FITS (multiple 2D files, or 1 MEF from SPHEREx.py)",
        "user_spinner": "Reading FITS…",
        "demo_caption": "7 simulated epochs (2010–2024) with 1 moving object (~11.5″/yr).",
        "param_expander": "⚙️ Analysis parameters",
        "param_sigma": "Detection threshold (σ)",
        "param_vmax": "Max proper motion (″/yr)",
        "param_tol": "Linking tolerance (px)",
        "param_min_ep": "Min epochs / tracklet",
        "param_max_det": "Max detections / frame",
        "param_min_clean": "Drop epochs with < x% clean pixels",
        "param_psf": "PSF matching",
        "param_scale": "Flux normalization",
        "param_exact": "reproject_exact (slow, flux-conserving)",
        "fetch_status_spx": "Fetching SPHEREx data from IRSA…",
        "fetch_status_wise": "Fetching WISE data from Legacy Survey…",
        "fetch_start": "Starting…",
        "fetch_prog": "Processed {done}/{total} requests",
        "fetch_tap": "Querying cutout list (IRSA TAP)…",
        "fetch_fail": "Download failed",
        "fetch_empty_title": "No data retrieved",
        "fetch_empty_msg": (
            "Could not retrieve enough data (server may be overloaded). Try again later, reduce epochs/threads, "
            "or switch to **WISE** / **Demo**."
        ),
        "fetch_done": "Completed in {sec:.0f}s",
        "no_data_info": (
            "👈 Select a data source in the sidebar and load data. If SPHEREx/IRSA is busy, try **WISE / NEOWISE** "
            "(2010–2024 baseline, ideal for proper motion), upload your own FITS, or try **Simulated Demo**."
        ),
        "all_detectors": "All",
        "det_select": "📡 Detector for analysis",
        "det_help": "Each detector has different PSF/background; analyzing per detector is recommended.",
        "warn_few_epochs": (
            "Only {n} epochs meet the criteria (need ≥ 3). Lower 'Drop epochs with < x% clean pixels', "
            "choose another detector, or fetch more epochs."
        ),
        "col_clean_px": "Clean pixels (%)",
        "warn_mjd_approx": (
            "⚠️ Some epochs lack MJD in their headers: using approximate dates (mid-year from layer name/year). "
            "Velocities are only accurate to ~±0.5 yr."
        ),
        "spinner_analysis": "Aligning WCS, matching PSF/flux, and searching for moving objects…",
        "err_analysis": "Analysis error: {etype}: {exc}",
        "err_analysis_tip": "Tip: check frame WCS, lower 'Drop epochs with < x% clean pixels', or disable PSF/flux matching.",
        "metric_epochs": "Epochs used",
        "metric_baseline": "Baseline",
        "unit_yr": "yr",
        "metric_pscale": "Pixel scale",
        "metric_fwhm": "Matched FWHM",
        "metric_cand": "Moving candidates",
        "cap_short_baseline": (
            "ℹ️ Baseline < 6 months: only very fast-moving objects can be detected. WISE (2010–2024) provides a "
            "much longer baseline."
        ),
        "spinner_frames": "Building frames…",
        "df_moving": "Moving",
        "df_epochs": "Epochs",
        "df_speed_px": "Speed (px/yr)",
        "df_pm_tot": "Total μ (″/yr)",
        "df_mu_ra": "μα* (″/yr)",
        "df_mu_dec": "μδ (″/yr)",
        "df_pa": "PA (° N→E)",
        "df_ra0": "RA (°) @1st epoch",
        "df_dec0": "Dec (°) @1st epoch",
        "df_snr": "Mean SNR",
        "tabs": [
            "🎬 Blinker",
            "🎯 Candidates & Velocity",
            "➖ Image Diff & Stack",
            "📈 Spectrum & Light Curve",
            "🔬 Sensitivity (Injection)",
            "ℹ️ About",
        ],
        "blink_stretch": "Stretch",
        "blink_cmap": "Colormap",
        "blink_per_frame": "Per-frame stretch",
        "blink_speed": "Speed (ms/frame)",
        "blink_show_marks": "🔴 Circle predicted positions of moving candidates",
        "blink_n_show": "Number of candidates shown",
        "blink_no_cand": "No moving candidates yet — try lowering the σ threshold or increasing tolerance.",
        "blink_det_count": "{n} detections",
        "blink_caption": (
            "Images are WCS-aligned, PSF-matched, flux-normalized, and background-subtracted. Moving objects shift "
            "linearly relative to stationary background stars."
        ),
        "col_mjd": "Date (MJD)",
        "col_detections": "Detections",
        "cand_subheader": "Linear uniform-motion tracklets",
        "cand_none": (
            "No tracklets linked. Try: lowering σ threshold, increasing tolerance (px) or 'Max proper motion', "
            "reducing min epochs, or relaxing clean-pixel filtering."
        ),
        "cand_only_mv": "Show moving candidates only",
        "cand_no_sig": "No significant moving candidates found (only nearly stationary tracklets or noise).",
        "cand_dl_csv": "⬇️ Download CSV",
        "cand_sel": "Inspect tracklet details",
        "cand_plot_title": "Tracklet #{sel} on static sky (color = epoch)",
        "cand_xlabel": "x (px, reference grid)",
        "cand_speed_lbl": "Speed",
        "cand_pa_lbl": "Direction (PA):",
        "cand_pos0_lbl": "Position @1st epoch:",
        "cand_stats": "Detected in {n}/{nf} epochs, linear fit RMS = {rms:.2f} px, mean SNR = {snr:.1f}",
        "cand_warn": (
            "This is a linear-motion *candidate*. Cross-match with catalogs (SIMBAD/AllWISE/Gaia) and visually inspect "
            "in the Blinker tab before concluding; annual parallax also causes small shifts."
        ),
        "cand_cutout_hdr": "**Cutouts around predicted positions per epoch**",
        "diff_t1": "Reference epoch (T1)",
        "diff_t2": "Target epoch (T2)",
        "diff_static": "Static sky (median stack)",
        "diff_title": "T2 − T1 | {n} detections (T2)",
        "diff_caption": "Green circles: sources above σ threshold on (T2 − static sky) after masking saturated/edge pixels.",
        "phot_subheader": "Spectrum & light curve (aperture photometry on raw images)",
        "phot_caption": "Uses raw images (before flux normalization) and transforms coordinates via each epoch's WCS.",
        "phot_pos": "Sampling position",
        "phot_fixed": "Fixed (pixel coordinates)",
        "phot_follow": "Follow candidate #{id}",
        "phot_x": "X (reference grid)",
        "phot_y": "Y (reference grid)",
        "phot_rad": "Aperture radius (px)",
        "phot_wl": "Wavelength (µm)",
        "phot_flux": "Aperture flux (image units)",
        "phot_spec_title": "Infrared Spectrum",
        "phot_lc_title": "Light Curve",
        "phot_no_spec": "This source only has a single spectral band, so no spectrum is plotted.",
        "sens_subheader": "Injection–Recovery: measure true pipeline sensitivity",
        "sens_caption": (
            "Injects synthetic Gaussian sources (varying SNR & speed) into the stack and measures recovery rate. "
            "Uses a compact grid for speed."
        ),
        "sens_btn": "▶️ Run simulation",
        "sens_spinner": "Running simulation (may take 0.5–2 min)…",
        "sens_snr": "Injected peak SNR",
        "sens_comp": "Recovery completeness",
        "sens_fp": "False positives / frame",
        "about_subheader": "📜 Data Acknowledgments",
        "about_info": (
            "This app uses SPHEREx data (JPL/Caltech, NASA) via NASA/IPAC IRSA, and WISE/NEOWISE imagery via Legacy Survey "
            "(unWISE co-adds). When publishing, please include the official IRSA/SPHEREx and WISE/NEOWISE acknowledgment statements."
        ),
        "about_md": """
**Pipeline:** `SPHEREx.py` / `ImageGenerate_WISE.py` → `data_io.py` → `Process_fits_data.py`
(WCS alignment, PSF matching, flux normalization, static background subtraction, detection) → `tracking.py`
(linear uniform-motion tracklet linking) → `viz.py` (blinker).

**Key limitations:** The SPHEREx baseline is only ~1–2 yr, making it sensitive primarily to high proper-motion objects;
very slow motion (such as a true Planet X) requires longer baselines like WISE/NEOWISE; annual parallax and spacecraft
orbit effects are not corrected.
""",
    },
}

T = TEXTS[st.session_state.lang]

import data_io  # noqa: E402
import tracking  # noqa: E402
import viz  # noqa: E402

try:
    import Process_fits_data as pf  # noqa: E402
except Exception as exc:  # thiếu cv2 / photutils / scipy ...
    st.error(T["err_import_pf"].format(etype=type(exc).__name__, exc=exc))
    st.stop()
if not hasattr(pf, "prepare_stack_arrays"):
    st.error(T["err_old_pf"])
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
    "M101 (Pinwheel Galaxy - test)": dict(ra=210.80227, dec=54.34895, prefix="M101", desc_key="M101"),
    "Barnard's Star (Fast Proper Motion)": dict(ra=269.45210, dec=4.69340, prefix="Barnards_Star", desc_key="Barnards_Star"),
    "WISE 0855 (Coldest Brown Dwarf)": dict(ra=133.79510, dec=-7.24510, prefix="WISE_0855", desc_key="WISE_0855"),
    "Epsilon Indi B (Binary Brown Dwarf)": dict(ra=331.04380, dec=-56.78280, prefix="Epsilon_Indi_B", desc_key="Epsilon_Indi_B"),
    "RW Aurigae (Variable Star)": dict(ra=76.95650, dec=30.40140, prefix="RW_Aurigae", desc_key="RW_Aurigae"),
    "R Draconis (Mira Variable Star)": dict(ra=248.16500, dec=66.75500, prefix="R_Draconis", desc_key="R_Draconis"),
    CUSTOM: dict(ra=None, dec=None, prefix=None, desc_key="CUSTOM"),
}
PALETTE = [(0, 255, 0), (255, 80, 80), (80, 200, 255), (255, 220, 0), (255, 0, 255), (0, 255, 200)]

SOURCE_KEYS = ["spherex", "wise", "user", "demo"]


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
# 2. TRẠNG THÁI BAN ĐẦU + URL QUERY PARAMS (link chia sẻ ?lang=&target=&ra=&dec=)
# ==============================================================================
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
@st.cache_resource(show_spinner=T["spinner_read_fits"], max_entries=8)
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
            mjd=data_io.mjd_from_year(yr), wl=3.4, detector="Demo",
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
    st.radio(
        T["lang_label"],
        ["vi", "en"],
        format_func=lambda x: "🇻🇳 Tiếng Việt" if x == "vi" else "🇬🇧 English",
        horizontal=True,
        key="lang",
    )
    T = TEXTS[st.session_state.lang]

    st.title("🔭 Planet X Hunter")
    st.caption("NASA Space Apps Challenge — SPHEREx & WISE")

    st.selectbox(
        T["target_select"],
        list(TARGETS),
        format_func=lambda k: T["custom_target"] if k == CUSTOM else k,
        key="target_sel",
        on_change=_on_target_change,
    )
    c1, c2 = st.columns(2)
    c1.number_input("RA (°)", min_value=0.0, max_value=360.0, step=0.001, format="%.5f", key="ra")
    c2.number_input("Dec (°)", min_value=-90.0, max_value=90.0, step=0.001, format="%.5f", key="dec")
    ra, dec = float(st.session_state.ra), float(st.session_state.dec)
    st.query_params.update({
        "lang": st.session_state.lang,
        "target": st.session_state.target_sel,
        "ra": f"{ra:.5f}",
        "dec": f"{dec:.5f}",
    })
    okey = object_key(ra, dec)

    st.divider()
    source = st.radio(
        T["source_radio"],
        SOURCE_KEYS,
        format_func=lambda k: T["sources"][k],
        key="source",
    )

    # ---------------- SPHEREx ----------------
    if source == "spherex":
        band = st.selectbox(T["spx_band"], [f"SPHEREx-D{i}" for i in range(1, 7)], index=1)
        size_deg = st.slider(T["spx_size"], 0.02, 0.30, 0.10, 0.01)
        max_cut = st.slider(T["spx_max_cut"], 4, 40, 12)
        with st.expander(T["spx_dl_opt"]):
            workers = st.slider(T["spx_workers"], 1, 8, 3, help=T["spx_workers_help"])
            limit_min = st.slider(T["spx_limit"], 1, 20, 6)
        path = spx_cache_path(ra, dec, band, size_deg, max_cut)
        b1, b2 = st.columns(2)
        go = b1.button(T["spx_btn_fetch"])
        force = b2.button(T["btn_refetch"])
        if go or force:
            fetch_req = dict(kind="spherex", ra=ra, dec=dec, band=band, size=size_deg, max_cut=max_cut,
                             workers=workers, limit=limit_min, path=path, force=force)
        use_local = st.checkbox(T["spx_use_local"])
        if use_local:
            default_local = os.path.join(APP_DIR, "spherex_cutouts_mef.fits")
            local_path = st.text_input(T["spx_local_path"], value=default_local)
            up = st.file_uploader(T["spx_upload_mef"], type=["fits", "fit"], key="mef_upload")
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
                st.error(T["spx_err_mef"].format(exc=exc))
        else:
            st.caption(T["spx_no_data"])

    # ---------------- WISE ----------------
    elif source == "wise":
        wband = st.radio(T["wise_band"], ["W1", "W2"], horizontal=True, help="W1 = 3.4 µm, W2 = 4.6 µm")
        wsize = st.slider(T["wise_size"], 100, 512, 200, 20)
        wpix = st.slider(T["wise_pix"], 1.0, 6.0, 2.75, 0.25)
        wname = okey if (wsize == 200 and abs(wpix - 2.75) < 1e-6) else f"{okey}_s{wsize}_p{wpix:g}"
        fits_dir, mef = wise_paths(wname)
        b1, b2 = st.columns(2)
        go = b1.button(T["wise_btn_fetch"])
        force = b2.button(T["btn_refetch"])
        if go or force:
            fetch_req = dict(kind="wise", ra=ra, dec=dec, name=wname, size=wsize, pix=wpix, force=force)
        if os.path.exists(mef):
            try:
                ds = dict(sig=f"wise|{mef}|{_mtime(mef)}|{wband}", source="wise",
                          epochs=_load("wise_mef", mef, _mtime(mef), wband))
            except Exception as exc:
                st.error(T["wise_err_mef"].format(exc=exc))
        elif os.path.isdir(fits_dir) and wband == "W1":
            ds = dict(sig=f"wisef|{fits_dir}|{_mtime(fits_dir)}", source="wise",
                      epochs=_load("wise_files", fits_dir, _mtime(fits_dir)))
        else:
            st.caption(T["wise_no_data"])

    # ---------------- Upload ----------------
    elif source == "user":
        ups = st.file_uploader(T["user_upload"], type=["fits", "fit", "fz"], accept_multiple_files=True)
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
                with st.spinner(T["user_spinner"]):
                    st.session_state.user_epochs = data_io.load_user_fits(paths)
                st.session_state.user_sig = sig
            ds = dict(sig=sig, source="user", epochs=st.session_state.get("user_epochs", []))

    # ---------------- Demo ----------------
    else:
        if "demo_epochs" not in st.session_state:
            st.session_state.demo_epochs = make_demo_epochs()
        ds = dict(sig="demo", source="demo", epochs=st.session_state.demo_epochs)
        st.caption(T["demo_caption"])

    # ---------------- Tham số phân tích ----------------
    st.divider()
    with st.expander(T["param_expander"], expanded=False):
        sigma = st.slider(T["param_sigma"], 3.0, 9.0, 5.0, 0.5)
        vmax_as = st.number_input(T["param_vmax"], 0.5, 600.0, 20.0, 1.0)
        tol_px = st.slider(T["param_tol"], 0.5, 4.0, 1.5, 0.25)
        min_ep = st.slider(T["param_min_ep"], 3, 12, 3)
        max_det = st.slider(T["param_max_det"], 20, 300, 80, 10)
        min_clean = st.slider(T["param_min_clean"], 0, 95, 50, 5)
        do_psf = st.checkbox(T["param_psf"], True)
        do_scale = st.checkbox(T["param_scale"], True)
        exact = st.checkbox(T["param_exact"], False)

# ==============================================================================
# 5. TẢI DỮ LIỆU (chạy ở vùng chính để thấy tiến độ)
# ==============================================================================
def run_fetch(req) -> bool:
    kind = req["kind"]
    status_lbl = T["fetch_status_spx"] if kind == "spherex" else T["fetch_status_wise"]
    with st.status(status_lbl, expanded=True) as status:
        bar = st.progress(0.0, text=T["fetch_start"])

        def cb(done, total):
            bar.progress(min(1.0, done / max(total, 1)), text=T["fetch_prog"].format(done=done, total=total))

        t0 = time.time()
        try:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            coord = SkyCoord(ra=req["ra"] * u.deg, dec=req["dec"] * u.deg, frame="icrs")
            if kind == "spherex":
                import SPHEREx as spx
                if req["force"] and os.path.exists(req["path"]):
                    os.remove(req["path"])
                st.write(T["fetch_tap"])
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
            status.update(label=T["fetch_fail"], state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return False
        if not out or not os.path.exists(out):
            status.update(label=T["fetch_empty_title"], state="error")
            st.error(T["fetch_empty_msg"])
            return False
        status.update(label=T["fetch_done"].format(sec=time.time() - t0), state="complete")
        return True


if fetch_req and run_fetch(fetch_req):
    st.rerun()

# ==============================================================================
# 6. TIÊU ĐỀ + KIỂM TRA DỮ LIỆU
# ==============================================================================
tinfo = TARGETS[st.session_state.target_sel]
target_title = T["custom_target"] if st.session_state.target_sel == CUSTOM else st.session_state.target_sel
target_desc = T["target_descs"][tinfo["desc_key"]]
st.title(f"🌌 {target_title}")
st.markdown(f"**ICRS:** `RA = {ra:.5f}°`, `Dec = {dec:+.5f}°` — *{target_desc}*")

if ds is None or not ds["epochs"]:
    st.info(T["no_data_info"])
    st.stop()

epochs_all = ds["epochs"]
sig_ds = ds["sig"]

counts: dict[str, int] = {}
for e in epochs_all:
    counts[e["detector"]] = counts.get(e["detector"], 0) + 1
det_opts = ["ALL"] + sorted(counts, key=lambda d: -counts[d])
if len(counts) > 1:
    det_choice = st.sidebar.selectbox(
        T["det_select"],
        det_opts,
        index=1,
        key=f"det_{sig_ds}",
        format_func=lambda x: T["all_detectors"] if x == "ALL" else x,
        help=T["det_help"],
    )
else:
    det_choice = "ALL"

epochs = [e for e in epochs_all
          if (det_choice == "ALL" or e["detector"] == det_choice) and e["clean_ratio"] >= min_clean]
epochs.sort(key=lambda e: (e["mjd"], e["index"]))

if len(epochs) < 3:
    st.warning(T["warn_few_epochs"].format(n=len(epochs)))
    st.dataframe(pd.DataFrame([{"#": e["index"], "MJD": round(e["mjd"], 3), "λ (µm)": round(e["wavelength_um"], 3),
                                "Detector": e["detector"], T["col_clean_px"]: round(e["clean_ratio"], 1)}
                               for e in epochs_all]))
    st.stop()

if any(e["mjd_approx"] for e in epochs):
    st.caption(T["warn_mjd_approx"])

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
    with st.spinner(T["spinner_analysis"]):
        res = run_analysis(sig_an, epochs, P)
except Exception as exc:
    st.error(T["err_analysis"].format(etype=type(exc).__name__, exc=exc))
    st.caption(T["err_analysis_tip"])
    st.stop()

used = [epochs[i] for i in res["keep"]]
nf = len(used)
tracks = res["tracks"]
moving = [t for t in tracks if t["moving"]]
span_yr = float(res["t_yr"].max())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(T["metric_epochs"], f"{nf}/{len(epochs_all)}")
m2.metric(T["metric_baseline"], f"{span_yr:.2f} {T['unit_yr']}")
m3.metric(T["metric_pscale"], f"{res['pscale']:.2f}″/px" if res["pscale"] else "n/a")
m4.metric(T["metric_fwhm"], f"{res['fwhm']:.2f} px")
m5.metric(T["metric_cand"], f"{len(moving)}")
if span_yr < 0.5:
    st.caption(T["cap_short_baseline"])

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


@st.cache_data(show_spinner=T["spinner_frames"], max_entries=6)
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
            "ID": t["id"],
            T["df_moving"]: "✅" if t["moving"] else "—",
            T["df_epochs"]: t["n"],
            T["df_speed_px"]: round(t["speed"], 3),
            T["df_pm_tot"]: None if np.isnan(t["pm_as"]) else round(t["pm_as"], 3),
            T["df_mu_ra"]: None if np.isnan(t["mu_ra"]) else round(t["mu_ra"], 3),
            T["df_mu_dec"]: None if np.isnan(t["mu_dec"]) else round(t["mu_dec"], 3),
            T["df_pa"]: None if np.isnan(t["pa"]) else round(t["pa"], 1),
            T["df_ra0"]: None if np.isnan(t["ra"]) else round(t["ra"], 5),
            T["df_dec0"]: None if np.isnan(t["dec"]) else round(t["dec"], 5),
            "RMS (px)": round(t["rms"], 3),
            T["df_snr"]: round(t["snr"], 1),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 9. TABS
# ==============================================================================
tab_blink, tab_cand, tab_diff, tab_phot, tab_sens, tab_about = st.tabs(T["tabs"])

# ---------------------------------------------------------------- Blinker
with tab_blink:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    mode = c1.selectbox(T["blink_stretch"], ["ZScale", "Asinh", "Log", "Linear"])
    cmap = c2.selectbox(T["blink_cmap"], ["inferno", "viridis", "magma", "gray", "cividis"])
    per_frame = c3.checkbox(T["blink_per_frame"], True)
    crop = c4.slider("FoV (%)", 30, 100, 100, 10)
    speed = c5.slider(T["blink_speed"], 100, 1500, 400, 50)

    show_marks = st.checkbox(T["blink_show_marks"], True)
    marks = [[] for _ in range(nf)]
    if show_marks and moving:
        n_show = st.slider(T["blink_n_show"], 1, min(6, len(moving)), min(3, len(moving)))
        marks = tracklet_marks(res, [t["id"] for t in moving[:n_show]])
    elif show_marks:
        st.caption(T["blink_no_cand"])

    marks_key = tuple(tuple((round(x, 2), round(y, 2), tuple(c)) for x, y, c in m) for m in marks)
    uris, nh, nw = build_blink_frames(f"{sig_an}|{hash(str(P))}", res["stack"], mode, cmap, per_frame, crop, marks_key)
    labels = [f"{e['label']}  |  {T['blink_det_count'].format(n=len(res['dets'][k]))}" for k, e in enumerate(used)]
    components.html(viz.blink_html(uris, labels, interval_ms=speed, max_width=560),
                    height=int(min(560, 560 * nh / nw)) + 120)
    st.caption(T["blink_caption"])
    st.dataframe(pd.DataFrame([{
        "#": k + 1,
        T["col_mjd"]: round(e["mjd"], 3),
        "λ (µm)": None if np.isnan(e["wavelength_um"]) else round(e["wavelength_um"], 3),
        "Detector": e["detector"],
        T["col_clean_px"]: round(e["clean_ratio"], 1),
        T["col_detections"]: len(res["dets"][k]),
    } for k, e in enumerate(used)]), hide_index=True)

# ---------------------------------------------------------------- Ứng viên
with tab_cand:
    st.subheader(T["cand_subheader"])
    if not tracks:
        st.info(T["cand_none"])
    else:
        only_mv = st.checkbox(T["cand_only_mv"], True)
        show_tr = [t for t in tracks if t["moving"]] if only_mv else tracks
        df = tracks_df(show_tr)
        if df.empty:
            st.info(T["cand_no_sig"])
        else:
            st.dataframe(df, hide_index=True)
            st.download_button(T["cand_dl_csv"], df.to_csv(index=False).encode("utf-8"),
                               file_name=f"candidates_{okey}.csv", mime="text/csv")
            sel = st.selectbox(T["cand_sel"], [t["id"] for t in show_tr],
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
                ax.set_title(T["cand_plot_title"].format(sel=sel))
                ax.set_xlabel(T["cand_xlabel"])
                ax.set_ylabel("y (px)")
                st.pyplot(fig)
                plt.close(fig)
            with right:
                yr_u = T["unit_yr"]
                st.metric(
                    T["cand_speed_lbl"],
                    f"{tr['speed']:.3f} px/{yr_u}" + (f" = {tr['pm_as']:.2f}″/{yr_u}" if not np.isnan(tr["pm_as"]) else ""),
                )
                if not np.isnan(tr["pa"]):
                    st.write(
                        f"**{T['cand_pa_lbl']}** {tr['pa']:.1f}°  |  "
                        f"**μα\\*:** {tr['mu_ra']:+.3f}″/{yr_u}  |  **μδ:** {tr['mu_dec']:+.3f}″/{yr_u}"
                    )
                    st.write(f"**{T['cand_pos0_lbl']}** RA = {tr['ra']:.5f}°, Dec = {tr['dec']:+.5f}°")
                st.write(T["cand_stats"].format(n=tr["n"], nf=nf, rms=tr["rms"], snr=tr["snr"]))
                st.warning(T["cand_warn"])
                fig, ax = plt.subplots(1, 2, figsize=(5.5, 3))
                ks = [k for k, _ in tr["members"]]
                xs = [res["dets"][k][m, 0] for k, m in tr["members"]]
                ys = [res["dets"][k][m, 1] for k, m in tr["members"]]
                tt_m = res["t_yr"][ks]
                for a, v, lab, v0, vv in ((ax[0], xs, "x", tr["x0"], tr["vx"]), (ax[1], ys, "y", tr["y0"], tr["vy"])):
                    a.plot(tt_m, v, "o")
                    a.plot(tt, v0 + vv * tt, "r-")
                    a.set_xlabel(f"t ({yr_u})")
                    a.set_ylabel(f"{lab} (px)")
                    a.grid(alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown(T["cand_cutout_hdr"])
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
    i1 = c1.selectbox(T["diff_t1"], range(nf), index=0, format_func=lambda i: used[i]["label"])
    i2 = c2.selectbox(T["diff_t2"], range(nf), index=nf - 1, format_func=lambda i: used[i]["label"])
    smode = c3.selectbox(T["blink_stretch"] + " ", ["Asinh", "ZScale", "Log", "Linear"], key="diff_stretch")
    img1, img2 = res["stack"][i1], res["stack"][i2]
    diff = img2 - img1
    col_a, col_b, col_c, col_d = st.columns(4)
    for col, im, ttl in ((col_a, img1, f"T1: {used[i1]['label'].split('|')[0]}"),
                         (col_b, img2, f"T2: {used[i2]['label'].split('|')[0]}"),
                         (col_c, res["static_sky"], T["diff_static"])):
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
        ax.set_title(T["diff_title"].format(n=len(res["dets"][i2])), fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    st.caption(T["diff_caption"])

# ---------------------------------------------------------------- Quang trắc
with tab_phot:
    st.subheader(T["phot_subheader"])
    st.caption(T["phot_caption"])
    h, w = res["shape"]
    follow_opts = [None] + [t["id"] for t in moving[:10]]
    follow = st.selectbox(
        T["phot_pos"],
        follow_opts,
        format_func=lambda x: T["phot_fixed"] if x is None else T["phot_follow"].format(id=x),
    )
    sc1, sc2, sc3 = st.columns(3)
    px0 = sc1.number_input(T["phot_x"], 0, w - 1, w // 2)
    py0 = sc2.number_input(T["phot_y"], 0, h - 1, h // 2)
    rad = sc3.slider(T["phot_rad"], 1.0, 8.0, 2.5, 0.5)

    ref_epoch = used[res["ref"]]
    ref_wcs = ref_epoch["wcs"]
    if follow is None:
        pos_ref = [(float(px0), float(py0))] * nf
    else:
        tr = next(t for t in tracks if t["id"] == follow)
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
            ax.set_xlabel(T["phot_wl"])
            ax.set_ylabel(T["phot_flux"])
            ax.set_title(T["phot_spec_title"])
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
    with cols[-1]:
        fig, ax = plt.subplots(figsize=(6, 4))
        o = np.argsort(mj)
        ax.plot(mj[o], fl[o], "s--", color="#4ecdc4")
        ax.set_xlabel("MJD")
        ax.set_ylabel(T["phot_flux"])
        ax.set_title(T["phot_lc_title"])
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    if not has_spec:
        st.caption(T["phot_no_spec"])

# ---------------------------------------------------------------- Độ nhạy
with tab_sens:
    st.subheader(T["sens_subheader"])
    st.caption(T["sens_caption"])
    inj_key = f"inj|{sig_an}|{hash(str(P))}"
    if st.button(T["sens_btn"]):
        margin_cap = res["shape"][1] / 2 - 25
        motions = tuple(m for m in (0.0, 1.0, 3.0, 6.0) if 20 + m * (nf - 1) < margin_cap) or (0.0,)
        with st.spinner(T["sens_spinner"]):
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
        ax[0].set_xlabel(T["sens_snr"])
        ax[0].set_ylabel(T["sens_comp"])
        ax[0].set_ylim(-0.02, 1.02)
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].set_xlabel(T["sens_snr"])
        ax[1].set_ylabel(T["sens_fp"])
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------------- About
with tab_about:
    st.subheader(T["about_subheader"])
    st.info(T["about_info"])
    st.markdown(T["about_md"])