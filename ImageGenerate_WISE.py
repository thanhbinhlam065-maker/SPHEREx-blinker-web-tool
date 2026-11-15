import argparse
import concurrent.futures
import http.client
import io
import logging
import os
import time
import urllib.error

import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS

# Tắt cảnh báo WCS phụ từ Astropy tương tự SPHEREx_data_insert.py
logging.getLogger('astropy').setLevel(logging.ERROR)

TRANSIENT = (
    TimeoutError,
    urllib.error.HTTPError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    urllib.error.URLError,
    requests.exceptions.RequestException,
)

# 1. Danh sách mục tiêu (Thiên hà M101 được đặt lên đầu)
TARGETS = {
    "M101": {
        "ra": "14h 03m 12.54s",
        "dec": "+54d 20m 56.2s",
        "desc": "Thiên hà xoắn ốc Chong Chóng (Pinwheel Galaxy)",
    },
    "Barnards_Star": {
        "ra": "17h 57m 48.5s",
        "dec": "+04d 41m 36s",
        "desc": "Di chuyển rất nhanh",
    },
    "Epsilon_Indi_B": {
        "ra": "22h 04m 10.5s",
        "dec": "-56d 46m 58s",
        "desc": "Hệ sao lùn nâu",
    },
    "WISE_0855": {
        "ra": "08h 55m 10.831s",
        "dec": "-07d 14m 42.53s",
        "desc": "Lùn nâu cực nhanh",
    },
    "RW_Aurigae": {
        "ra": "05h 07m 49.57s",
        "dec": "+30d 24m 05.2s",
        "desc": "Sao biến quang",
    },
    "R_Draconis": {
        "ra": "16h 32m 40.2s",
        "dec": "+66d 45m 17.6s",
        "desc": "Sao biến quang",
    },
}

# Danh sách các lớp dữ liệu unWISE trên Legacy Survey
# Gồm giai đoạn khảo sát AllWISE gốc (2010) và 11 năm NEOWISE (2014 - 2024)
UNWISE_LAYERS = {
    "2010": "unwise-allwise",  # Khảo sát gốc AllWISE
    "2014": "unwise-neo1",     # NEOWISE Year 1 (12/2013 - 12/2014)
    "2015": "unwise-neo2",     # NEOWISE Year 2
    "2016": "unwise-neo3",     # NEOWISE Year 3
    "2017": "unwise-neo4",     # NEOWISE Year 4
    "2018": "unwise-neo5",     # NEOWISE Year 5
    "2019": "unwise-neo6",     # NEOWISE Year 6
    "2020": "unwise-neo7",     # NEOWISE Year 7
    "2021": "unwise-neo8",     # NEOWISE Year 8
    "2022": "unwise-neo9",     # NEOWISE Year 9
    "2023": "unwise-neo10",    # NEOWISE Year 10
    "2024": "unwise-neo11",    # NEOWISE Year 11
}

DEFAULT_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def prepare_target_dirs(base_output_dir, target_name):
    """Tự động tạo cấu trúc thư mục riêng cho từng thiên thể."""
    target_dir = os.path.join(base_output_dir, target_name)
    dirs = {
        "root": target_dir,
        "fits": os.path.join(target_dir, "fits"),
        "annotated": os.path.join(target_dir, "annotated"),
        "clean": os.path.join(target_dir, "clean"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def extract_2d_wcs_header(raw_header):
    """Trích xuất header WCS 2D sạch từ header 3D gốc của unWISE cutout."""
    try:
        wcs_2d = WCS(raw_header, naxis=2)
        return wcs_2d.to_header()
    except Exception:
        return fits.Header()


def download_one_cutout(job, retries=3, timeout=30):
    """Tải 1 cutout FITS từ Legacy Survey có cơ chế retry khi lỗi mạng tạm thời."""
    target_name = job["target"]
    year_str = job["year"]
    layer = job["layer"]
    url = job["url"]

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            if response.status_code == 200 and response.content.startswith(b'SIMPLE'):
                fits_bytes = response.content
                with fits.open(io.BytesIO(fits_bytes)) as hdul:
                    raw_header = hdul[0].header.copy()
                    data = hdul[0].data
                    if data is None:
                        return None

                    if data.ndim == 3:
                        w1_data = np.array(data[0], copy=True)
                        w2_data = np.array(data[1], copy=True) if data.shape[0] >= 2 else None
                    else:
                        w1_data = np.array(data, copy=True)
                        w2_data = None

                    if not np.isfinite(w1_data).any():
                        return None

                    wcs_header_2d = extract_2d_wcs_header(raw_header)

                return {
                    "target": target_name,
                    "year": year_str,
                    "layer": layer,
                    "url": url,
                    "fits_bytes": fits_bytes,
                    "w1_data": w1_data,
                    "w2_data": w2_data,
                    "wcs_header": wcs_header_2d,
                }
            elif response.status_code in (429, 500, 502, 503, 504):
                raise urllib.error.HTTPError(
                    url, response.status_code, "Transient HTTP error", response.headers, None
                )
            else:
                return None
        except TRANSIENT:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"  [!] Lỗi bất ngờ với {target_name} ({year_str}): {e}")
            return None

    print(f"  [x] Thất bại sau {retries} lần thử: {target_name} - {year_str}")
    return None


def save_combined_mef(target_name, target_info, coord, records, mef_filename, overwrite=True):
    """Đóng gói toàn bộ các mốc thời gian của 1 thiên thể vào tệp Multi-Extension FITS."""
    os.makedirs(os.path.dirname(mef_filename), exist_ok=True)
    records.sort(key=lambda r: int(r["year"]))

    for idx, rec in enumerate(records, start=1):
        rec["epoch_index"] = idx

    # HDU 0: PrimaryHDU lưu thông tin tổng quan của thiên thể
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["OBJECT"] = (target_name, "Target identifier")
    primary_hdu.header["RA_STR"] = (target_info["ra"], "ICRS Right Ascension")
    primary_hdu.header["DEC_STR"] = (target_info["dec"], "ICRS Declination")
    primary_hdu.header["RA_DEG"] = (round(float(coord.ra.deg), 6), "RA in degrees (ICRS)")
    primary_hdu.header["DEC_DEG"] = (round(float(coord.dec.deg), 6), "Dec in degrees (ICRS)")
    primary_hdu.header["N_EPOCHS"] = (len(records), "Total valid WISE/NEOWISE epochs")

    # HDU 1: Bảng EPOCH_INFO tương tự CUTOUT_INFO trong SPHEREx_data_insert.py
    cols = fits.ColDefs([
        fits.Column(
            name="epoch_index", format="J",
            array=[r["epoch_index"] for r in records], unit=""
        ),
        fits.Column(
            name="year", format="A8",
            array=[r["year"] for r in records], unit="yr"
        ),
        fits.Column(
            name="layer", format="A20",
            array=[r["layer"] for r in records], unit=""
        ),
        fits.Column(
            name="w1_um", format="D",
            array=[3.4 for _ in records], unit="um"
        ),
        fits.Column(
            name="w2_um", format="D",
            array=[4.6 for _ in records], unit="um"
        ),
        fits.Column(
            name="has_w2", format="L",
            array=[r["w2_data"] is not None for r in records], unit=""
        ),
        fits.Column(
            name="access_url", format="A220",
            array=[r["url"] for r in records], unit=""
        ),
    ])
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["EXTNAME"] = "EPOCH_INFO"

    # HDU 2..N: Các extension ảnh W1_{year} và W2_{year}
    image_hdus = []
    for rec in records:
        yr = rec["year"]
        w1_hdr = rec["wcs_header"].copy()
        w1_hdr["YEAR"] = (yr, "Observation epoch year")
        w1_hdr["LAYER"] = (rec["layer"], "Legacy Survey unWISE layer")
        w1_hdr["BAND"] = ("W1", "WISE band 1 (3.4 um)")
        image_hdus.append(fits.ImageHDU(data=rec["w1_data"], header=w1_hdr, name=f"W1_{yr}"))

        if rec["w2_data"] is not None:
            w2_hdr = rec["wcs_header"].copy()
            w2_hdr["YEAR"] = (yr, "Observation epoch year")
            w2_hdr["LAYER"] = (rec["layer"], "Legacy Survey unWISE layer")
            w2_hdr["BAND"] = ("W2", "WISE band 2 (4.6 um)")
            image_hdus.append(fits.ImageHDU(data=rec["w2_data"], header=w2_hdr, name=f"W2_{yr}"))

    hdulist = fits.HDUList([primary_hdu, table_hdu, *image_hdus])
    hdulist.writeto(mef_filename, overwrite=overwrite)
    print(f"  => Đã đóng gói MEF: {mef_filename} ({len(records)} mốc thời gian, {len(hdulist)} HDUs)")


def save_epoch_pngs(img_data, band_label, wl_label, target_name, year_str, dirs):
    """Xuất bộ đôi ảnh PNG (annotated & clean) vào thư mục con của từng thiên thể."""
    finite_vals = img_data[np.isfinite(img_data)]
    if finite_vals.size > 0 and np.nanmin(finite_vals) != np.nanmax(finite_vals):
        norm = ImageNormalize(img_data, interval=ZScaleInterval())
    else:
        norm = None

    suffix = "" if band_label == "W1" else f"_{band_label}"

    # 1. PHIÊN BẢN ANNOTATED (Có trục & Legend)
    annotated_png = os.path.join(
        dirs["annotated"], f"{target_name}_{year_str}{suffix}_annotated.png"
    )
    plt.figure(figsize=(6, 6))
    plt.imshow(img_data, cmap="magma", origin="lower", norm=norm)
    plt.colorbar(label="Pixel Intensity (ZScale)")
    plt.title(f"{target_name} - {year_str} ({band_label} {wl_label})")
    plt.xlabel("RA (pixels)")
    plt.ylabel("DEC (pixels)")
    plt.savefig(annotated_png, bbox_inches="tight", dpi=150)
    plt.close()

    # 2. PHIÊN BẢN CLEAN PNG (Không viền/trục, dùng làm GIF / Blinking)
    clean_png = os.path.join(
        dirs["clean"], f"{target_name}_{year_str}{suffix}_clean.png"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_data, cmap="magma", origin="lower", norm=norm)
    ax.axis("off")
    fig.savefig(clean_png, bbox_inches="tight", pad_inches=0, dpi=150, transparent=True)
    plt.close(fig)


def plot_mef_preview(mef_filename, preview_png_path, max_plots=12, show_window=False):
    """Đọc lại tệp MEF và vẽ lưới trực quan hóa chuỗi thời gian (tối đa 12 mốc)."""
    summary_table = Table.read(mef_filename, hdu="EPOCH_INFO")
    nbr_images = min(max_plots, len(summary_table))
    if nbr_images == 0:
        return

    imgs = []
    years = []
    layers = []
    with fits.open(mef_filename) as hdul:
        for ii in range(nbr_images):
            yr = str(summary_table["year"][ii])
            lyr = str(summary_table["layer"][ii])
            imgs.append(hdul[f"W1_{yr}"].data)
            years.append(yr)
            layers.append(lyr)

    ncols = 4
    nrows = int(np.ceil(nbr_images / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    axs = np.atleast_1d(axs).ravel()
    fig.suptitle(f"{os.path.basename(mef_filename)} — W1 (3.4 µm) Timeline", fontsize=13)

    for ii in range(len(axs)):
        if ii < nbr_images:
            data = imgs[ii]
            finite_vals = data[np.isfinite(data)]
            norm = (
                ImageNormalize(data, interval=ZScaleInterval())
                if (finite_vals.size > 0 and np.nanmin(finite_vals) != np.nanmax(finite_vals))
                else None
            )
            axs[ii].imshow(data, cmap="magma", origin="lower", norm=norm)
            axs[ii].text(
                0.05, 0.06,
                f"{years[ii]} ({layers[ii]})",
                va="bottom", ha="left", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, lw=0),
                transform=axs[ii].transAxes,
            )
            axs[ii].set_xticks([])
            axs[ii].set_yticks([])
        else:
            axs[ii].axis("off")

    plt.tight_layout()
    fig.savefig(preview_png_path, bbox_inches="tight", dpi=150)
    if show_window:
        plt.show()
    plt.close(fig)


def download_wise_target(
    name,
    coord,
    out_dir="ImageGenerate_WISE",
    size=200,
    pixscale=2.75,
    workers=6,
    retries=3,
    progress_cb=None,
):
    """Tải mọi mốc WISE/NEOWISE cho 1 toạ độ bất kỳ (dùng cho app Streamlit, không vẽ PNG).

    Ghi <out_dir>/<name>/fits/<name>_<năm>_WISE.fits và <name>_WISE_mef.fits.
    Trả về đường dẫn MEF, hoặc None nếu không tải được mốc nào.
    """
    dirs = prepare_target_dirs(out_dir, name)
    jobs = []
    for year_str, layer in UNWISE_LAYERS.items():
        url = (
            f"https://www.legacysurvey.org/viewer/fits-cutout?"
            f"ra={coord.ra.deg:.6f}&dec={coord.dec.deg:.6f}&layer={layer}"
            f"&pixscale={pixscale}&size={size}"
        )
        jobs.append({"target": name, "year": year_str, "layer": layer, "url": url})

    records, done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one_cutout, job, retries) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            res = fut.result()
            if res is not None:
                records.append(res)
            if progress_cb:
                progress_cb(done, len(jobs))

    if not records:
        return None
    records.sort(key=lambda r: int(r["year"]))
    for rec in records:
        path = os.path.join(dirs["fits"], f"{name}_{rec['year']}_WISE.fits")
        with open(path, "wb") as f:
            f.write(rec["fits_bytes"])

    info = {"ra": f"{coord.ra.deg:.6f} deg", "dec": f"{coord.dec.deg:.6f} deg"}
    mef = os.path.join(dirs["fits"], f"{name}_WISE_mef.fits")
    save_combined_mef(name, info, coord, records, mef, overwrite=True)
    return mef


def main():
    ap = argparse.ArgumentParser(
        description="Tải dữ liệu WISE/NEOWISE theo từng thiên thể, đóng gói MEF và xuất ảnh PNG."
    )
    ap.add_argument(
        "--targets", nargs="+", default=list(TARGETS.keys()),
        help="Danh sách thiên thể cần chạy (mặc định chạy toàn bộ, bắt đầu từ M101)"
    )
    ap.add_argument(
        "--out-dir", default="ImageGenerate_WISE",
        help="Thư mục gốc chứa các thư mục con của từng thiên thể"
    )
    ap.add_argument(
        "--size", type=int, default=200,
        help="Kích thước khung cắt tính bằng pixel (mặc định: 200; gợi ý 400-512 cho M101)"
    )
    ap.add_argument(
        "--pixscale", type=float, default=2.75,
        help="Độ phân giải arcsec/pixel (mặc định: 2.75)"
    )
    ap.add_argument(
        "--band", choices=["W1", "W2", "both"], default="W1",
        help="Kênh phổ xuất ra ảnh PNG (file MEF luôn lưu đủ cả W1 và W2)"
    )
    ap.add_argument(
        "--workers", type=int, default=8,
        help="Số luồng tải song song (mặc định: 8)"
    )
    ap.add_argument(
        "--retries", type=int, default=3,
        help="Số lần thử lại tối đa khi gặp lỗi mạng tạm thời (mặc định: 3)"
    )
    ap.add_argument(
        "--no-plot", action="store_true",
        help="Bỏ qua cửa sổ hiển thị matplotlib trực tiếp (vẫn lưu ảnh preview PNG)"
    )
    args = ap.parse_args()

    # Lọc danh sách mục tiêu hợp lệ theo đúng thứ tự yêu cầu
    selected_targets = {}
    for name in args.targets:
        if name in TARGETS:
            selected_targets[name] = TARGETS[name]
        else:
            print(f"[Cảnh báo] Không tìm thấy mục tiêu '{name}' trong danh sách, bỏ qua.")

    if not selected_targets:
        print("Không có thiên thể hợp lệ nào để xử lý.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print("--- FIRST STAGE: QUÉT VÀ TẢI DỮ LIỆU ĐA LUỒNG THEO THIÊN THỂ ---")
    t_start = time.time()
    coords_map = {}
    jobs = []

    for name, info in selected_targets.items():
        coord = SkyCoord(f"{info['ra']} {info['dec']}", frame="icrs")
        coords_map[name] = coord
        for year_str, layer in UNWISE_LAYERS.items():
            fits_url = (
                f"https://www.legacysurvey.org/viewer/fits-cutout?"
                f"ra={coord.ra.deg:.6f}&dec={coord.dec.deg:.6f}&layer={layer}"
                f"&pixscale={args.pixscale}&size={args.size}"
            )
            jobs.append({
                "target": name,
                "year": year_str,
                "layer": layer,
                "url": fits_url,
            })

    print(f"Tổng số yêu cầu cần kiểm tra: {len(jobs)} cutouts "
          f"({len(selected_targets)} thiên thể x {len(UNWISE_LAYERS)} mốc thời gian) "
          f"với {args.workers} luồng...")

    target_records = {name: [] for name in selected_targets}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one_cutout, job, args.retries)
            for job in jobs
        ]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res is not None:
                target_records[res["target"]].append(res)
                print(f"  + {res['target']} [{res['year']} - {res['layer']}]: Tải thành công")

    elapsed_dl = time.time() - t_start
    total_valid = sum(len(v) for v in target_records.values())
    print(f"\n=> Hoàn tất tải {total_valid}/{len(jobs)} cutouts hợp lệ trong {elapsed_dl:.2f} giây.")

    if total_valid == 0:
        print("Không tìm thấy dữ liệu nào. Hủy quá trình.")
        return

    print("\n--- SECOND STAGE: PHÂN LOẠI THƯ MỤC FITS, ĐÓNG GÓI MEF VÀ XUẤT ẢNH ---")
    saved_mef_files = []

    for name, info in selected_targets.items():
        records = target_records[name]
        if not records:
            print(f"\n[{name}] Không có dữ liệu ở bất kỳ mốc nào, bỏ qua.")
            continue

        records.sort(key=lambda r: int(r["year"]))
        dirs = prepare_target_dirs(args.out_dir, name)
        print(f"\n[{name}] ({info['desc']}) -> {len(records)} mốc thời gian: "
              f"{', '.join(r['year'] for r in records)}")

        # 1. Lưu từng tệp FITS gốc vào thư mục <out_dir>/<name>/fits/
        for rec in records:
            yr = rec["year"]
            fits_filename = os.path.join(dirs["fits"], f"{name}_{yr}_WISE.fits")
            with open(fits_filename, "wb") as f:
                f.write(rec["fits_bytes"])

            # 2. Xuất ảnh PNG (annotated & clean) theo kênh phổ đã chọn
            if args.band in ("W1", "both"):
                save_epoch_pngs(rec["w1_data"], "W1", "3.4µm", name, yr, dirs)
            if args.band in ("W2", "both") and rec["w2_data"] is not None:
                save_epoch_pngs(rec["w2_data"], "W2", "4.6µm", name, yr, dirs)

        # 3. Đóng gói tệp Multi-Extension FITS (MEF) tổng hợp cho thiên thể
        mef_filename = os.path.join(dirs["fits"], f"{name}_WISE_mef.fits")
        save_combined_mef(name, info, coords_map[name], records, mef_filename, overwrite=True)

        # 4. Xuất ảnh lưới tổng hợp chuỗi thời gian từ tệp MEF
        preview_png = os.path.join(dirs["root"], f"{name}_WISE_timeline_preview.png")
        saved_mef_files.append((name, mef_filename, preview_png))

    # Vẽ preview và mở cửa sổ xem trước cho thiên thể đầu tiên (M101) nếu bật plot
    for idx, (name, mef_filename, preview_png) in enumerate(saved_mef_files):
        show_now = (not args.no_plot) and (idx == 0)
        plot_mef_preview(mef_filename, preview_png, max_plots=12, show_window=show_now)

    print(f"\nHoàn tất! Toàn bộ dữ liệu đã được sắp xếp theo từng thiên thể tại '{args.out_dir}/'.")


if __name__ == "__main__":
    main()