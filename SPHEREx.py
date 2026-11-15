from __future__ import annotations

import concurrent.futures
import http.client
import logging
import time
import urllib.error

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import conf
from astropy.visualization import quantity_support
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa

# Tắt cảnh báo WCS phụ của astropy và tăng thời gian chờ tải dữ liệu từ xa
logging.getLogger("astropy").setLevel(logging.ERROR)
conf.remote_timeout = 120
quantity_support()


# ==============================================================================
# BƯỚC 1: TRUY VẤN DANH SÁCH CUTOUT TỪ IRSA TAP
# ==============================================================================
def query_spherex_cutouts(coord: SkyCoord, size: u.Quantity, bandpass: str) -> Table:
    """Truy vấn danh sách URL các ảnh phổ SPHEREx bao phủ tọa độ mục tiêu."""
    size_deg = size.to(u.degree).value
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri || '?center={coord.ra.deg},{coord.dec.deg}d&size={size_deg}' AS uri,
        p.time_bounds_lower
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {coord.ra.deg}, {coord.dec.deg}), p.poly)
          AND p.energy_bandpassname = '{bandpass}'
    ORDER BY p.time_bounds_lower
    """
    t1 = time.time()
    results, last_err = None, None
    for attempt in range(3):  # IRSA hay quá tải: thử lại với backoff
        try:
            results = Irsa.query_tap(query).to_table()
            break
        except Exception as err:
            last_err = err
            print(f"  -> Truy vấn TAP lỗi (lần {attempt + 1}/3): {err}")
            time.sleep(2 ** (attempt + 1))
    if results is None:
        raise RuntimeError(f"Không truy vấn được IRSA TAP: {last_err}")
    print(f"[1/4] Truy vấn TAP hoàn tất trong {time.time() - t1:.2f} giây. Tìm thấy {len(results)} ảnh.")
    return results


# ==============================================================================
# BƯỚC 2: XỬ LÝ TỪNG CUTOUT & BẮT LỖI MẠNG
# ==============================================================================
def process_single_cutout(row, coord: SkyCoord, cache: bool = False):
    """Tải cutout, tính bước sóng tại tâm và gắn chỉ số cho các HDU."""
    with fits.open(row["uri"], cache=cache) as hdulist:
        # HDU 0: Primary header; HDU 1..6: IMAGE, FLAGS, VARIANCE, ZODI, EPSF, WCS-WAVE
        header = hdulist[1].header

        # Chuyển đổi tọa độ thiên cầu (RA, Dec) sang tọa độ pixel (x, y)
        spatial_wcs = WCS(header)
        x, y = spatial_wcs.world_to_pixel(coord)

        # Tính bước sóng tại vị trí pixel trung tâm
        spectral_wcs = WCS(header, fobj=hdulist, key="W")
        spectral_wcs.sip = None
        wavelength, _ = spectral_wcs.pixel_to_world(x, y)
        row["central_wavelength"] = wavelength.to(u.micrometer).value

        # Lưu lại phiên bản xử lý (VERSION) từ Primary Header
        row["version"] = hdulist[0].header.get("VERSION", "")

        # Đổi tên EXTNAME kèm cutout_index để tránh trùng lặp khi gộp chung một MEF
        hdus = []
        for hdu in hdulist[1:]:
            hdu.header["EXTNAME"] = f"{hdu.header['EXTNAME']}{row['cutout_index']}"
            hdus.append(hdu.copy())
        row["hdus"] = hdus


def safe_process_cutout(row, coord: SkyCoord, cache: bool = False, retries: int = 3):
    """Bọc hàm xử lý cutout: thử lại tối đa `retries` lần, hết lần thì bỏ qua cutout."""
    last = None
    for attempt in range(retries):
        try:
            process_single_cutout(row, coord, cache=cache)
            return
        except Exception as err:  # lỗi mạng chập chờn (URLError, OSError, Timeout...)
            last = err
            time.sleep(min(2 ** attempt, 8))
    print(f"  -> Bỏ qua cutout #{row['cutout_index']} do lỗi tải: {last}")
    row["hdus"] = None


# ==============================================================================
# BƯỚC 3: TẢI DỮ LIỆU (HỖ TRỢ SONG SONG HOẶC TUẦN TỰ)
# ==============================================================================
def download_all_cutouts(
    results_table: Table,
    coord: SkyCoord,
    parallel: bool = True,
    max_workers: int = 10,
    cache: bool = False,
    time_limit_s: float | None = None,
    progress_cb=None,
) -> Table:
    """Tải toàn bộ cutout (song song hoặc tuần tự). Dừng chờ khi quá `time_limit_s` giây
    và giữ lại những cutout đã tải xong. progress_cb(done, total) được gọi sau mỗi cutout."""
    n = len(results_table)
    results_table["cutout_index"] = range(1, n + 1)
    results_table["central_wavelength"] = np.full(n, np.nan)
    results_table["version"] = np.full(n, "", dtype="U32")
    results_table["hdus"] = np.full(n, None)

    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress_cb is not None:
            try:
                progress_cb(done, n)
            except Exception:
                pass

    t1 = time.time()
    if parallel:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        futures = [executor.submit(safe_process_cutout, row, coord, cache) for row in results_table]
        try:
            for _ in concurrent.futures.as_completed(futures, timeout=time_limit_s):
                tick()
        except concurrent.futures.TimeoutError:
            print(f"  -> Hết thời gian tải ({time_limit_s:.0f} giây): giữ lại phần đã tải xong.")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        mode_str = f"song song ({max_workers} luồng)"
    else:
        for row in results_table:
            if time_limit_s is not None and time.time() - t1 > time_limit_s:
                print("  -> Hết thời gian tải: giữ lại phần đã tải xong.")
                break
            safe_process_cutout(row, coord, cache)
            tick()
        mode_str = "tuần tự"

    elapsed_min = (time.time() - t1) / 60.0
    valid_mask = [h is not None for h in results_table["hdus"]]
    failed_count = n - sum(valid_mask)
    print(f"[2/4] Tải {mode_str} hoàn tất trong {elapsed_min:.2f} phút (Thành công: {sum(valid_mask)}, Lỗi/bỏ qua: {failed_count}).")

    return results_table[valid_mask]


# ==============================================================================
# BƯỚC 4: ĐÓNG GÓI THÀNH TỆP MULTI-EXTENSION FITS (MEF)
# ==============================================================================
def build_and_save_mef(valid_table: Table, output_filename: str, overwrite: bool = True):
    """Tạo bảng tóm tắt CUTOUT_INFO và ghi toàn bộ HDU ra tệp FITS."""
    cols = fits.ColDefs([
        fits.Column(name="cutout_index", format="J", array=valid_table["cutout_index"], unit=""),
        fits.Column(name="observation_date", format="D", array=valid_table["time_bounds_lower"], unit="d"),
        fits.Column(name="central_wavelength", format="D", array=valid_table["central_wavelength"], unit="um"),
        fits.Column(name="version", format="A32", array=valid_table["version"], unit=""),
        fits.Column(name="access_url", format="A200", array=valid_table["uri"], unit=""),
    ])
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header["EXTNAME"] = "CUTOUT_INFO"

    primary_hdu = fits.PrimaryHDU()
    hdulist_list = [primary_hdu, table_hdu]
    hdulist_list.extend(hdu for fits_hdulist in valid_table["hdus"] for hdu in fits_hdulist)

    combined_hdulist = fits.HDUList(hdulist_list)
    combined_hdulist.writeto(output_filename, overwrite=overwrite)
    print(f"[3/4] Đã lưu tệp MEF thành công tại: {output_filename} (Tổng cộng {len(combined_hdulist)} HDUs).")


# ==============================================================================
# BƯỚC 5: KIỂM TRA VÀ TRỰC QUAN HÓA KẾT QUẢ
# ==============================================================================
def inspect_and_plot_mef(output_filename: str, max_plots: int = 10, show: bool = True):
    """Đọc lại tệp MEF và hiển thị tối đa 10 ảnh đầu tiên kèm bước sóng trung tâm."""
    summary_table = Table.read(output_filename, hdu=1)
    nbr_images = min(max_plots, len(summary_table))
    print(f"[4/4] Đang hiển thị {nbr_images}/{len(summary_table)} ảnh từ bảng CUTOUT_INFO...")

    imgs = []
    with fits.open(output_filename) as hdul:
        for ii in range(nbr_images):
            extname = f"IMAGE{summary_table['cutout_index'][ii]}"
            imgs.append(hdul[extname].data)

    central_wavelength = summary_table["central_wavelength"].quantity

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes_flat = axes.flatten()

    for ii in range(len(axes_flat)):
        if ii < nbr_images:
            axes_flat[ii].imshow(imgs[ii], norm="log", origin="lower")
            wl = central_wavelength[ii].to(u.micrometer)
            axes_flat[ii].text(
                0.05, 0.05,
                rf"$\lambda_{{\rm center}} = {wl.value:2.4g} \,{{\rm \mu m}}$",
                va="bottom", ha="left", color="white", transform=axes_flat[ii].transAxes
            )
            axes_flat[ii].set_title(f"Cutout #{summary_table['cutout_index'][ii]}")
        else:
            axes_flat[ii].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==============================================================================
# HÀM CHẠY TOÀN BỘ PIPELINE
# ==============================================================================
def run_spherex_cutout_pipeline(
    coord: SkyCoord,
    size: u.Quantity = 0.1 * u.degree,
    bandpass: str = "SPHEREx-D2",
    output_filename: str = "spherex_cutouts_mef.fits",
    parallel: bool = True,
    max_workers: int = 10,
    cache: bool = False,
    max_cutouts: int | None = None,
    overwrite: bool = True,
    show_plot: bool = True,
    plot: bool | None = None,
    time_limit_s: float | None = None,
    min_cutouts: int = 1,
    progress_cb=None,
):
    """Chạy toàn bộ pipeline; trả về đường dẫn tệp MEF (hoặc None nếu thất bại).

    plot: bí danh của show_plot (app Streamlit truyền plot=False).
    max_cutouts: chọn tối đa N ảnh RẢI ĐỀU theo thời gian (không lấy N ảnh đầu tiên).
    time_limit_s / progress_cb: giới hạn thời gian tải và callback tiến độ cho giao diện.
    """
    if plot is not None:
        show_plot = plot

    results_table = query_spherex_cutouts(coord, size, bandpass)
    if len(results_table) == 0:
        print("Không tìm thấy dữ liệu phù hợp với tọa độ và dải tần đã chọn.")
        return None

    if max_cutouts is not None and len(results_table) > max_cutouts:
        idx = np.unique(np.linspace(0, len(results_table) - 1, max_cutouts).round().astype(int))
        results_table = results_table[idx]
        print(f"-> Chọn {len(results_table)} ảnh rải đều theo thời gian.")

    valid_table = download_all_cutouts(
        results_table=results_table,
        coord=coord,
        parallel=parallel,
        max_workers=max_workers,
        cache=cache,
        time_limit_s=time_limit_s,
        progress_cb=progress_cb,
    )

    if len(valid_table) == 0 or len(valid_table) < min_cutouts:
        print(f"Chỉ tải được {len(valid_table)} cutout (cần ≥ {max(min_cutouts, 1)}), không tạo tệp MEF.")
        return None

    build_and_save_mef(valid_table, output_filename, overwrite=overwrite)
    if show_plot:
        inspect_and_plot_mef(output_filename, show=True)
    return output_filename


if __name__ == "__main__":
    # Ví dụ chạy thử với thiên hà Pinwheel (M101) trên detector D2
    target_coord = SkyCoord(ra=210.80227 * u.degree, dec=54.34895 * u.degree, frame="icrs")

    run_spherex_cutout_pipeline(
        coord=target_coord,
        size=0.1 * u.degree,
        bandpass="SPHEREx-D2",
        output_filename="spherex_cutouts_mef.fits",
        parallel=True,       # Đặt False nếu muốn chạy tuần tự (serial)
        max_workers=10,      # Khuyến nghị 7-12 luồng cho máy 8 nhân
        cache=False,         # Đặt True nếu muốn lưu cache vào ~/.astropy/cache/ để chạy lại nhanh hơn
        max_cutouts=10,      # Đặt None để tải toàn bộ; đặt số nhỏ (VD: 10) để test nhanh
        overwrite=True,      # Ghi đè tệp cũ khi chạy thử nghiệm nhiều lần
    )