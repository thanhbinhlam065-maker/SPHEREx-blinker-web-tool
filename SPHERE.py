import argparse
import concurrent.futures
import http.client
import logging
import os
import shutil
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

logging.getLogger('astropy').setLevel(logging.ERROR)  # Tắt cảnh báo alt-WCS
conf.remote_timeout = 120  # Tăng timeout khi đọc dữ liệu SPHEREx từ xa
quantity_support()

TRANSIENT = (TimeoutError, urllib.error.HTTPError, http.client.IncompleteRead,
             http.client.RemoteDisconnected, urllib.error.URLError)


def query_images(coord, size_deg, bandpass=None):
    """Truy vấn danh sách URL cutout và thời gian quan sát từ IRSA TAP."""
    band_clause = (f"AND p.energy_bandpassname = '{bandpass}'"
                   if bandpass else '')
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri
        || '?center={coord.ra.deg},{coord.dec.deg}d&size={size_deg}' AS uri,
        p.time_bounds_lower
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {coord.ra.deg}, {coord.dec.deg}), p.poly)
          {band_clause}
    ORDER BY p.time_bounds_lower
    """
    t0 = time.time()
    res = Irsa.query_tap(query).to_table()
    print(f'TAP query: {len(res)} images in {time.time() - t0:.2f} s')
    return res


def download_one(job, coord, wmin=None, wmax=None, retries=3):
    """Tải 1 cutout, tính bước sóng trung tâm và trả về dữ liệu HDU trong bộ nhớ."""
    uri, t_obs = job['uri'], job['t']
    for attempt in range(retries):
        try:
            with fits.open(uri, cache=False) as hdul:
                # HDU0: Primary (không chứa dữ liệu ảnh);
                # HDU1..6: IMAGE, FLAGS, VARIANCE, ZODI, EPSF, WCS-WAVE
                header = hdul[1].header
                x, y = WCS(header).world_to_pixel(coord)
                sw = WCS(header, fobj=hdul, key='W')
                sw.sip = None
                wl, _ = sw.pixel_to_world(x, y)
                wl_um = float(wl.to(u.um).value)

                if (wmin is not None and wl_um < wmin) or (
                    wmax is not None and wl_um > wmax):
                    return None

                # Sao chép các HDU mở rộng vào RAM để dùng sau khi đóng kết nối
                copied_hdus = [hdu.copy() for hdu in hdul[1:]]
                return dict(
                    uri=uri,
                    mjd=float(t_obs),
                    wavelength_um=wl_um,
                    version=str(hdul[0].header.get('VERSION', '')),
                    hdus=copied_hdus,
                )
        except TRANSIENT:
            # Đã sửa lỗi cú pháp ở dòng này
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f'  skip {uri.split("/")[-1][:60]}...: {e}')
            return None

    print(f'  failed after {retries} tries: {uri.split("/")[-1][:60]}...')
    return None


def save_combined_mef(rows, output_filename, overwrite=True):
    """Đóng gói toàn bộ cutout thành 1 tệp Multi-Extension FITS chuẩn Ban tổ chức."""
    # Sắp xếp theo thời gian quan sát và đánh số cutout_index từ 1..N
    rows.sort(key=lambda r: r['mjd'])
    for idx, r in enumerate(rows, start=1):
        r['cutout_index'] = idx
        for hdu in r['hdus']:
            hdu.header['EXTNAME'] = f"{hdu.header['EXTNAME']}{idx}"

    # Tạo bảng HDU tóm tắt (HDU 1: CUTOUT_INFO) đúng định dạng cột của Ban tổ chức
    cols = fits.ColDefs([
        fits.Column(name='cutout_index', format='J',
                    array=[r['cutout_index'] for r in rows], unit=''),
        fits.Column(name='observation_date', format='D',
                    array=[r['mjd'] for r in rows], unit='d'),
        fits.Column(name='central_wavelength', format='D',
                    array=[r['wavelength_um'] for r in rows], unit='um'),
        fits.Column(name='version', format='A32',
                    array=[r['version'] for r in rows], unit=''),
        fits.Column(name='access_url', format='A200',
                    array=[r['uri'] for r in rows], unit=''),
    ])
    table_hdu = fits.BinTableHDU.from_columns(cols)
    table_hdu.header['EXTNAME'] = 'CUTOUT_INFO'

    # Ghép PrimaryHDU trống + bảng CUTOUT_INFO + toàn bộ HDU ảnh
    primary_hdu = fits.PrimaryHDU()
    hdulist_list = [primary_hdu, table_hdu]
    hdulist_list.extend(hdu for r in rows for hdu in r['hdus'])

    combined_hdulist = fits.HDUList(hdulist_list)
    combined_hdulist.writeto(output_filename, overwrite=overwrite)
    print(f'Saved MEF file: {output_filename} ({len(combined_hdulist)} HDUs total)')


def plot_mef_preview(mef_filename, max_plots=10):
    """Đọc lại tệp MEF và vẽ tối đa 10 ảnh đầu tiên giống phần 11 của Ban tổ chức."""
    summary_table = Table.read(mef_filename, hdu=1)
    nbr_images = min(max_plots, len(summary_table))
    if nbr_images == 0:
        return

    imgs = []
    with fits.open(mef_filename) as hdul:
        for ii in range(nbr_images):
            extname = f"IMAGE{summary_table['cutout_index'][ii]}"
            imgs.append(hdul[extname].data)

    central_wavelength = summary_table['central_wavelength'].quantity

    fig = plt.figure(figsize=(15, 5))
    axs = [fig.add_subplot(2, 5, ii + 1) for ii in range(10)]

    for ii in range(10):
        if ii < nbr_images:
            axs[ii].imshow(imgs[ii], norm='log', origin='lower')
            wl = central_wavelength[ii].to(u.micrometer)
            axs[ii].text(
                0.05, 0.05,
                rf"$\lambda_{{\rm center}} = {wl.value:2.4g} \,{{\rm \mu m}}$",
                va='bottom', ha='left', color='white', transform=axs[ii].transAxes
            )
        else:
            axs[ii].axis('off')

    plt.tight_layout()
    plt.show()


def group_by_wavelength(rows, resolving_power=41.0):
    """Phân nhóm bước sóng (tính năng mở rộng của bạn để làm median differencing)."""
    order = np.argsort([r['wavelength_um'] for r in rows])
    groups, cur = [], []
    for idx in order:
        w = rows[idx]['wavelength_um']
        if cur and w > rows[cur[0]]['wavelength_um'] * (1 + 1.0 / resolving_power):
            groups.append(cur)
            cur = []
        cur.append(idx)
    if cur:
        groups.append(cur)
    return groups


def main():
    ap = argparse.ArgumentParser()
    # Đặt giá trị mặc định là thiên hà Pinwheel (M101) giống tài liệu mẫu để chạy thử ngay
    ap.add_argument('--ra', type=float, default=133.7951, help='deg (ICRS)')
    ap.add_argument('--dec', type=float, default=-7.2451, help='deg (ICRS)')
    ap.add_argument('--size', type=float, default=0.08, help='cutout size, deg')
    ap.add_argument('--bandpass', default='SPHEREx-D2',
                    help="e.g. SPHEREx-D2 (pass '' for all detectors)")
    ap.add_argument('--wmin', type=float, default=None, help='min wavelength, um')
    ap.add_argument('--wmax', type=float, default=None, help='max wavelength, um')
    ap.add_argument('--out', default='WISE_0855_mef.fits',
                    help='output MEF FITS filename')
    ap.add_argument('--workers', type=int, default=10,
                    help='number of parallel threads (recommended 7-12)')
    ap.add_argument('--max-images', type=int, default=None,
                    help='limit number of images (for a quick test)')
    ap.add_argument('--no-plot', action='store_true',
                    help='skip matplotlib preview window')
    ap.add_argument('--resolving-power', type=float, default=41.0)
    args = ap.parse_args()

    coord = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg, frame='icrs')
    bandpass = args.bandpass if args.bandpass else None

    res = query_images(coord, args.size, bandpass)
    if args.max_images:
        res = res[:args.max_images]

    jobs = [dict(uri=str(r['uri']), t=float(r['time_bounds_lower']))
            for r in res]

    rows = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, j, coord, args.wmin, args.wmax)
                for j in jobs]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)

    print(f'Downloaded {len(rows)} / {len(jobs)} cutouts in {(time.time() - t0) / 60:.2f} min')
    if not rows:
        return

    # 1. Xuất tệp MEF đúng chuẩn Ban tổ chức
    save_combined_mef(rows, args.out, overwrite=True)

    # 2. Thống kê nhóm bước sóng (giữ lại phần phân tích của bạn)
    groups = group_by_wavelength(rows, args.resolving_power)
    print(f'\n{len(groups)} wavelength groups (lambda/R = 1/{args.resolving_power:g}):')
    for gi, g in enumerate(groups):
        wl = [rows[i]['wavelength_um'] for i in g]
        mjd = sorted(rows[i]['mjd'] for i in g)
        print(f'  group {gi:02d}: {len(g):3d} images, '
              f'{min(wl):.3f}-{max(wl):.3f} um, MJD {mjd[0]:.1f}-{mjd[-1]:.1f}')

    # 3. Trực quan hóa 10 ảnh đầu tiên từ tệp MEF
    if not args.no_plot:
        plot_mef_preview(args.out, max_plots=10)


if __name__ == '__main__':
    main()