import argparse
import glob
import os
import warnings

from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
import cv2
import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import CircularAperture
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')
FWHM_TO_SIGMA = 1.0 / 2.3548


# --------------------------------------------------------------------------
# 1. Đọc dữ liệu FITS & trích xuất tọa độ tâm
# --------------------------------------------------------------------------
def _xy(sources):
  """Trả về mảng (N, 2) tọa độ (x, y) tương thích mọi phiên bản photutils."""
  cols = sources.colnames
  for cx, cy in (
      ('x_centroid', 'y_centroid'),
      ('xcentroid', 'ycentroid'),
      ('xcenter', 'ycenter'),
  ):
    if cx in cols:
      return np.transpose((np.asarray(sources[cx]), np.asarray(sources[cy])))
  raise KeyError(f'Không tìm thấy cột tọa độ tâm trong {cols}')


def load_fits_with_wcs(filepath):
  """Đọc ảnh FITS 2D (float32, giữ nguyên NaN) kèm thông tin WCS thiên văn."""
  with fits.open(
      filepath, ignore_missing_simple=True, ignore_missing_end=True
  ) as hdul:
    for hdu in hdul:
      if hdu.data is None:
        continue
      data = np.squeeze(hdu.data).astype(np.float32)
      if data.ndim > 2:
        data = data[0]
      if data.ndim != 2:
        continue
      try:
        wcs = WCS(hdu.header).celestial
        if wcs.naxis != 2 or not wcs.has_celestial:
          wcs = None
      except Exception:
        wcs = None
      return data, wcs
  raise ValueError(f'Không tìm thấy dữ liệu ảnh 2D hợp lệ trong {filepath}')


# --------------------------------------------------------------------------
# 2. Đồng trục tọa độ (Reprojection)
# --------------------------------------------------------------------------
def align_to_reference(frames, wcs_list, ref_index=0, exact=False):
  """Chiếu lại toàn bộ khung hình về lưới pixel của frames[ref_index].

  exact=False -> reproject_interp (nhanh, dùng khi tỷ lệ pixel tương đương)
  exact=True  -> reproject_exact  (bảo toàn thông lượng, dùng khi lệch scale lớn)
  """
  if any(w is None for w in wcs_list):
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
      raise ValueError(
          'Có khung hình thiếu WCS và khác kích thước pixel; không thể đồng trục.'
      )
    print('  ⚠️ Một số khung hình thiếu WCS; giữ nguyên lưới pixel gốc.')
    aligned = [f.astype(np.float32) for f in frames]
    invalid = np.zeros(aligned[0].shape, dtype=bool)
    for a in aligned:
      invalid |= ~np.isfinite(a)
    return aligned, invalid

  try:
    from reproject import reproject_exact, reproject_interp
  except ImportError as exc:
    raise ImportError(
        'Cần cài đặt thư viện "reproject" (pip install reproject) để đồng trục WCS.'
    ) from exc

  func = reproject_exact if exact else reproject_interp
  ref_wcs = wcs_list[ref_index]
  shape = frames[ref_index].shape
  aligned = []

  for i, (data, wcs) in enumerate(zip(frames, wcs_list)):
    if i == ref_index:
      aligned.append(data.astype(np.float32))
    else:
      out, _ = func((data, wcs), ref_wcs, shape_out=shape)
      aligned.append(out.astype(np.float32))

  invalid = np.zeros(shape, dtype=bool)
  for a in aligned:
    invalid |= ~np.isfinite(a)
  return aligned, invalid


# --------------------------------------------------------------------------
# 3. Đồng bộ hàm lan truyền điểm (PSF Matching)
# --------------------------------------------------------------------------
def estimate_fwhm(img, n_stars=15, box=7):
  """Ước lượng FWHM trung vị (pixel) bằng khớp hàm Gaussian 2D lên sao sáng."""
  med_val = float(np.nanmedian(img))
  img_clean = np.nan_to_num(img, nan=med_val, posinf=med_val, neginf=med_val)
  _, med, std = sigma_clipped_stats(img_clean, sigma=3.0)
  if not np.isfinite(std) or std <= 0:
    return None

  finder = DAOStarFinder(fwhm=3.0, threshold=10.0 * std)
  src = finder(img_clean - med)
  if src is None or len(src) < 5:
    return None

  src.sort('peak')
  # Loại bỏ 20% sao sáng nhất (tránh bão hòa), lấy tối đa n_stars sao tiếp theo
  start = max(0, int(0.8 * len(src)) - n_stars)
  src = src[start : int(0.8 * len(src))]
  xy = _xy(src)

  fitter = fitting.LevMarLSQFitter()
  ny, nx = img_clean.shape
  yy, xx = np.mgrid[-box : box + 1, -box : box + 1]
  fwhms = []

  for x, y in xy:
    xi, yi = int(round(x)), int(round(y))
    if xi < box or yi < box or xi >= nx - box or yi >= ny - box:
      continue
    cut = img_clean[yi - box : yi + box + 1, xi - box : xi + box + 1] - med
    g0 = models.Gaussian2D(cut.max(), 0, 0, 1.5, 1.5)
    try:
      g = fitter(g0, xx, yy, cut)
      s = 0.5 * (abs(g.x_stddev.value) + abs(g.y_stddev.value))
      if 0.5 < s < 6.0:
        fwhms.append(s / FWHM_TO_SIGMA)
    except Exception:
      continue

  return float(np.median(fwhms)) if len(fwhms) >= 3 else None


def match_psf(frames, fwhms):
  """Làm mờ các khung hình sắc nét để đồng bộ về PSF rộng nhất trong chuỗi."""
  valid = [f for f in fwhms if f is not None]
  if not valid:
    return frames, None

  target = max(valid)
  out = []
  for img, f in zip(frames, fwhms):
    if f is None or f >= target - 0.05:
      out.append(img)
      continue
    sig = np.sqrt(target**2 - f**2) * FWHM_TO_SIGMA
    nan_mask = ~np.isfinite(img)
    filled = np.where(nan_mask, np.nanmedian(img), img)
    blurred = gaussian_filter(filled, sig)
    blurred[nan_mask] = np.nan
    out.append(blurred.astype(np.float32))
  return out, target


# --------------------------------------------------------------------------
# 4. Chuẩn hóa thông lượng & trừ phông nền (Flux / Background Matching)
# --------------------------------------------------------------------------
def scale_to_reference(img, ref, mask=None, n_iter=6, clip=3.0, step=3):
  """Khớp tuyến tính img ~ a*ref + b với sigma-clipping; trả về (img - b) / a."""
  ok = np.isfinite(img) & np.isfinite(ref)
  if mask is not None:
    ok &= ~mask

  x = ref[::step, ::step][ok[::step, ::step]]
  y = img[::step, ::step][ok[::step, ::step]]
  keep = np.ones(x.size, dtype=bool)
  a, b = 1.0, 0.0

  for _ in range(n_iter):
    if keep.sum() < 100:
      break
    a, b = np.polyfit(x[keep], y[keep], 1)
    resid = y - (a * x + b)
    s = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
    keep = np.abs(resid) < clip * max(s, 1e-12)

  if not np.isfinite(a) or a <= 0:
    print('  ⚠️ Khớp tỷ lệ thông lượng thất bại, giữ nguyên thang đo gốc.')
    return img
  return ((img - b) / a).astype(np.float32)


def prepare_stack_arrays(
    data,
    wcss,
    names,
    ref_index=0,
    do_psf=True,
    do_scale=True,
    exact=False,
    min_valid_frac=0.05,
    return_keep=False,
):
  """Tiền xử lý từ mảng có sẵn (không cần đọc file): Reproject -> PSF -> Flux -> Trừ nền.

  data : list ảnh 2D float32 (NaN = pixel xấu)
  wcss : list WCS 2D (hoặc None)
  Khung nào sau reproject có < min_valid_frac pixel hợp lệ (không phủ vùng tham
  chiếu) sẽ bị loại. Trả về (stack, names, invalid_mask, fwhm_target); nếu
  return_keep=True thì thêm danh sách chỉ số gốc của các khung được giữ lại.
  """
  if len(data) < 3:
    raise ValueError(f'Cần ít nhất 3 khung hình hợp lệ (hiện có {len(data)}).')
  if not 0 <= ref_index < len(data):
    ref_index = 0

  frames, _ = align_to_reference(data, wcss, ref_index, exact)

  keep = [
      i
      for i, f in enumerate(frames)
      if i == ref_index or float(np.isfinite(f).mean()) >= min_valid_frac
  ]
  if len(keep) < 3:
    raise ValueError(
        f'Chỉ {len(keep)} khung phủ vùng tham chiếu sau khi đồng trục (cần >= 3).'
    )
  if len(keep) < len(frames):
    print(f'  ⚠️ Loại {len(frames) - len(keep)} khung không phủ vùng tham chiếu.')
  frames = [frames[i] for i in keep]
  names = [names[i] for i in keep]
  ref_index = keep.index(ref_index)

  invalid = np.zeros(frames[0].shape, dtype=bool)
  for f in frames:
    invalid |= ~np.isfinite(f)

  fwhm_target = None
  if do_psf:
    fwhms = [estimate_fwhm(f) for f in frames]
    print(
        '  • FWHM đo được (px):',
        [None if v is None else round(v, 2) for v in fwhms],
    )
    frames, fwhm_target = match_psf(frames, fwhms)
    if fwhm_target is not None:
      print(f'  • Đồng bộ PSF về mục tiêu: {fwhm_target:.2f} px')

  if do_scale:
    ref = frames[ref_index]
    frames = [
        f if i == ref_index else scale_to_reference(f, ref, invalid)
        for i, f in enumerate(frames)
    ]

  out = []
  for f in frames:
    med = float(np.nanmedian(f))
    g = np.where(np.isfinite(f), f, med) - med
    out.append(g.astype(np.float32))

  result = (np.array(out), list(names), invalid, fwhm_target)
  return result + (keep,) if return_keep else result


def prepare_stack(
    files, ref_index=0, do_psf=True, do_scale=True, exact=False
):
  """Chuỗi tiền xử lý từ danh sách file: Load -> prepare_stack_arrays."""
  data, wcss, names = [], [], []
  for f in files:
    try:
      d, w = load_fits_with_wcs(f)
      data.append(d)
      wcss.append(w)
      names.append(os.path.basename(f))
    except Exception as e:
      print(f"  ❌ Bỏ qua file lỗi '{os.path.basename(f)}': {e}")

  return prepare_stack_arrays(data, wcss, names, ref_index, do_psf, do_scale, exact)


# --------------------------------------------------------------------------
# 5. Tạo mặt nạ loại trừ & phát hiện ứng viên (Detection)
# --------------------------------------------------------------------------
def build_exclusion_mask(stack, static_sky, extra=None):
  """Tạo mặt nạ loại trừ lõi sao bão hòa, tia nhiễu xạ, viền ảnh và vùng ngoài WCS."""
  _, med_sky, std_sky = sigma_clipped_stats(static_sky, sigma=3.0)
  core = (static_sky > med_sky + 15.0 * std_sky).astype(np.uint8) * 255
  core = cv2.dilate(
      core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
  )

  tstd = np.std(stack, axis=0)
  _, med_t, std_t = sigma_clipped_stats(tstd, sigma=3.0)
  spike = (tstd > med_t + 8.0 * std_t).astype(np.uint8) * 255
  spike = cv2.dilate(
      spike, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
  )

  mask = cv2.bitwise_or(core, spike) > 0
  b = 8
  mask[:b, :] = mask[-b:, :] = True
  mask[:, :b] = mask[:, -b:] = True
  if extra is not None:
    mask |= extra

  if mask.mean() > 0.70:
    print('  ⚠️ Exclusion mask che >70% diện tích ảnh; chỉ giữ lại mặt nạ viền.')
    mask = np.zeros_like(mask, dtype=bool)
    mask[:b, :] = mask[-b:, :] = True
    mask[:, :b] = mask[:, -b:] = True
  return mask


def prepare_reference(stack, extra_mask=None):
  """Tính ảnh nền tĩnh (median stack) và mặt nạ loại trừ nền tảng."""
  static_sky = np.median(stack, axis=0)
  return static_sky, build_exclusion_mask(stack, static_sky, extra_mask)


def detect_candidates(
    stack, i, static_sky, base_mask, threshold_sigma=5.0, fwhm=3.0
):
  """Trừ nền tĩnh, cập nhật mặt nạ bão hòa động và tìm nguồn điểm bằng DAOStarFinder."""
  img = stack[i]
  diff = img - static_sky

  _, med_y, std_y = sigma_clipped_stats(img, sigma=3.0)
  sat = (img > med_y + 15.0 * std_y).astype(np.uint8) * 255
  sat = (
      cv2.dilate(sat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
      > 0
  )
  excl = base_mask | sat

  valid = diff[~excl]
  if valid.size > 100:
    _, med_d, std_d = sigma_clipped_stats(valid, sigma=3.0)
  else:
    _, med_d, std_d = sigma_clipped_stats(diff, sigma=3.0)

  if not np.isfinite(std_d) or std_d <= 0:
    med_d = float(np.nanmedian(diff))
    std_d = float(np.nanstd(diff)) or 1.0

  finder = DAOStarFinder(
      fwhm=fwhm,
      threshold=threshold_sigma * std_d,
      sharplo=0.2,
      sharphi=1.0,
      roundlo=-1.0,
      roundhi=1.0,
  )
  src = finder(diff - med_d, mask=excl)
  return src, diff, excl, med_d, std_d


# --------------------------------------------------------------------------
# 6. Kiểm thử độ nhạy Injection-Recovery
# --------------------------------------------------------------------------
def add_gaussian(img, x, y, fwhm, peak):
  """Bơm nguồn điểm Gaussian tròn (biên độ peak) trực tiếp vào ảnh."""
  s = fwhm * FWHM_TO_SIGMA
  half = int(np.ceil(5 * s))
  x0 = int(max(0, np.floor(x) - half))
  x1 = int(min(img.shape[1], np.floor(x) + half + 2))
  y0 = int(max(0, np.floor(y) - half))
  y1 = int(min(img.shape[0], np.floor(y) + half + 2))
  if x1 <= x0 or y1 <= y0:
    return
  yy, xx = np.mgrid[y0:y1, x0:x1]
  img[y0:y1, x0:x1] += peak * np.exp(
      -((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)
  )


def _random_positions(rng, n, shape, margin_x, margin_y, min_sep=40):
  ny, nx = shape
  if nx - margin_x <= margin_x or ny - margin_y <= margin_y:
    return []
  pts = []
  tries = 0
  while len(pts) < n and tries < 5000:
    tries += 1
    p = (
        rng.uniform(margin_x, nx - margin_x),
        rng.uniform(margin_y, ny - margin_y),
    )
    if all(
        (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > min_sep**2 for q in pts
    ):
      pts.append(p)
  return pts


def injection_recovery(
    stack,
    snr_grid=(3, 4, 5, 6, 8, 10, 12),
    motion_grid=(0.0, 1.0, 3.0, 6.0, 12.0),
    n_trials=8,
    n_inject=6,
    fwhm=3.0,
    threshold_sigma=5.0,
    match_radius=3.0,
    extra_mask=None,
    seed=0,
):
  """Đánh giá tỷ lệ phát hiện (completeness) theo SNR và tốc độ dịch chuyển."""
  rng = np.random.default_rng(seed)
  nf, ny, nx = stack.shape
  sigmas = [sigma_clipped_stats(f, sigma=3.0)[2] for f in stack]
  rows = []

  print('\n🔬 Đang chạy mô phỏng Injection-Recovery...')
  for motion in motion_grid:
    for snr in snr_grid:
      rec = tot = fp = n_img = 0
      for _ in range(n_trials):
        margin_x = 20 + abs(motion) * (nf - 1)
        pos = _random_positions(rng, n_inject, (ny, nx), margin_x, 20)
        if not pos:
          continue
        inj = stack.copy()
        truth = [[] for _ in range(nf)]
        for x0, y0 in pos:
          for i in range(nf):
            x, y = x0 + i * motion, y0
            add_gaussian(inj[i], x, y, fwhm, snr * sigmas[i])
            truth[i].append((x, y))

        static_sky, base = prepare_reference(inj, extra_mask)
        for i in range(nf):
          src, _, _, _, _ = detect_candidates(
              inj, i, static_sky, base, threshold_sigma, fwhm
          )
          det = (
              _xy(src) if src is not None and len(src) else np.empty((0, 2))
          )
          t = np.array(truth[i])
          for tx, ty in t:
            tot += 1
            if (
                len(det)
                and np.min(np.hypot(det[:, 0] - tx, det[:, 1] - ty))
                <= match_radius
            ):
              rec += 1
          for dx, dy in det:
            if np.min(np.hypot(t[:, 0] - dx, t[:, 1] - dy)) > match_radius:
              fp += 1
          n_img += 1

      rows.append(
          dict(
              motion=motion,
              snr=snr,
              completeness=rec / max(tot, 1),
              fp_per_frame=fp / max(n_img, 1),
              n_injected=tot,
          )
      )
      print(
          f'  motion={motion:5.1f}px | SNR={snr:4.1f} | '
          f'recovered={rec / max(tot, 1):5.1%} | '
          f'FP/frame={fp / max(n_img, 1):.2f}'
      )
  return rows


def plot_injection_results(
    rows, out_png='injection_recovery.png', show=True
):
  """Vẽ biểu đồ đường cong hoàn thiện và tỷ lệ dương tính giả."""
  motions = sorted({r['motion'] for r in rows})
  fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
  for m in motions:
    rr = [r for r in rows if r['motion'] == m]
    ax[0].plot(
        [r['snr'] for r in rr],
        [r['completeness'] for r in rr],
        marker='o',
        label=f'{m:g} px/epoch',
    )
    ax[1].plot(
        [r['snr'] for r in rr],
        [r['fp_per_frame'] for r in rr],
        marker='s',
        label=f'{m:g} px/epoch',
    )

  ax[0].set_xlabel('SNR đỉnh bơm vào (σ / pixel)')
  ax[0].set_ylabel('Tỷ lệ phát hiện (Recovery fraction)')
  ax[0].set_ylim(-0.02, 1.02)
  ax[0].set_title('Độ hoàn thiện theo SNR & Tốc độ dịch chuyển')
  ax[0].legend()
  ax[0].grid(alpha=0.3)

  ax[1].set_xlabel('SNR đỉnh bơm vào (σ / pixel)')
  ax[1].set_ylabel('Ứng viên dương tính giả / khung hình')
  ax[1].set_title('Tỷ lệ dương tính giả (False Positives)')
  ax[1].legend()
  ax[1].grid(alpha=0.3)

  plt.tight_layout()
  if out_png:
    plt.savefig(out_png, dpi=140)
    print(f'💾 Đã lưu biểu đồ Injection-Recovery tại: {out_png}')
  if show:
    plt.show(block=True)
  else:
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. Quy trình chính & Giao diện GUI / CLI
# --------------------------------------------------------------------------
def find_planet_x_candidates(
    folder_path,
    threshold_sigma=5.0,
    ref_index=0,
    do_psf=True,
    do_scale=True,
    exact_reproject=False,
    run_injection=False,
    n_trials=8,
    show=True,
):
  """Phân tích chuỗi ảnh FITS của một thiên thể và trực quan hóa kết quả."""
  target_name = os.path.basename(os.path.normpath(folder_path))
  all_files = sorted(glob.glob(os.path.join(folder_path, '*.fits')))
  all_files = [
      f
      for f in all_files
      if '_starless' not in f and '_stars_mask' not in f
  ]

  if len(all_files) < 3:
    print(
        f'\n⚠️ [{target_name}]: Bỏ qua do cần ít nhất 3 mốc thời gian (hiện có'
        f' {len(all_files)} file).'
    )
    return

  print('\n==================================================')
  print(
      f'🚀 MỤC TIÊU: {target_name} | NGƯỠNG: {threshold_sigma}σ ({len(all_files)}'
      ' file)'
  )
  print('==================================================')

  try:
    stack, names, invalid_mask, fwhm_target = prepare_stack(
        all_files,
        ref_index=ref_index,
        do_psf=do_psf,
        do_scale=do_scale,
        exact=exact_reproject,
    )
  except Exception as e:
    print(f'❌ [{target_name}]: Lỗi tiền xử lý chuỗi ảnh: {e}')
    return

  fwhm_used = fwhm_target or 3.0
  num_frames = len(stack)
  static_sky, base_excl_mask = prepare_reference(
      stack, extra_mask=invalid_mask
  )

  fig, axes = plt.subplots(
      num_frames, 3, figsize=(16, 4.2 * num_frames), dpi=110
  )
  interval = ZScaleInterval()

  for i in range(num_frames):
    img_year = stack[i]
    sources, diff_img, excl_mask, med_d, std_d = detect_candidates(
        stack,
        i,
        static_sky,
        base_excl_mask,
        threshold_sigma=threshold_sigma,
        fwhm=fwhm_used,
    )

    vmin, vmax = interval.get_limits(img_year)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

    axes[i, 0].imshow(img_year, cmap='inferno', origin='lower', norm=norm)
    axes[i, 0].set_title(f'Đã đồng bộ PSF & Flux: {names[i]}', fontsize=9)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(static_sky, cmap='inferno', origin='lower', norm=norm)
    axes[i, 1].contour(
        excl_mask, levels=[0.5], colors='cyan', linewidths=1.0, linestyles='--'
    )
    axes[i, 1].set_title('Nền tĩnh + Viền Exclusion Mask', fontsize=9)
    axes[i, 1].axis('off')

    axes[i, 2].imshow(
        diff_img,
        cmap='coolwarm',
        origin='lower',
        vmin=med_d - 5 * std_d,
        vmax=med_d + 5 * std_d,
    )
    axes[i, 2].contour(
        excl_mask, levels=[0.5], colors='black', linewidths=0.8, alpha=0.5
    )
    axes[i, 2].set_title(
        f'Ảnh Vi Sai (Ngưỡng: {threshold_sigma}σ | FWHM: {fwhm_used:.1f}px)',
        fontsize=9,
    )
    axes[i, 2].axis('off')

    if sources is not None and len(sources) > 0:
      positions = _xy(sources)
      apertures = CircularAperture(positions, r=8.0)
      apertures.plot(color='lime', lw=2.0, ax=axes[i, 2])
      print(
          f'  • {names[i]}: 🎯 Tìm thấy {len(sources)} ứng viên vượt'
          f' {threshold_sigma}σ!'
      )
    else:
      print(
          f'  • {names[i]}: Sạch (Không có vật thể lạ vượt'
          f' {threshold_sigma}σ).'
      )

  plt.suptitle(
      f'TÌM KIẾM HÀNH TINH X ({threshold_sigma}σ) - {target_name}',
      fontsize=13,
      fontweight='bold',
  )
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  if show:
    plt.show(block=True)

  if run_injection:
    rows = injection_recovery(
        stack,
        fwhm=fwhm_used,
        threshold_sigma=threshold_sigma,
        n_trials=n_trials,
        extra_mask=invalid_mask,
    )
    out_path = os.path.join(
        folder_path, f'{target_name}_injection_recovery.png'
    )
    plot_injection_results(rows, out_png=out_path, show=show)

  # show=False (chạy trên server): trả figure để Streamlit tự vẽ rồi tự đóng.
  return None if show else fig


def make_synthetic_stack(n=4, size=300, seed=1):
  """Tạo chuỗi ảnh giả lập dùng để kiểm thử nhanh thuật toán."""
  rng = np.random.default_rng(seed)
  sky = np.zeros((size, size), dtype=np.float32)
  for _ in range(30):
    add_gaussian(
        sky,
        rng.uniform(20, size - 20),
        rng.uniform(20, size - 20),
        3.0,
        rng.uniform(5, 200),
    )
  return np.array(
      [
          sky + rng.normal(0, 1.0, sky.shape).astype(np.float32)
          for _ in range(n)
      ]
  )


def run_gui_tool():
  """Giao diện chọn thư mục, cấu hình ngưỡng sigma và tùy chọn Injection-Recovery."""
  # Import tại chỗ: Streamlit Cloud/server không có tkinter hay màn hình.
  import tkinter as tk
  from tkinter import filedialog, messagebox, simpledialog

  root = tk.Tk()
  root.withdraw()
  root.attributes('-topmost', True)

  print('Đang mở cửa sổ chọn thư mục dữ liệu FITS...')
  selected_dir = filedialog.askdirectory(
      title='Chọn thư mục thiên thể hoặc thư mục gốc FITS_DATA'
  )

  if not selected_dir:
    print('Thao tác bị hủy.')
    return

  prompt_msg = (
      'Nhập giá trị Threshold Sigma (Độ nhạy phát hiện):\n\n'
      '• 4.5 - 5.0 : Mục tiêu mờ, vùng trời thoáng (vd: WISE 0855)\n'
      '• 5.5 - 6.5 : Mục tiêu chuẩn / Sao chuyển động nhanh (vd: Barnard,'
      ' Epsilon Indi B)\n'
      '• 7.0 - 9.0 : Sao biến quang cực sáng, nhiều hào quang (vd: R'
      ' Draconis)\n'
  )

  sigma_val = simpledialog.askfloat(
      title='Cấu hình Threshold Sigma',
      prompt=prompt_msg,
      initialvalue=5.0,
      minvalue=0.0,
      maxvalue=20.0,
      parent=root,
  )

  if sigma_val is None:
    print('Sử dụng ngưỡng mặc định: 5.0σ')
    sigma_val = 5.0

  run_inj = messagebox.askyesno(
      title='Kiểm thử độ nhạy (Injection-Recovery)',
      message=(
          'Bạn có muốn chạy thêm mô phỏng Injection-Recovery\n'
          'để đo tỷ lệ phát hiện theo SNR & tốc độ di chuyển không?'
      ),
      parent=root,
  )

  fits_in_dir = [
      f
      for f in glob.glob(os.path.join(selected_dir, '*.fits'))
      if '_starless' not in f and '_stars_mask' not in f
  ]

  if len(fits_in_dir) > 0:
    find_planet_x_candidates(
        selected_dir, threshold_sigma=sigma_val, run_injection=run_inj
    )
  else:
    subfolders = [
        f.path
        for f in os.scandir(selected_dir)
        if f.is_dir() and f.name != 'Starless'
    ]
    if not subfolders:
      print('⚠️ Không tìm thấy dữ liệu FITS hợp lệ.')
      return

    for sub_dir in sorted(subfolders):
      target_sub = os.path.basename(sub_dir)
      sub_sigma = simpledialog.askfloat(
          title=f'Threshold Sigma cho {target_sub}',
          prompt=f'Đang chuẩn bị phân tích: {target_sub}\n\n{prompt_msg}',
          initialvalue=sigma_val,
          minvalue=0.0,
          maxvalue=20.0,
          parent=root,
      )
      if sub_sigma is None:
        print(f'⏭️ Bỏ qua thiên thể: {target_sub}')
        continue

      find_planet_x_candidates(
          sub_dir, threshold_sigma=sub_sigma, run_injection=run_inj
      )


def main():
  ap = argparse.ArgumentParser(
      description='Tìm kiếm vật thể chuyển động & kiểm thử Injection-Recovery trên ảnh FITS.'
  )
  ap.add_argument('--folder', help='Đường dẫn thư mục chứa các file .fits')
  ap.add_argument('--ref', type=int, default=0, help='Chỉ số khung hình tham chiếu')
  ap.add_argument('--sigma', type=float, default=5.0, help='Ngưỡng phát hiện sigma')
  ap.add_argument('--exact', action='store_true', help='Dùng reproject_exact thay vì interp')
  ap.add_argument('--no-psf', action='store_true', help='Tắt đồng bộ PSF')
  ap.add_argument('--no-scale', action='store_true', help='Tắt chuẩn hóa thông lượng tuyến tính')
  ap.add_argument('--injection', action='store_true', help='Chạy kiểm thử Injection-Recovery')
  ap.add_argument('--demo', action='store_true', help='Chạy thử nghiệm trên dữ liệu giả lập')
  ap.add_argument('--trials', type=int, default=8, help='Số lượt lặp mỗi điểm lưới Injection-Recovery')
  args = ap.parse_args()

  if args.demo:
    stack = make_synthetic_stack()
    rows = injection_recovery(
        stack, fwhm=3.0, threshold_sigma=args.sigma, n_trials=args.trials
    )
    plot_injection_results(rows)
  elif args.folder:
    find_planet_x_candidates(
        args.folder,
        threshold_sigma=args.sigma,
        ref_index=args.ref,
        do_psf=not args.no_psf,
        do_scale=not args.no_scale,
        exact_reproject=args.exact,
        run_injection=args.injection,
        n_trials=args.trials,
    )
  else:
    run_gui_tool()


if __name__ == '__main__':
  main()