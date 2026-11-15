"""Tiện ích hiển thị ảnh thiên văn: kéo giãn độ sáng, tô màu, và bộ chớp ảnh (blink) chạy
hoàn toàn bằng JavaScript phía trình duyệt (không cần Streamlit rerun mỗi khung)."""
from __future__ import annotations

import base64
import io
import json

import matplotlib
import numpy as np
from PIL import Image


def stretch_limits(img, mode: str):
    v = np.asarray(img, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    if mode == "ZScale":
        try:
            from astropy.visualization import ZScaleInterval
            lo, hi = ZScaleInterval().get_limits(v)
        except Exception:
            lo, hi = np.percentile(v, [5, 99.5])
    elif mode == "Log":
        pos = v[v > 0]
        lo = np.percentile(pos, 5) if pos.size else 0.0
        hi = np.percentile(v, 99.5)
    else:  # Asinh, Linear
        lo, hi = np.percentile(v, [1, 99.5])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def to_unit(img, mode: str, limits=None):
    """Đưa ảnh về thang 0..1 theo kiểu kéo giãn; NaN được giữ nguyên."""
    lo, hi = limits if limits is not None else stretch_limits(img, mode)
    t = np.clip((np.asarray(img, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    if mode == "Log":
        t = np.log10(1.0 + 1000.0 * t) / np.log10(1001.0)
    elif mode == "Asinh":
        t = np.arcsinh(10.0 * t) / np.arcsinh(10.0)
    return t


def colorize(unit_img, cmap_name: str, scale: int = 1) -> Image.Image:
    """Ảnh 0..1 -> ảnh PIL RGB (gốc tọa độ ở dưới-trái như origin='lower'); NaN = đen."""
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad((0.0, 0.0, 0.0, 1.0))
    rgba = cmap(np.ma.masked_invalid(unit_img))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)[::-1]
    im = Image.fromarray(np.ascontiguousarray(rgb))
    if scale > 1:
        im = im.resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    return im


def to_data_uri(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_frames(images, mode, cmap_name, per_frame=True, target_px=520):
    """Trả về list data-URI PNG cho các khung (đã phóng to để nhìn rõ)."""
    h, w = images[0].shape
    scale = int(max(1, min(10, target_px // max(h, w))))
    if per_frame:
        units = [to_unit(im, mode) for im in images]
    else:
        allv = np.concatenate([im[np.isfinite(im)] for im in images])
        lim = stretch_limits(allv, mode)
        units = [to_unit(im, mode, lim) for im in images]
    return [to_data_uri(colorize(u, cmap_name, scale)) for u in units]


def blink_html(uris, labels, interval_ms=350, max_width=560) -> str:
    n = len(uris)
    return f"""
<div style="font-family:sans-serif;text-align:center">
  <img id="fr" style="width:100%;max-width:{max_width}px;image-rendering:pixelated;
       border:1px solid #666;border-radius:6px"/>
  <div style="margin:8px 0">
    <span id="lab" style="background:#111;color:#eee;padding:3px 12px;border-radius:12px;
          font-size:13px;display:inline-block"></span>
  </div>
  <div style="display:flex;gap:10px;align-items:center;justify-content:center;
       max-width:{max_width}px;margin:0 auto">
    <button id="pp" style="padding:4px 14px;cursor:pointer">⏸ Pause</button>
    <input id="sl" type="range" min="0" max="{max(n - 1, 0)}" value="0" style="flex:1"/>
  </div>
</div>
<script>
const F = {json.dumps(uris)};
const L = {json.dumps(labels, ensure_ascii=False)};
let i = 0, play = true;
const img = document.getElementById('fr'), lab = document.getElementById('lab'),
      sl = document.getElementById('sl'), pp = document.getElementById('pp');
function show(k) {{ i = k; img.src = F[i]; lab.textContent = (i + 1) + '/' + F.length + '  |  ' + L[i]; sl.value = i; }}
setInterval(() => {{ if (play && F.length > 1) show((i + 1) % F.length); }}, {int(interval_ms)});
sl.oninput = () => {{ play = false; pp.textContent = '▶ Play'; show(parseInt(sl.value)); }};
pp.onclick = () => {{ play = !play; pp.textContent = play ? '⏸ Pause' : '▶ Play'; }};
show(0);
</script>
"""