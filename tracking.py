"""Liên kết các phát hiện qua nhiều epoch thành quỹ đạo chuyển động thẳng đều (tracklet).

Toàn bộ tính toán trên lưới pixel chung (các khung đã reproject về cùng WCS).
Thuần numpy, không phụ thuộc astropy.
"""
from __future__ import annotations

import numpy as np


def link_tracklets(dets, t_yr, v_max_px_yr, tol_px=1.5, min_epochs=3, min_dt_yr=1.0 / 365.25):
    """dets: list (theo epoch) các mảng (N_i, 2) toạ độ (x, y) pixel.
    t_yr: thời điểm mỗi epoch (năm). Trả về list tracklet (dict), tốt nhất trước.

    Mỗi tracklet: id, members [(epoch, det_index)], n, x0, y0 (tại t=0), vx, vy (px/năm),
    rms (px), speed (px/năm), disp (px, tổng dịch chuyển trên toàn bộ khoảng thời gian),
    moving (bool).
    """
    t = np.asarray(t_yr, dtype=float)
    dets = [np.asarray(d, dtype=float).reshape(-1, 2) for d in dets]
    nf = len(dets)
    span = float(t.max() - t.min()) if nf else 0.0
    cands = []

    for a in range(nf):
        for b in range(a + 1, nf):
            dt = t[b] - t[a]
            if abs(dt) < min_dt_yr or len(dets[a]) == 0 or len(dets[b]) == 0:
                continue
            V = (dets[b][None, :, :] - dets[a][:, None, :]) / dt          # (Na, Nb, 2)
            speed = np.hypot(V[..., 0], V[..., 1])
            for i, j in zip(*np.where(speed <= v_max_px_yr)):
                v = V[i, j]
                members = []
                for k in range(nf):
                    if len(dets[k]) == 0:
                        continue
                    pred = dets[a][i] + v * (t[k] - t[a])
                    d = np.hypot(dets[k][:, 0] - pred[0], dets[k][:, 1] - pred[1])
                    m = int(np.argmin(d))
                    if d[m] <= tol_px:
                        members.append((k, m))
                if len(members) >= min_epochs:
                    cands.append(members)

    fitted = []
    for members in cands:
        tm = np.array([t[k] for k, _ in members])
        if np.ptp(tm) < min_dt_yr:
            continue
        xs = np.array([dets[k][m, 0] for k, m in members])
        ys = np.array([dets[k][m, 1] for k, m in members])
        vx, x0 = np.polyfit(tm, xs, 1)
        vy, y0 = np.polyfit(tm, ys, 1)
        rms = float(np.sqrt(np.mean((xs - (x0 + vx * tm)) ** 2 + (ys - (y0 + vy * tm)) ** 2)))
        if rms <= tol_px:
            fitted.append((members, float(x0), float(y0), float(vx), float(vy), rms))

    fitted.sort(key=lambda r: (-len(r[0]), r[5]))
    used, out = set(), []
    for members, x0, y0, vx, vy, rms in fitted:
        if any(mk in used for mk in members):
            continue
        used.update(members)
        sp = float(np.hypot(vx, vy))
        disp = sp * span
        out.append({
            "id": len(out) + 1, "members": members, "n": len(members),
            "x0": x0, "y0": y0, "vx": vx, "vy": vy, "rms": rms,
            "speed": sp, "disp": disp,
            "moving": bool(disp > 2.0 * max(tol_px, rms)),
        })
    return out