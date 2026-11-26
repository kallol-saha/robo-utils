"""Image panels of what a policy observes, for videos (numpy + PIL; no window, no GPU).

    point_cloud_views   orthographic top and side views of a point cloud, with a goal (center and
                        heading) and gripper points, on a 25 cm grid
    depth_grid          depth images tiled in rows, near bright
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

POINT_RGB = np.array([217, 107, 31])  # orange points
GOAL_RGB = (30, 150, 70)
GRIPPER_RGB = (40, 90, 200)
BACKGROUND = 250
GRID = 228


def point_cloud_views(
    points: np.ndarray,
    goal: np.ndarray,
    goal_heading: np.ndarray,
    grippers: np.ndarray,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    point_px: int,
    font: Path,
    title: str = "top",
) -> np.ndarray:
    """Top view (x right, y up) over side view (x right, z up), `height` px tall; the width follows
    the ranges and is even (H.264 frames). Returns (height, W, 3) uint8.

    points (P, 3) shaded by height (top) or y (side), goal (3,) a green cross, goal_heading (3,) its
    x axis (a 15 cm line in the top view), grippers (R, 3) blue dots; all in one frame [m].
    `font`: a TrueType file for the labels; `title`: the top view's label."""
    if not Path(font).exists():
        raise FileNotFoundError(f"font {font} not found")
    (x0, x1), (y0, y1), (z0, z1) = x_range, y_range, z_range
    s = height / ((y1 - y0) + (z1 - z0))  # px per m
    width = 2 * round((x1 - x0) * s / 2)
    top_h = round((y1 - y0) * s)
    img = Image.new("RGB", (width, height), (BACKGROUND,) * 3)
    draw = ImageDraw.Draw(img)
    r = int(point_px)

    def px(x: float, v: float, lo: float, hi: float, top: int, h: int) -> tuple[float, float]:
        return (x - x0) * s, top + (hi - v) / (hi - lo) * h

    views = [(1, y0, y1, 0, top_h, 2), (2, z0, z1, top_h, height - top_h, 1)]  # axis, range, shade
    for axis, lo, hi, top, h, shade_axis in views:
        for gx in np.arange(np.ceil(x0 * 4) / 4, x1, 0.25):
            draw.line([px(gx, hi, lo, hi, top, h), px(gx, lo, lo, hi, top, h)], fill=(GRID,) * 3)
        for gw in np.arange(np.ceil(lo * 4) / 4, hi, 0.25):
            draw.line([px(x0, gw, lo, hi, top, h), px(x1, gw, lo, hi, top, h)], fill=(GRID,) * 3)
        shade = points[:, shade_axis]
        t = (shade - shade.min()) / max(float(np.ptp(shade)), 1e-6)  # 0: low / far, 1: high / near
        order = np.argsort(t)
        for p, tt in zip(points[order], t[order], strict=True):
            u, v = px(float(p[0]), float(p[axis]), lo, hi, top, h)
            rgb = tuple(int(q) for q in POINT_RGB * (0.55 + 0.45 * tt))
            draw.rectangle([u - r, v - r, u + r, v + r], fill=rgb)
        gu, gv = px(float(goal[0]), float(goal[axis]), lo, hi, top, h)
        k = 3 * r + 4
        draw.line([gu - k, gv - k, gu + k, gv + k], fill=GOAL_RGB, width=3)
        draw.line([gu - k, gv + k, gu + k, gv - k], fill=GOAL_RGB, width=3)
        if axis == 1:
            hu, hv = px(
                float(goal[0] + 0.15 * goal_heading[0]),
                float(goal[1] + 0.15 * goal_heading[1]),
                lo,
                hi,
                top,
                h,
            )
            draw.line([gu, gv, hu, hv], fill=GOAL_RGB, width=3)
        for q in grippers:
            u, v = px(float(q[0]), float(q[axis]), lo, hi, top, h)
            draw.ellipse([u - k / 2, v - k / 2, u + k / 2, v + k / 2], fill=GRIPPER_RGB)
        draw.line([0, top + h - 1, width, top + h - 1], fill=(120,) * 3, width=2)
    face = ImageFont.truetype(str(font), max(12, height // 28))
    draw.text((8, 6), title, fill=(60,) * 3, font=face)
    draw.text((8, top_h + 6), "side", fill=(60,) * 3, font=face)
    return np.asarray(img)


def depth_grid(depth: np.ndarray, rows: int, height: int) -> np.ndarray:
    """Depth images (C, h, w) in [0, 1] tiled in `rows` rows (C / rows per row), each scaled to
    height / rows px (nearest), near bright, padded to `height`: (height, W, 3) uint8."""
    c, h, w = depth.shape
    if c % rows:
        raise ValueError(f"{c} images don't fill {rows} rows")
    cols, tile = c // rows, height // rows
    size = (tile * w // h, tile)  # PIL: (width, height)
    gray = ((1.0 - np.clip(depth, 0.0, 1.0)) * 255).astype(np.uint8)
    tiles = [np.asarray(Image.fromarray(g).resize(size, Image.Resampling.NEAREST)) for g in gray]
    grid = np.concatenate(
        [np.concatenate(tiles[r * cols : (r + 1) * cols], 1) for r in range(rows)], 0
    )
    grid = np.pad(grid[:height], ((0, max(0, height - grid.shape[0])), (0, 0)))
    return np.repeat(grid[..., None], 3, -1)
