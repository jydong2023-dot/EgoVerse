#!/usr/bin/env python3
"""Single 3D figure per frame: left/right obs_keypoints with joint indices; tight axis limits; MP4 output.

Skeleton edges are off by default; pass --draw-lines to show HAND_CONNECTIONS between joints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapy as mpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.scripts.mecka_process.visualize_mecka_zarr_3d import (  # noqa: E402
    draw_hand_keypoints,
)
from egomimic.scripts.mecka_process.zarr_to_lerobot3 import (  # noqa: E402
    decode_numeric_sharded_array,
)

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

DEFAULT_EPISODE = Path(
    "/home/djy/EgoVerse/data/mecka/fold-clothes/692e711e5aae241ad236e7f6"
)

# Default extra viewpoints for first-frame PNGs (elevation°, azimuth°) — independent of --elev/--azim video camera.
DEFAULT_MULTI_VIEWS: list[tuple[float, float]] = [
    (20.0, -60.0),
    (20.0, 120.0),
    (15.0, -135.0),
    (60.0, 45.0),
    (85.0, -90.0),
]


def parse_view_angles(spec: str | None) -> list[tuple[float, float]]:
    if spec is None or not spec.strip():
        return list(DEFAULT_MULTI_VIEWS)
    out: list[tuple[float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad view angle (need elev:azim): {part!r}")
        a, b = part.split(":", 1)
        out.append((float(a), float(b)))
    if not out:
        return list(DEFAULT_MULTI_VIEWS)
    return out


def _valid_mask_21(keypoints_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (21,3) array and boolean mask length 21 for drawable joints."""
    k = np.asarray(keypoints_flat, dtype=np.float64).reshape(21, 3)
    finite = np.isfinite(k).all(axis=1)
    valid = finite & (k[:, 2] > -0.999)
    return k, valid


def axis_aligned_bounds_two_hands(
    left_flat: np.ndarray,
    right_flat: np.ndarray,
    *,
    margin_ratio: float,
    min_axis_span: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tight per-axis min/max (not a large equal cube), so the hand fills the view."""
    pieces: list[np.ndarray] = []
    for flat in (left_flat, right_flat):
        k, valid = _valid_mask_21(flat)
        if np.any(valid):
            pieces.append(k[valid])
    if not pieces:
        z = np.array([-0.15, -0.15, -0.15], dtype=np.float64)
        span = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        return z, z + span
    pts = np.concatenate(pieces, axis=0)
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    span = np.maximum(hi - lo, min_axis_span)
    pad = span * margin_ratio
    return lo - pad, hi + pad


def setup_axes_tight(
    ax: Any,
    *,
    title: str,
    lower: np.ndarray,
    upper: np.ndarray,
    elev: float,
    azim: float,
) -> None:
    ax.set_title(title)
    ax.set_xlim(float(lower[0]), float(upper[0]))
    ax.set_ylim(float(lower[1]), float(upper[1]))
    ax.set_zlim(float(lower[2]), float(upper[2]))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(
            (
                float(upper[0] - lower[0]),
                float(upper[1] - lower[1]),
                float(upper[2] - lower[2]),
            )
        )
    except Exception:
        pass


def render_frame(
    left_row: np.ndarray,
    right_row: np.ndarray,
    frame_idx: int,
    total_frames: int,
    *,
    figsize: tuple[float, float],
    dpi: int,
    elev: float,
    azim: float,
    draw_lines: bool,
    margin_ratio: float,
    min_axis_span: float,
    index_fontsize: float,
    title_suffix: str | None = None,
) -> np.ndarray:
    lower, upper = axis_aligned_bounds_two_hands(
        left_row,
        right_row,
        margin_ratio=margin_ratio,
        min_axis_span=min_axis_span,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    title = f"obs_keypoints | frame {frame_idx} / {total_frames - 1}"
    if title_suffix:
        title = f"{title} | {title_suffix}"
    setup_axes_tight(
        ax,
        title=title,
        lower=lower,
        upper=upper,
        elev=elev,
        azim=azim,
    )
    draw_hand_keypoints(
        ax,
        left_row,
        label="left",
        color="limegreen",
        draw_lines=draw_lines,
        label_indices=True,
        index_prefix="L",
        index_fontsize=index_fontsize,
    )
    draw_hand_keypoints(
        ax,
        right_row,
        label="right",
        color="royalblue",
        draw_lines=draw_lines,
        label_indices=True,
        index_prefix="R",
        index_fontsize=index_fontsize,
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    if legend_labels:
        by_label = dict(zip(legend_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        .reshape(height, width, 4)[..., :3]
        .copy()
    )
    plt.close(fig)
    return frame


def save_first_frame_multiview_pngs(
    left_row: np.ndarray,
    right_row: np.ndarray,
    total_frames: int,
    out_dir: Path,
    views: list[tuple[float, float]],
    *,
    figsize: tuple[float, float],
    dpi: int,
    draw_lines: bool,
    margin_ratio: float,
    min_axis_span: float,
    index_fontsize: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (elev, azim) in enumerate(views):
        rgb = render_frame(
            left_row,
            right_row,
            0,
            total_frames,
            figsize=figsize,
            dpi=dpi,
            elev=elev,
            azim=azim,
            draw_lines=draw_lines,
            margin_ratio=margin_ratio,
            min_axis_span=min_axis_span,
            index_fontsize=index_fontsize,
            title_suffix=f"elev={elev:.0f}° azim={azim:.0f}°",
        )
        fname = (
            f"frame000_view{idx:02d}_elev{int(round(elev))}_azim{int(round(azim))}.png"
        )
        imageio.imwrite(str(out_dir / fname), rgb)

    print(f"Saved {len(views)} first-frame PNGs to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize left/right obs_keypoints in 3D with joint indices; tight bounds; save MP4."
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output video path (default: <episode-dir>/obs_keypoints_3d_indexed.mp4)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-60.0)
    parser.add_argument("--figsize", type=float, nargs=2, default=(8.0, 7.0))
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.12,
        help="Padding as a fraction of per-axis span (smaller = tighter crop).",
    )
    parser.add_argument(
        "--min-axis-span",
        type=float,
        default=0.04,
        help="Minimum axis length (meters) when the hand is flat along one dimension.",
    )
    parser.add_argument("--index-fontsize", type=float, default=7.0)
    parser.add_argument(
        "--draw-lines",
        action="store_true",
        help="Draw skeleton edges between keypoints (default: points and labels only).",
    )
    parser.add_argument(
        "--first-frame-images-dir",
        type=Path,
        default=None,
        help="Directory for multi-angle PNGs of frame 0 (default: <episode-dir>/obs_keypoints_first_frame_views).",
    )
    parser.add_argument(
        "--view-angles",
        type=str,
        default=None,
        help="Comma-separated elev:azim pairs for first-frame PNGs, e.g. '20:-60,60:45,85:-90'. "
        "If omitted, uses built-in default views.",
    )
    parser.add_argument(
        "--no-first-frame-images",
        action="store_true",
        help="Do not write multi-view PNGs for frame 0.",
    )
    args = parser.parse_args()

    ep = args.episode_dir
    if not ep.is_dir():
        raise FileNotFoundError(ep)

    left = decode_numeric_sharded_array(ep / "left.obs_keypoints")
    right = decode_numeric_sharded_array(ep / "right.obs_keypoints")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"Length mismatch left {left.shape[0]} vs right {right.shape[0]}"
        )
    n = left.shape[0]
    if n == 0:
        raise ValueError("Empty keypoint stream")

    out = args.out
    if out is None:
        out = ep / "obs_keypoints_3d_indexed.mp4"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    limit = n if args.max_frames is None else min(args.max_frames, n)
    draw_lines = args.draw_lines
    views = parse_view_angles(args.view_angles)

    if not args.no_first_frame_images:
        img_dir = args.first_frame_images_dir
        if img_dir is None:
            img_dir = ep / "obs_keypoints_first_frame_views"
        img_dir = img_dir.resolve()
        save_first_frame_multiview_pngs(
            left[0],
            right[0],
            limit,
            img_dir,
            views,
            figsize=(args.figsize[0], args.figsize[1]),
            dpi=args.dpi,
            draw_lines=draw_lines,
            margin_ratio=args.margin_ratio,
            min_axis_span=args.min_axis_span,
            index_fontsize=args.index_fontsize,
        )

    writer = imageio.get_writer(str(out), fps=args.fps, macro_block_size=1)
    written = 0
    try:
        for frame_idx in range(0, limit, args.stride):
            rgb = render_frame(
                left[frame_idx],
                right[frame_idx],
                frame_idx,
                limit,
                figsize=(args.figsize[0], args.figsize[1]),
                dpi=args.dpi,
                elev=args.elev,
                azim=args.azim,
                draw_lines=draw_lines,
                margin_ratio=args.margin_ratio,
                min_axis_span=args.min_axis_span,
                index_fontsize=args.index_fontsize,
            )
            writer.append_data(rgb)
            written += 1
    finally:
        writer.close()

    if written == 0:
        raise ValueError("No frames written.")
    print(f"Wrote {written} frames to {out}", flush=True)


if __name__ == "__main__":
    main()
