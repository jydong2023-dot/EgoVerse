#!/usr/bin/env python3
"""Render 3D videos for processed Mecka Zarr pose/keypoint streams."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import imageio
import imageio_ffmpeg
import matplotlib
import mediapy as mpy
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.scripts.mecka_process.zarr_to_lerobot3 import (  # noqa: E402
    DEFAULT_INPUT_ZARR,
    decode_numeric_sharded_array,
    load_json,
)

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

LOGGER = logging.getLogger(__name__)

PoseGroup = Literal[
    "head_pose", "obs_keypoints", "obs_wrist_pose", "obs_ee_pose", "all"
]

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

DEFAULT_OUTPUT_DIR = Path("/home/djy/EgoVerse/data/keypoint_inspection/mecka_3d_viz")
DEFAULT_COMBINED_VIDEO_NAME = "mecka_3d_all_views.mp4"


def default_panel_groups() -> list[PoseGroup]:
    return ["head_pose", "obs_keypoints", "obs_wrist_pose", "obs_ee_pose", "all"]


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a WXYZ quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_mecka_pose_arrays(episode_dir: Path) -> dict[str, np.ndarray]:
    keys = [
        "obs_head_pose",
        "left.obs_keypoints",
        "right.obs_keypoints",
        "left.obs_wrist_pose",
        "right.obs_wrist_pose",
        "left.obs_ee_pose",
        "right.obs_ee_pose",
    ]
    return {key: decode_numeric_sharded_array(episode_dir / key) for key in keys}


def collect_points_for_bounds(arrays: dict[str, np.ndarray]) -> np.ndarray:
    points: list[np.ndarray] = []
    for key, values in arrays.items():
        values = np.asarray(values)
        if key.endswith("obs_keypoints"):
            if values.ndim != 2 or values.shape[1] != 63:
                continue
            points.append(values.reshape(-1, 21, 3).reshape(-1, 3))
        elif values.ndim == 2 and values.shape[1] >= 3:
            points.append(values[:, :3])
    if not points:
        raise ValueError("No 3D points found for bounds")
    all_points = np.concatenate(points, axis=0)
    finite = np.isfinite(all_points).all(axis=1)
    nonzero = ~np.isclose(all_points, 0.0).all(axis=1)
    filtered = all_points[finite & nonzero]
    return filtered if len(filtered) else all_points[finite]


def compute_equal_bounds(
    points: np.ndarray, margin_ratio: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.nanmax(maxs - mins))
    if not np.isfinite(span) or span < 1e-6:
        span = 1.0
    half = span * (0.5 + margin_ratio)
    return center - half, center + half


def _pose_origin(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose[:3], dtype=np.float64)


def draw_pose_frame(
    ax: Any,
    pose: np.ndarray,
    *,
    label: str,
    color: str,
    axis_length: float,
) -> None:
    origin = _pose_origin(pose)
    rotation = quaternion_wxyz_to_matrix(np.asarray(pose[3:7], dtype=np.float64))
    axis_colors = ("red", "green", "blue")
    for axis_idx, axis_color in enumerate(axis_colors):
        endpoint = origin + rotation[:, axis_idx] * axis_length
        ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=axis_color,
            linewidth=2.0,
        )
    ax.scatter(origin[0], origin[1], origin[2], color=color, s=40, label=label)
    ax.text(origin[0], origin[1], origin[2], label, color=color, fontsize=8)


def draw_hand_keypoints(
    ax: Any,
    keypoints_flat: np.ndarray,
    *,
    label: str,
    color: str,
    draw_lines: bool,
    label_indices: bool = False,
    index_prefix: str = "",
    index_fontsize: float = 6.0,
) -> None:
    keypoints = np.asarray(keypoints_flat, dtype=np.float64).reshape(21, 3)
    finite = np.isfinite(keypoints).all(axis=1)
    valid = finite & (keypoints[:, 2] > -0.999)
    pts = keypoints[valid]
    if pts.size == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=18, label=label)
    if label_indices:
        for i in range(21):
            if not valid[i]:
                continue
            p = keypoints[i]
            ax.text(
                p[0],
                p[1],
                p[2],
                f"{index_prefix}{i}",
                color=color,
                fontsize=index_fontsize,
                ha="left",
                va="bottom",
            )
    if draw_lines:
        for start_idx, end_idx in HAND_CONNECTIONS:
            if valid[start_idx] and valid[end_idx]:
                start = keypoints[start_idx]
                end = keypoints[end_idx]
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=color,
                    linewidth=1.2,
                    alpha=0.85,
                )


def setup_3d_axis(
    ax: Any,
    *,
    title: str,
    bounds: tuple[np.ndarray, np.ndarray],
    elev: float,
    azim: float,
) -> None:
    lower, upper = bounds
    ax.set_title(title)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def render_group_frame(
    arrays: dict[str, np.ndarray],
    frame_idx: int,
    *,
    group: PoseGroup,
    bounds: tuple[np.ndarray, np.ndarray],
    figsize: tuple[float, float],
    dpi: int,
    elev: float,
    azim: float,
    draw_lines: bool,
    axis_length: float,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axis(
        ax,
        title=f"{group} | frame {frame_idx}",
        bounds=bounds,
        elev=elev,
        azim=azim,
    )

    if group in ("head_pose", "all"):
        draw_pose_frame(
            ax,
            arrays["obs_head_pose"][frame_idx],
            label="head",
            color="black",
            axis_length=axis_length,
        )

    show_kp_idx = group == "obs_keypoints"
    if group in ("obs_keypoints", "all"):
        draw_hand_keypoints(
            ax,
            arrays["left.obs_keypoints"][frame_idx],
            label="left keypoints",
            color="limegreen",
            draw_lines=draw_lines,
            label_indices=show_kp_idx,
            index_prefix="L",
            index_fontsize=7.0 if show_kp_idx else 6.0,
        )
        draw_hand_keypoints(
            ax,
            arrays["right.obs_keypoints"][frame_idx],
            label="right keypoints",
            color="royalblue",
            draw_lines=draw_lines,
            label_indices=show_kp_idx,
            index_prefix="R",
            index_fontsize=7.0 if show_kp_idx else 6.0,
        )

    if group in ("obs_wrist_pose", "all"):
        draw_pose_frame(
            ax,
            arrays["left.obs_wrist_pose"][frame_idx],
            label="left wrist",
            color="darkgreen",
            axis_length=axis_length,
        )
        draw_pose_frame(
            ax,
            arrays["right.obs_wrist_pose"][frame_idx],
            label="right wrist",
            color="navy",
            axis_length=axis_length,
        )

    if group in ("obs_ee_pose", "all"):
        draw_pose_frame(
            ax,
            arrays["left.obs_ee_pose"][frame_idx],
            label="left ee",
            color="orange",
            axis_length=axis_length,
        )
        draw_pose_frame(
            ax,
            arrays["right.obs_ee_pose"][frame_idx],
            label="right ee",
            color="purple",
            axis_length=axis_length,
        )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=7)

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        .reshape(height, width, 4)[..., :3]
        .copy()
    )
    plt.close(fig)
    return frame


def draw_group_on_axis(
    ax: Any,
    arrays: dict[str, np.ndarray],
    frame_idx: int,
    *,
    group: PoseGroup,
    bounds: tuple[np.ndarray, np.ndarray],
    elev: float,
    azim: float,
    draw_lines: bool,
    axis_length: float,
) -> None:
    setup_3d_axis(
        ax,
        title=f"{group}",
        bounds=bounds,
        elev=elev,
        azim=azim,
    )

    if group in ("head_pose", "all"):
        draw_pose_frame(
            ax,
            arrays["obs_head_pose"][frame_idx],
            label="head",
            color="black",
            axis_length=axis_length,
        )

    show_kp_idx = group == "obs_keypoints"
    if group in ("obs_keypoints", "all"):
        draw_hand_keypoints(
            ax,
            arrays["left.obs_keypoints"][frame_idx],
            label="left keypoints",
            color="limegreen",
            draw_lines=draw_lines,
            label_indices=show_kp_idx,
            index_prefix="L",
            index_fontsize=5.5 if show_kp_idx else 6.0,
        )
        draw_hand_keypoints(
            ax,
            arrays["right.obs_keypoints"][frame_idx],
            label="right keypoints",
            color="royalblue",
            draw_lines=draw_lines,
            label_indices=show_kp_idx,
            index_prefix="R",
            index_fontsize=5.5 if show_kp_idx else 6.0,
        )

    if group in ("obs_wrist_pose", "all"):
        draw_pose_frame(
            ax,
            arrays["left.obs_wrist_pose"][frame_idx],
            label="left wrist",
            color="darkgreen",
            axis_length=axis_length,
        )
        draw_pose_frame(
            ax,
            arrays["right.obs_wrist_pose"][frame_idx],
            label="right wrist",
            color="navy",
            axis_length=axis_length,
        )

    if group in ("obs_ee_pose", "all"):
        draw_pose_frame(
            ax,
            arrays["left.obs_ee_pose"][frame_idx],
            label="left ee",
            color="orange",
            axis_length=axis_length,
        )
        draw_pose_frame(
            ax,
            arrays["right.obs_ee_pose"][frame_idx],
            label="right ee",
            color="purple",
            axis_length=axis_length,
        )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=6)


def render_combined_frame(
    arrays: dict[str, np.ndarray],
    frame_idx: int,
    *,
    groups: list[PoseGroup],
    bounds: tuple[np.ndarray, np.ndarray],
    figsize: tuple[float, float],
    dpi: int,
    elev: float,
    azim: float,
    draw_lines: bool,
    axis_length: float,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(f"Mecka 3D Visualization | frame {frame_idx}", fontsize=14)
    rows, cols = 2, 3
    for panel_idx, group in enumerate(groups):
        ax = fig.add_subplot(rows, cols, panel_idx + 1, projection="3d")
        draw_group_on_axis(
            ax,
            arrays,
            frame_idx,
            group=group,
            bounds=bounds,
            elev=elev,
            azim=azim,
            draw_lines=draw_lines,
            axis_length=axis_length,
        )

    for empty_idx in range(len(groups), rows * cols):
        ax = fig.add_subplot(rows, cols, empty_idx + 1)
        ax.axis("off")

    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    frame = (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        .reshape(height, width, 4)[..., :3]
        .copy()
    )
    plt.close(fig)
    return frame


def write_combined_video(
    arrays: dict[str, np.ndarray],
    *,
    output_path: Path,
    groups: list[PoseGroup],
    bounds: tuple[np.ndarray, np.ndarray],
    frame_count: int,
    fps: int,
    frame_stride: int,
    max_frames: int | None,
    figsize: tuple[float, float],
    dpi: int,
    elev: float,
    azim: float,
    draw_lines: bool,
    axis_length: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = frame_count if max_frames is None else min(max_frames, frame_count)
    writer = imageio.get_writer(str(output_path), fps=fps, macro_block_size=1)
    written = 0
    try:
        for frame_idx in range(0, limit, frame_stride):
            frame = render_combined_frame(
                arrays,
                frame_idx,
                groups=groups,
                bounds=bounds,
                figsize=figsize,
                dpi=dpi,
                elev=elev,
                azim=azim,
                draw_lines=draw_lines,
                axis_length=axis_length,
            )
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    if written == 0:
        raise ValueError(f"No frames written to {output_path}")
    return written


def write_group_video(
    arrays: dict[str, np.ndarray],
    *,
    output_path: Path,
    group: PoseGroup,
    bounds: tuple[np.ndarray, np.ndarray],
    frame_count: int,
    fps: int,
    frame_stride: int,
    max_frames: int | None,
    figsize: tuple[float, float],
    dpi: int,
    elev: float,
    azim: float,
    draw_lines: bool,
    axis_length: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = frame_count if max_frames is None else min(max_frames, frame_count)
    writer = imageio.get_writer(str(output_path), fps=fps, macro_block_size=1)
    written = 0
    try:
        for frame_idx in range(0, limit, frame_stride):
            frame = render_group_frame(
                arrays,
                frame_idx,
                group=group,
                bounds=bounds,
                figsize=figsize,
                dpi=dpi,
                elev=elev,
                azim=azim,
                draw_lines=draw_lines,
                axis_length=axis_length,
            )
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    if written == 0:
        raise ValueError(f"No frames written to {output_path}")
    return written


def visualize_episode_3d(
    episode_dir: Path,
    output_dir: Path,
    *,
    groups: list[PoseGroup],
    fps: int | None,
    frame_stride: int,
    max_frames: int | None,
    figsize: tuple[float, float],
    dpi: int,
    elev: float,
    azim: float,
    draw_lines: bool,
    combined: bool = True,
    combined_name: str = DEFAULT_COMBINED_VIDEO_NAME,
) -> dict[str, Path]:
    arrays = load_mecka_pose_arrays(episode_dir)
    root_meta = load_json(episode_dir / "zarr.json")
    attrs = root_meta.get("attributes", {})
    frame_count = min(array.shape[0] for array in arrays.values())
    output_fps = int(fps or attrs.get("fps") or 30)
    bounds = compute_equal_bounds(collect_points_for_bounds(arrays))
    axis_length = max(float((bounds[1] - bounds[0]).max()) * 0.04, 1e-3)

    outputs: dict[str, Path] = {}
    if combined:
        out = output_dir / combined_name
        written = write_combined_video(
            arrays,
            output_path=out,
            groups=groups,
            bounds=bounds,
            frame_count=frame_count,
            fps=output_fps,
            frame_stride=frame_stride,
            max_frames=max_frames,
            figsize=figsize,
            dpi=dpi,
            elev=elev,
            azim=azim,
            draw_lines=draw_lines,
            axis_length=axis_length,
        )
        LOGGER.info("Saved %s (%s frames)", out, written)
        outputs["combined"] = out
        return outputs

    for group in groups:
        out = output_dir / f"{group}_3d.mp4"
        written = write_group_video(
            arrays,
            output_path=out,
            group=group,
            bounds=bounds,
            frame_count=frame_count,
            fps=output_fps,
            frame_stride=frame_stride,
            max_frames=max_frames,
            figsize=figsize,
            dpi=dpi,
            elev=elev,
            azim=azim,
            draw_lines=draw_lines,
            axis_length=axis_length,
        )
        LOGGER.info("Saved %s (%s frames)", out, written)
        outputs[group] = out
    return outputs


def parse_groups(raw: str) -> list[PoseGroup]:
    if raw == "all_outputs":
        return default_panel_groups()
    values = [value.strip() for value in raw.split(",") if value.strip()]
    valid = {"head_pose", "obs_keypoints", "obs_wrist_pose", "obs_ee_pose", "all"}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise ValueError(f"Invalid groups: {invalid}; valid={sorted(valid)}")
    return values  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D visualize processed Mecka Zarr pose/keypoint streams."
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_INPUT_ZARR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-video",
        type=str,
        default=DEFAULT_COMBINED_VIDEO_NAME,
        help="Filename for the combined multi-window video.",
    )
    parser.add_argument(
        "--separate-videos",
        action="store_true",
        help="Write one video per visualization group instead of one combined multi-window video.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="all_outputs",
        help="Comma-separated subset: head_pose,obs_keypoints,obs_wrist_pose,obs_ee_pose,all; default writes all 5 videos.",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fig-width", type=float, default=15.0)
    parser.add_argument("--fig-height", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-62.0)
    parser.add_argument(
        "--draw-lines", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    outputs = visualize_episode_3d(
        args.episode_dir,
        args.output_dir,
        groups=parse_groups(args.groups),
        fps=args.fps,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        figsize=(args.fig_width, args.fig_height),
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        draw_lines=args.draw_lines,
        combined=not args.separate_videos,
        combined_name=args.output_video,
    )
    for group, path in outputs.items():
        print(f"{group}: {path}")


if __name__ == "__main__":
    main()
