#!/usr/bin/env python3
"""Overlay Mecka wrist poses on every image frame and save the rendered images."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.scripts.mecka_process.zarr_to_lerobot3 import (  # noqa: E402
    DEFAULT_IMAGE_KEY,
    ImageShardReader,
    decode_jpeg_rgb,
    decode_numeric_sharded_array,
    load_json,
)
from egomimic.utils.egomimicUtils import (  # noqa: E402
    MECKA_INTRINSICS,
    cam_frame_to_cam_pixels,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_EPISODE_DIR = Path(
    "/home/djy/EgoVerse/data/mecka/fold-clothes/692e711e5aae241ad236e7f6"
)
DEFAULT_OUTPUT_DIRNAME = "obs_wrist_pose_overlay_frames"


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
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


def pose_xyzwxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def resolve_intrinsics(
    image_shape: tuple[int, int, int], intrinsics: dict[str, Any]
) -> np.ndarray:
    h, w = image_shape[:2]
    if intrinsics:
        src_w = float(intrinsics.get("w") or w)
        src_h = float(intrinsics.get("h") or h)
        sx = w / src_w if src_w else 1.0
        sy = h / src_h if src_h else 1.0
        fx = (
            float(
                intrinsics.get("fl_x")
                or intrinsics.get("focal_length")
                or MECKA_INTRINSICS[0, 0]
            )
            * sx
        )
        fy = (
            float(
                intrinsics.get("fl_y")
                or intrinsics.get("focal_length")
                or MECKA_INTRINSICS[1, 1]
            )
            * sy
        )
        cx = float(intrinsics.get("cx") or MECKA_INTRINSICS[0, 2]) * sx
        cy = float(intrinsics.get("cy") or MECKA_INTRINSICS[1, 2]) * sy
        return np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    default = np.asarray(MECKA_INTRINSICS, dtype=np.float64).copy()
    if (w, h) != (640, 360):
        default[0, 0] *= w / 640.0
        default[1, 1] *= h / 360.0
        default[0, 2] *= w / 640.0
        default[1, 2] *= h / 360.0
    return default


def project_points(points_cam: np.ndarray, intrinsics_matrix: np.ndarray) -> np.ndarray:
    pixels = cam_frame_to_cam_pixels(
        np.asarray(points_cam, dtype=np.float64), intrinsics_matrix
    )
    return np.asarray(pixels[:, :2], dtype=np.float64)


def wrist_axes_points(
    xyz_head: np.ndarray, rot_head: np.ndarray, axis_length: float
) -> np.ndarray:
    axes = np.stack(
        [
            xyz_head,
            xyz_head + rot_head[:, 0] * axis_length,
            xyz_head + rot_head[:, 1] * axis_length,
            xyz_head + rot_head[:, 2] * axis_length,
        ],
        axis=0,
    )
    return axes


def draw_wrist_pose(
    image_bgr: np.ndarray,
    pose_cam: np.ndarray,
    *,
    intrinsics_matrix: np.ndarray,
    label: str,
    anchor_color: tuple[int, int, int],
    axis_length: float,
) -> np.ndarray:
    pose_tf = pose_xyzwxyz_to_matrix(pose_cam)
    xyz_cam = pose_tf[:3, 3]
    rot_cam = pose_tf[:3, :3]
    points_cam = wrist_axes_points(xyz_cam, rot_cam, axis_length)
    pixels = project_points(points_cam, intrinsics_matrix)
    h, w = image_bgr.shape[:2]

    def in_bounds(point: np.ndarray) -> bool:
        return (
            np.isfinite(point[0])
            and np.isfinite(point[1])
            and 0 <= point[0] < w
            and 0 <= point[1] < h
        )

    origin = pixels[0]
    if not in_bounds(origin):
        return image_bgr

    axis_colors = [
        (0, 0, 255),  # x axis in red
        (0, 255, 0),  # y axis in green
        (255, 0, 0),  # z axis in blue
    ]
    origin_xy = (int(round(origin[0])), int(round(origin[1])))
    for endpoint, color in zip(pixels[1:], axis_colors):
        if in_bounds(endpoint):
            endpoint_xy = (int(round(endpoint[0])), int(round(endpoint[1])))
            cv2.line(image_bgr, origin_xy, endpoint_xy, color, 3, cv2.LINE_AA)

    cv2.circle(image_bgr, origin_xy, 10, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image_bgr, origin_xy, 7, anchor_color, -1, cv2.LINE_AA)
    text_xy = (origin_xy[0] + 12, origin_xy[1] - 12)
    cv2.putText(
        image_bgr,
        label,
        text_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        label,
        text_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        anchor_color,
        3,
        cv2.LINE_AA,
    )
    return image_bgr


def overlay_episode_wrist_poses(
    episode_dir: Path,
    output_dir: Path,
    *,
    image_key: str,
    axis_length: float,
    max_frames: int | None,
) -> int:
    root_meta = load_json(episode_dir / "zarr.json")
    attrs = root_meta.get("attributes", {})
    intrinsics = attrs.get("intrinsics", {})
    left_wrist = decode_numeric_sharded_array(episode_dir / "left.obs_wrist_pose")
    right_wrist = decode_numeric_sharded_array(episode_dir / "right.obs_wrist_pose")
    image_reader = ImageShardReader(episode_dir / image_key)

    frame_count = min(left_wrist.shape[0], right_wrist.shape[0], image_reader.length)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count <= 0:
        raise ValueError(f"No frames available in {episode_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(frame_count):
        image_rgb = decode_jpeg_rgb(image_reader.get(frame_idx))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        intrinsics_matrix = resolve_intrinsics(image_bgr.shape, intrinsics)
        image_bgr = draw_wrist_pose(
            image_bgr,
            left_wrist[frame_idx],
            intrinsics_matrix=intrinsics_matrix,
            label="L",
            anchor_color=(255, 64, 255),
            axis_length=axis_length,
        )
        image_bgr = draw_wrist_pose(
            image_bgr,
            right_wrist[frame_idx],
            intrinsics_matrix=intrinsics_matrix,
            label="R",
            anchor_color=(255, 255, 0),
            axis_length=axis_length,
        )
        out_path = output_dir / f"frame_{frame_idx:06d}.png"
        if not cv2.imwrite(str(out_path), image_bgr):
            raise ValueError(f"Failed to write image: {out_path}")

    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay left/right obs_wrist_pose on each Mecka image frame."
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-key", type=str, default=DEFAULT_IMAGE_KEY)
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Axis length in meters for each wrist local frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    output_dir = args.output_dir or (args.episode_dir / DEFAULT_OUTPUT_DIRNAME)
    written = overlay_episode_wrist_poses(
        args.episode_dir,
        output_dir,
        image_key=args.image_key,
        axis_length=args.axis_length,
        max_frames=args.max_frames,
    )
    LOGGER.info("Saved %s overlay frames to %s", written, output_dir)


if __name__ == "__main__":
    main()
