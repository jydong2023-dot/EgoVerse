#!/usr/bin/env python3
"""Overlay Mecka left/right hand keypoints on every image frame and save PNGs."""

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
DEFAULT_OUTPUT_DIRNAME = "obs_keypoints_overlay_frames"
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
LEFT_HAND_COLOR = (0, 255, 0)
RIGHT_HAND_COLOR = (0, 0, 255)


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


def project_keypoints_to_pixels(
    keypoints_cam: np.ndarray, intrinsics_matrix: np.ndarray
) -> np.ndarray:
    pixels = cam_frame_to_cam_pixels(
        np.asarray(keypoints_cam, dtype=np.float64), intrinsics_matrix
    )
    return np.asarray(pixels[:, :2], dtype=np.float64)


def draw_hand_skeleton(
    image_bgr: np.ndarray,
    keypoints_2d: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    def in_bounds(point: np.ndarray) -> bool:
        return (
            np.isfinite(point[0])
            and np.isfinite(point[1])
            and 0 <= point[0] < w
            and 0 <= point[1] < h
        )

    for start_idx, end_idx in HAND_CONNECTIONS:
        start = keypoints_2d[start_idx]
        end = keypoints_2d[end_idx]
        if in_bounds(start) and in_bounds(end):
            cv2.line(
                image_bgr,
                (int(round(start[0])), int(round(start[1]))),
                (int(round(end[0])), int(round(end[1]))),
                color,
                2,
                cv2.LINE_AA,
            )

    for idx, point in enumerate(keypoints_2d):
        if in_bounds(point):
            radius = 8 if idx == 0 else 5
            cv2.circle(
                image_bgr,
                (int(round(point[0])), int(round(point[1]))),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )
    return image_bgr


def overlay_episode_hand_keypoints(
    episode_dir: Path,
    output_dir: Path,
    *,
    image_key: str,
    max_frames: int | None,
) -> int:
    root_meta = load_json(episode_dir / "zarr.json")
    attrs = root_meta.get("attributes", {})
    intrinsics = attrs.get("intrinsics", {})
    left = decode_numeric_sharded_array(episode_dir / "left.obs_keypoints").reshape(
        -1, 21, 3
    )
    right = decode_numeric_sharded_array(episode_dir / "right.obs_keypoints").reshape(
        -1, 21, 3
    )
    image_reader = ImageShardReader(episode_dir / image_key)

    frame_count = min(left.shape[0], right.shape[0], image_reader.length)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    if frame_count <= 0:
        raise ValueError(f"No frames available in {episode_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(frame_count):
        image_rgb = decode_jpeg_rgb(image_reader.get(frame_idx))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        intrinsics_matrix = resolve_intrinsics(image_bgr.shape, intrinsics)
        left_2d = project_keypoints_to_pixels(left[frame_idx], intrinsics_matrix)
        right_2d = project_keypoints_to_pixels(right[frame_idx], intrinsics_matrix)
        image_bgr = draw_hand_skeleton(image_bgr, left_2d, LEFT_HAND_COLOR)
        image_bgr = draw_hand_skeleton(image_bgr, right_2d, RIGHT_HAND_COLOR)
        cv2.putText(
            image_bgr,
            f"frame {frame_idx:06d}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        out_path = output_dir / f"frame_{frame_idx:06d}.png"
        if not cv2.imwrite(str(out_path), image_bgr):
            raise ValueError(f"Failed to write image: {out_path}")
    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Mecka left/right obs_keypoints on each image frame."
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-key", type=str, default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    output_dir = args.output_dir or (args.episode_dir / DEFAULT_OUTPUT_DIRNAME)
    written = overlay_episode_hand_keypoints(
        args.episode_dir,
        output_dir,
        image_key=args.image_key,
        max_frames=args.max_frames,
    )
    LOGGER.info("Saved %s keypoint-overlay frames to %s", written, output_dir)


if __name__ == "__main__":
    main()
