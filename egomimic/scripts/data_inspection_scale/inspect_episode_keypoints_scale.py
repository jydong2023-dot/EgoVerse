#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio
import imageio_ffmpeg
import mediapy as mpy
import numpy as np
import torch

# Allow direct execution without pre-setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.rldb.embodiment.human import (  # noqa: E402
    Aria,
    Scale,
    _build_aria_keypoints_bimanual_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset  # noqa: E402
from egomimic.utils.egomimicUtils import INTRINSICS  # noqa: E402
from egomimic.utils.pose_utils import _split_keypoints  # noqa: E402

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


DEFAULT_EPISODE = Path(
    "/home/djy/EgoVerse/data/scale/flagship_scoop_granular/2026-05-01-06-57-26-956787/"
)
KEYPOINT_KEYS = ("left.obs_keypoints", "right.obs_keypoints")
DEFAULT_VIDEO_OUT = Path(
    "/home/djy/EgoVerse/data/keypoint_inspection/selected_keypoints.mp4"
)
DEFAULT_HANDS = ("left", "right")
SCALE_INTRINSICS_KEY = "scale_episode"
SCALE_CAMERA_INTRINSICS = {
    "fx": 174.22564697265625,
    "fy": 209.0707763671875,
    "cx": 340.4532063802083,
    "cy": 260.15646972656253,
    "width": 640,
    "height": 480,
}
SCALE_INTRINSICS_SOURCE_SIZE = (
    float(SCALE_CAMERA_INTRINSICS["width"]),
    float(SCALE_CAMERA_INTRINSICS["height"]),
)
SCALE_INTRINSICS = np.array(
    [
        [SCALE_CAMERA_INTRINSICS["fx"], 0.0, SCALE_CAMERA_INTRINSICS["cx"], 0.0],
        [0.0, SCALE_CAMERA_INTRINSICS["fy"], SCALE_CAMERA_INTRINSICS["cy"], 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def parse_keypoint_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None

    text = raw.strip()
    if not text:
        return None

    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [part.strip() for part in text.split(",") if part.strip()]

    indices = [int(v) for v in values]
    if not indices:
        return None
    invalid = [idx for idx in indices if idx < 0 or idx > 20]
    if invalid:
        raise ValueError(
            f"keypoint indices must be in [0, 20], got invalid values: {invalid}"
        )
    return sorted(set(indices))


def normalize_hands(raw_hands: list[str] | None) -> list[str]:
    hands = raw_hands or list(DEFAULT_HANDS)
    normalized = []
    for hand in hands:
        hand = hand.lower().strip()
        if hand not in DEFAULT_HANDS:
            raise ValueError(
                f"Unsupported hand '{hand}', expected one of: {DEFAULT_HANDS}"
            )
        if hand not in normalized:
            normalized.append(hand)
    return normalized


def image_hw_from_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 4:
        if shape[1] in (1, 3):
            return int(shape[2]), int(shape[3])
        if shape[-1] in (1, 3):
            return int(shape[1]), int(shape[2])
    if len(shape) == 3:
        if shape[0] in (1, 3):
            return int(shape[1]), int(shape[2])
        if shape[-1] in (1, 3):
            return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported image shape for intrinsic scaling: {shape}")


def intrinsics_matrix_from_dict(params: dict[str, Any]) -> np.ndarray:
    fx = float(params["fx"])
    fy = float(params["fy"])
    cx = float(params["cx"])
    cy = float(params["cy"])
    return np.array(
        [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def load_episode_camera_intrinsics(
    episode_dir: Path,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Load Scale episode camera intrinsics from zarr metadata, with script defaults as fallback."""
    attrs = load_json(episode_dir / "zarr.json").get("attributes", {})
    camera_intrinsics = attrs.get("camera_intrinsics")
    if isinstance(camera_intrinsics, dict) and {
        "fx",
        "fy",
        "cx",
        "cy",
    }.issubset(camera_intrinsics):
        width = float(camera_intrinsics.get("width", SCALE_CAMERA_INTRINSICS["width"]))
        height = float(
            camera_intrinsics.get("height", SCALE_CAMERA_INTRINSICS["height"])
        )
        return intrinsics_matrix_from_dict(camera_intrinsics), (width, height)
    return SCALE_INTRINSICS.copy(), SCALE_INTRINSICS_SOURCE_SIZE


def scale_intrinsics_for_image_shape(
    intrinsics: np.ndarray,
    source_size: tuple[float, float],
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Scale camera intrinsics from the reference resolution to the visualization image size."""
    image_h, image_w = image_hw_from_shape(tuple(int(v) for v in image_shape))
    src_w, src_h = source_size
    sx = image_w / src_w
    sy = image_h / src_h
    scaled = intrinsics.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def register_scale_intrinsics(episode_dir: Path, image_shape: tuple[int, ...]) -> str:
    """Register Scale camera intrinsics, scaled to the current image resolution."""
    base_intrinsics, source_size = load_episode_camera_intrinsics(episode_dir)
    INTRINSICS[SCALE_INTRINSICS_KEY] = scale_intrinsics_for_image_shape(
        base_intrinsics, source_size, image_shape
    )
    return SCALE_INTRINSICS_KEY


def inspect_feature_metadata(episode_dir: Path) -> list[str]:
    lines: list[str] = []
    root_meta = load_json(episode_dir / "zarr.json")
    attrs = root_meta.get("attributes", {})
    features = attrs.get("features", {})

    lines.append(f"episode: {episode_dir}")
    lines.append(f"embodiment: {attrs.get('embodiment')}")
    lines.append(f"total_frames(attr): {attrs.get('total_frames')}")
    lines.append("")
    lines.append("top-level feature check:")

    for key in KEYPOINT_KEYS:
        feature = features.get(key)
        exists = feature is not None
        lines.append(f"  - {key}: exists={format_bool(exists)}")
        if exists:
            lines.append(
                f"    dtype={feature.get('dtype')}, per_frame_shape={feature.get('shape')}"
            )
            lines.append(
                f"    has literal nested 'keypoints' field: {format_bool('keypoints' in feature)}"
            )

    has_literal_keypoints_feature = "keypoints" in features
    lines.append(
        f"  - literal top-level feature named 'keypoints': {format_bool(has_literal_keypoints_feature)}"
    )
    lines.append("")
    return lines


def inspect_array_node(episode_dir: Path, key: str) -> list[str]:
    lines: list[str] = []
    array_dir = episode_dir / key
    meta_path = array_dir / "zarr.json"

    lines.append(f"{key}:")
    lines.append(f"  path_exists={format_bool(array_dir.exists())}")
    lines.append(f"  zarr_json_exists={format_bool(meta_path.exists())}")
    if not meta_path.exists():
        lines.append("  result: missing array metadata")
        lines.append("")
        return lines

    meta = load_json(meta_path)
    node_type = meta.get("node_type")
    shape = meta.get("shape")
    dtype = meta.get("data_type")

    lines.append(f"  node_type={node_type}")
    lines.append(f"  shape={shape}")
    lines.append(f"  data_type={dtype}")
    lines.append(
        f"  has attributes.keypoints={format_bool('keypoints' in meta.get('attributes', {}))}"
    )
    lines.append(
        f"  has top-level 'keypoints' field in node metadata={format_bool('keypoints' in meta)}"
    )

    if node_type == "array":
        lines.append(
            "  interpretation: this node is a raw Zarr array, not an object/dict with a 'keypoints' key."
        )
        lines.append(
            "  interpretation: keypoint values are stored directly in the array rows."
        )
        if shape and len(shape) == 2 and shape[1] == 63:
            lines.append(
                "  interpretation: each frame stores 21 * 3 flattened MANO keypoints."
            )

    chunk_path = array_dir / "c" / "0" / "0"
    lines.append(f"  first_chunk_exists={format_bool(chunk_path.exists())}")
    lines.append("")
    return lines


def try_read_samples_with_zarr(episode_dir: Path, sample_frames: int) -> list[str]:
    lines: list[str] = ["optional sample read:"]
    try:
        import zarr  # type: ignore
    except ImportError:
        lines.append("  zarr import failed; metadata-only inspection completed.")
        lines.append(
            "  install project dependencies or run inside an environment with zarr to read sample values."
        )
        lines.append("")
        return lines

    for key in KEYPOINT_KEYS:
        arr = zarr.open(str(episode_dir / key), mode="r")
        frame_count = min(sample_frames, arr.shape[0])
        preview = arr[:frame_count]
        first_row = preview[0].tolist() if frame_count > 0 else []
        lines.append(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
        lines.append(f"    sample_frames_read={frame_count}")
        lines.append(f"    first_frame_first_9_values={first_row[:9]}")
    lines.append("")
    return lines


def load_keypoint_array(episode_dir: Path, hand: str) -> np.ndarray:
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "zarr is required for visualization. Run this script in the project environment."
        ) from exc

    arr = zarr.open(str(episode_dir / f"{hand}.obs_keypoints"), mode="r")
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 63:
        raise ValueError(
            f"{hand}.obs_keypoints expected shape (T, 63), got {tuple(data.shape)}"
        )
    return data.reshape(data.shape[0], 21, 3)


def build_dataset(episode_dir: Path) -> ZarrDataset:
    # Aria keymap includes obs_head_pose required by head-frame keypoint transforms.
    key_map = Aria.get_keymap(keymap_mode="keypoints")
    transform_list = _build_aria_keypoints_bimanual_transform_list(
        stride=Scale.ACTION_STRIDE,
        is_quat=False,
    )
    return ZarrDataset(
        Episode_path=episode_dir,
        key_map=key_map,
        transform_list=transform_list,
    )


def invalidate_unselected_keypoints(
    actions_keypoints: torch.Tensor,
    indices: list[int] | None,
    hands: list[str],
) -> torch.Tensor:
    if indices is None and set(hands) == set(DEFAULT_HANDS):
        return actions_keypoints

    actions_np = (
        actions_keypoints.detach().cpu().numpy()
        if isinstance(actions_keypoints, torch.Tensor)
        else np.asarray(actions_keypoints)
    )
    filtered = actions_np.copy()
    selected = set(range(21) if indices is None else indices)
    enabled_hands = set(hands)

    if filtered.shape[-1] == 140:
        left_xyz, left_rot, left_keypoints, right_xyz, right_rot, right_keypoints = (
            _split_keypoints(filtered, wrist_in_data=True, is_quat=True)
        )
        left_pose = np.concatenate([left_xyz, left_rot], axis=-1)
        right_pose = np.concatenate([right_xyz, right_rot], axis=-1)
    elif filtered.shape[-1] == 138:
        left_xyz, left_rot, left_keypoints, right_xyz, right_rot, right_keypoints = (
            _split_keypoints(filtered, wrist_in_data=True, is_quat=False)
        )
        left_pose = np.concatenate([left_xyz, left_rot], axis=-1)
        right_pose = np.concatenate([right_xyz, right_rot], axis=-1)
    else:
        raise ValueError(
            f"Unsupported actions_keypoints shape {tuple(filtered.shape)}; expected last dim 138 or 140."
        )

    left_keypoints = left_keypoints.reshape(*left_keypoints.shape[:-1], 21, 3)
    right_keypoints = right_keypoints.reshape(*right_keypoints.shape[:-1], 21, 3)
    invalid_point = np.array([0.0, 0.0, -1.0], dtype=filtered.dtype)

    for hand_name, keypoints_arr in (
        ("left", left_keypoints),
        ("right", right_keypoints),
    ):
        if hand_name not in enabled_hands:
            keypoints_arr[...] = invalid_point
            continue
        for kp_idx in range(21):
            if kp_idx not in selected:
                keypoints_arr[..., kp_idx, :] = invalid_point

    left_flat = left_keypoints.reshape(*left_keypoints.shape[:-2], 63)
    right_flat = right_keypoints.reshape(*right_keypoints.shape[:-2], 63)
    combined = np.concatenate([left_pose, left_flat, right_pose, right_flat], axis=-1)
    return torch.as_tensor(
        combined,
        device=actions_keypoints.device,
        dtype=actions_keypoints.dtype,
    )


def save_visualization_video(
    episode_dir: Path,
    video_out: Path,
    keypoint_indices: list[int] | None,
    hands: list[str],
    frame_start: int,
    frame_end: int | None,
    frame_stride: int,
    fps: int,
    *,
    vis_index: bool = False,
    draw_skeleton_lines: bool = False,
) -> list[str]:
    dataset = build_dataset(episode_dir)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    base_intrinsics, source_size = load_episode_camera_intrinsics(episode_dir)
    intrinsics_key = SCALE_INTRINSICS_KEY
    intrinsics_matrix = base_intrinsics.copy()

    total_frames = len(dataset)
    start = max(0, frame_start)
    end = total_frames if frame_end is None else min(frame_end, total_frames)
    if start >= end:
        raise ValueError(
            f"Invalid frame range: start={start}, end={end}, total_frames={total_frames}"
        )
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0, got {frame_stride}")

    if vis_index:
        if keypoint_indices is None:
            raise ValueError("--vis-index requires --keypoint-indices (e.g. 0,4,8).")
        filter_indices = keypoint_indices
        label_indices = keypoint_indices
    else:
        filter_indices = None
        label_indices = None

    writer = imageio.get_writer(str(video_out), fps=fps)
    written = 0
    try:
        for frame_idx in range(start, end, frame_stride):
            sample = dataset[frame_idx]
            batch = {}
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    batch[key] = torch.from_numpy(value).unsqueeze(0)
                elif isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0)
                else:
                    batch[key] = [value]

            batch["actions_keypoints"] = invalidate_unselected_keypoints(
                batch["actions_keypoints"],
                indices=filter_indices,
                hands=hands,
            )

            image_shape = tuple(int(v) for v in batch[Aria.VIZ_IMAGE_KEY].shape)
            intrinsics_key = register_scale_intrinsics(episode_dir, image_shape)
            intrinsics_matrix = INTRINSICS[intrinsics_key]

            old_intrinsics_key = Aria.VIZ_INTRINSICS_KEY
            try:
                Aria.VIZ_INTRINSICS_KEY = intrinsics_key
                vis = Aria.viz_transformed_batch(
                    batch,
                    mode="keypoints",
                    viz_batch_key="actions_keypoints",
                    vis_line=draw_skeleton_lines,
                    labeled_keypoint_indices=label_indices,
                )
            finally:
                Aria.VIZ_INTRINSICS_KEY = old_intrinsics_key
            writer.append_data(np.asarray(vis, dtype=np.uint8))
            written += 1
    finally:
        writer.close()

    if written == 0:
        raise ValueError(f"No video frames were written for {episode_dir}")

    lines = ["visualization:"]
    lines.append(f"  selected_hands={hands}")
    lines.append(f"  vis_index={format_bool(vis_index)}")
    lines.append(
        "  keypoints_shown="
        + (
            "all joints (0-20); no MANO index labels"
            if not vis_index
            else f"subset only + L#/R# labels on {filter_indices}"
        )
    )
    lines.append(f"  draw_skeleton_lines={format_bool(draw_skeleton_lines)}")
    lines.append(
        f"  intrinsics={intrinsics_key} "
        f"(fx={intrinsics_matrix[0, 0]:.1f}, fy={intrinsics_matrix[1, 1]:.1f}, "
        f"cx={intrinsics_matrix[0, 2]:.1f}, cy={intrinsics_matrix[1, 2]:.1f}; "
        f"reference={int(source_size[0])}x{int(source_size[1])} Scale camera_intrinsics)"
    )
    lines.append(
        "  labeled_keypoints="
        + (
            "(none; use --vis-index with --keypoint-indices)"
            if not vis_index
            else f"L/R on {label_indices}"
        )
    )
    lines.append(f"  frame_range=({start}, {end}, stride={frame_stride})")
    lines.append(f"  fps={fps}")
    lines.append(f"  saved_video={video_out}")
    lines.append(f"  frames_written={written}")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect EgoVerse episode keypoint arrays and check whether they contain a literal 'keypoints' key."
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=DEFAULT_EPISODE,
        help="Path to one EgoVerse episode directory.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=1,
        help="How many frames to sample when zarr is available.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also generate an overlay video from the original keypoint visualization.",
    )
    parser.add_argument(
        "--keypoint-indices",
        type=str,
        default=None,
        help="MANO joint indices 0–20; e.g. '0,4,8'. Used only when --vis-index is set (subset overlay + L#/R# labels).",
    )
    parser.add_argument(
        "--vis-index",
        action="store_true",
        help="If set: show only --keypoint-indices joints and draw L#/R# labels. Default: all 21 joints, no index labels.",
    )
    parser.add_argument(
        "--draw-lines",
        action="store_true",
        help="Draw colored skeleton edges between keypoints (default: points only, no limb lines).",
    )
    parser.add_argument(
        "--hands",
        nargs="+",
        default=None,
        help="Hands to visualize: left right. Default visualizes both.",
    )
    parser.add_argument(
        "--video-out",
        type=Path,
        default=DEFAULT_VIDEO_OUT,
        help="Output mp4 path for visualization video.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Start frame for visualization.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="End frame (exclusive) for visualization. Default uses all available frames.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Stride used when sampling frames for visualization video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for saved visualization video.",
    )
    args = parser.parse_args()

    episode_dir = args.episode_dir
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")

    keypoint_indices = parse_keypoint_indices(args.keypoint_indices)
    hands = normalize_hands(args.hands)

    lines: list[str] = []
    lines.extend(inspect_feature_metadata(episode_dir))
    for key in KEYPOINT_KEYS:
        lines.extend(inspect_array_node(episode_dir, key))
    lines.extend(try_read_samples_with_zarr(episode_dir, args.sample_frames))
    if args.visualize:
        lines.extend(
            save_visualization_video(
                episode_dir=episode_dir,
                hands=hands,
                keypoint_indices=keypoint_indices,
                video_out=args.video_out,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                frame_stride=args.frame_stride,
                fps=args.fps,
                vis_index=args.vis_index,
                draw_skeleton_lines=args.draw_lines,
            )
        )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
