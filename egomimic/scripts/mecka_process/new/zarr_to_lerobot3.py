#!/usr/bin/env python3
"""Convert processed Mecka Zarr episodes to LeRobot and render keypoint overlays."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import imageio
import imageio_ffmpeg
import mediapy as mpy
import numpy as np
from numcodecs import CRC32C, VLenBytes, Zstd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_LEROBOT = REPO_ROOT / "external" / "lerobot"
for path in (REPO_ROOT, EXTERNAL_LEROBOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

LOGGER = logging.getLogger(__name__)
_WARNED_ARIA_FALLBACK = False
DEFAULT_INPUT_ZARR = Path(
    "/home/djy/EgoVerse/data/mecka/fold-clothes/692e711e5aae241ad236e7f6"
)
DEFAULT_VIDEO_OUT = Path(
    "/home/djy/EgoVerse/data/keypoint_inspection/mecka_zarr_overlay.mp4"
)
DEFAULT_REPO_ID = "mecka/fold-clothes-zarr"
DEFAULT_IMAGE_KEY = "images.front_1"
LEROBOT_IMAGE_KEY = "observations.images.front_img_1"
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _decode_sharded_entries(raw: bytes, num_subchunks: int) -> np.ndarray:
    index_nbytes = num_subchunks * 2 * 8
    index_payload = CRC32C().decode(raw[-(index_nbytes + 4) :])
    return np.frombuffer(index_payload, dtype="<u8").reshape(num_subchunks, 2)


def _is_shard_index_sentinel(offset: int | np.uint64, nbytes: int | np.uint64) -> bool:
    max_u64 = np.iinfo(np.uint64).max
    off = np.uint64(offset)
    nb = np.uint64(nbytes)
    return bool(off == max_u64 or nb == max_u64 or nb == 0)


def _read_sharded_chunk(
    raw: bytes,
    entries: np.ndarray,
    inner_shape: tuple[int, ...],
    dtype: np.dtype,
    shard_rows: int,
    shard_cols: int,
) -> np.ndarray:
    zstd = Zstd()
    blocks: list[np.ndarray | None] = []
    for off_u, nbytes_u in entries:
        if _is_shard_index_sentinel(off_u, nbytes_u):
            blocks.append(None)
            continue
        off = int(off_u)
        nbytes = int(nbytes_u)
        decoded = zstd.decode(raw[off : off + nbytes])
        blocks.append(np.frombuffer(decoded, dtype=dtype).reshape(inner_shape))

    if len(inner_shape) == 1:
        valid = [block for block in blocks if block is not None]
        if not valid:
            raise ValueError("No valid 1D subchunks found")
        return np.concatenate(valid, axis=0)

    rows: list[np.ndarray] = []
    for row_idx in range(shard_rows):
        row_blocks = [
            blocks[row_idx * shard_cols + col_idx] for col_idx in range(shard_cols)
        ]
        if all(block is None for block in row_blocks):
            continue
        if any(block is None for block in row_blocks):
            raise ValueError(f"Partially missing shard row {row_idx}")
        rows.append(np.concatenate(row_blocks, axis=1))
    if not rows:
        raise ValueError("No valid 2D subchunks found")
    return np.concatenate(rows, axis=0)


def decode_numeric_sharded_array(array_dir: Path) -> np.ndarray:
    """Read numeric Zarr v3 sharding_indexed arrays produced by Mecka conversion."""
    meta = load_json(array_dir / "zarr.json")
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"]).newbyteorder("<")
    outer_shape = tuple(
        int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]
    )
    inner_shape = tuple(
        int(v) for v in meta["codecs"][0]["configuration"]["chunk_shape"]
    )

    if len(shape) != 2:
        raise ValueError(f"Expected 2D numeric array at {array_dir}, got shape={shape}")

    outer_rows = _ceil_div(shape[0], outer_shape[0])
    outer_cols = _ceil_div(shape[1], outer_shape[1])
    row_chunks: list[np.ndarray] = []
    for outer_row in range(outer_rows):
        col_chunks: list[np.ndarray] = []
        for outer_col in range(outer_cols):
            chunk_path = array_dir / "c" / str(outer_row) / str(outer_col)
            if not chunk_path.exists():
                continue
            raw = chunk_path.read_bytes()
            shard_rows = _ceil_div(outer_shape[0], inner_shape[0])
            shard_cols = _ceil_div(outer_shape[1], inner_shape[1])
            entries = _decode_sharded_entries(raw, shard_rows * shard_cols)
            decoded = _read_sharded_chunk(
                raw, entries, inner_shape, dtype, shard_rows, shard_cols
            )
            col_chunks.append(decoded)
        if col_chunks:
            row_chunks.append(np.concatenate(col_chunks, axis=1))

    if not row_chunks:
        raise ValueError(f"No chunks decoded for {array_dir}")

    full = np.concatenate(row_chunks, axis=0)
    slices = tuple(slice(0, min(shape[i], full.shape[i])) for i in range(len(shape)))
    return np.asarray(full[slices])


class ImageShardReader:
    """Read JPEG bytes from a Zarr v3 sharding_indexed variable-length image array."""

    def __init__(self, array_dir: Path):
        meta = load_json(array_dir / "zarr.json")
        self.length = int(meta["shape"][0])
        outer_shape = tuple(
            int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]
        )
        inner_shape = tuple(
            int(v) for v in meta["codecs"][0]["configuration"]["chunk_shape"]
        )
        if inner_shape != (1,):
            raise ValueError(
                f"Expected image inner chunk_shape=(1,), got {inner_shape}"
            )

        self._chunks: list[tuple[bytes, np.ndarray]] = []
        outer_rows = _ceil_div(self.length, outer_shape[0])
        num_subchunks = _ceil_div(outer_shape[0], inner_shape[0])
        for outer_row in range(outer_rows):
            chunk_path = array_dir / "c" / str(outer_row)
            if not chunk_path.exists():
                continue
            raw = chunk_path.read_bytes()
            entries = _decode_sharded_entries(raw, num_subchunks)
            self._chunks.append((raw, entries))
        if not self._chunks:
            raise ValueError(f"No image chunks decoded for {array_dir}")

        self._outer_len = outer_shape[0]
        self._zstd = Zstd()
        self._vlen = VLenBytes()

    def get(self, index: int) -> bytes:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        chunk_idx = index // self._outer_len
        local_idx = index % self._outer_len
        raw, entries = self._chunks[chunk_idx]
        offset, nbytes = entries[local_idx]
        if _is_shard_index_sentinel(offset, nbytes):
            raise ValueError(f"Image frame {index} points to an unused shard slot")
        off = int(offset)
        nb = int(nbytes)
        decoded = self._zstd.decode(raw[off : off + nb])
        return self._vlen.decode(decoded)[0]


def decode_jpeg_rgb(jpeg_payload: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_payload, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG payload")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prestack_future(values: np.ndarray, chunk_size: int) -> np.ndarray:
    """Stack each frame with future values, repeating the final frame for padding."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError(
            f"values must have leading time dimension and feature dims, got {values.shape}"
        )
    out = np.empty((values.shape[0], chunk_size, *values.shape[1:]), dtype=values.dtype)
    last = values.shape[0] - 1
    for t in range(values.shape[0]):
        for offset in range(chunk_size):
            out[t, offset] = values[min(t + offset, last)]
    return out


def build_lerobot_features(
    image_shape: tuple[int, int, int],
    chunk_size: int,
    encode_video: bool,
) -> dict[str, dict[str, Any]]:
    h, w, c = image_shape
    return {
        LEROBOT_IMAGE_KEY: {
            "dtype": "video" if encode_video else "image",
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        },
        "observations.state.ee_pose": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["left_right_xyz_quat_wxyz"],
        },
        "observations.state.head_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["xyz_quat_wxyz"],
        },
        "actions_ee_pose_world": {
            "dtype": "prestacked_float32",
            "shape": (chunk_size, 14),
            "names": ["chunk_size", "left_right_xyz_quat_wxyz"],
        },
        "actions_ee_keypoints_world": {
            "dtype": "prestacked_float32",
            "shape": (chunk_size, 126),
            "names": ["chunk_size", "left_right_hand_keypoints"],
        },
        "metadata.embodiment": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["dim_0"],
        },
    }


def _embodiment_value() -> int:
    try:
        from egomimic.rldb.embodiment.embodiment import EMBODIMENT

        return int(EMBODIMENT.MECKA_BIMANUAL.value)
    except Exception:
        return 0


class ProcessedMeckaZarr:
    def __init__(self, episode_dir: Path, image_key: str = DEFAULT_IMAGE_KEY):
        self.episode_dir = episode_dir
        self.image_key = image_key
        root_meta = load_json(episode_dir / "zarr.json")
        self.attrs = root_meta.get("attributes", {})
        self.left_keypoints = decode_numeric_sharded_array(
            episode_dir / "left.obs_keypoints"
        ).astype(np.float32)
        self.right_keypoints = decode_numeric_sharded_array(
            episode_dir / "right.obs_keypoints"
        ).astype(np.float32)
        self.head_pose = decode_numeric_sharded_array(
            episode_dir / "obs_head_pose"
        ).astype(np.float32)
        self.left_pose = decode_numeric_sharded_array(
            episode_dir / "left.obs_ee_pose"
        ).astype(np.float32)
        self.right_pose = decode_numeric_sharded_array(
            episode_dir / "right.obs_ee_pose"
        ).astype(np.float32)
        self.image_reader = ImageShardReader(episode_dir / image_key)
        self.total_frames = min(
            int(self.attrs.get("total_frames") or 0),
            self.left_keypoints.shape[0],
            self.right_keypoints.shape[0],
            self.head_pose.shape[0],
            self.left_pose.shape[0],
            self.right_pose.shape[0],
            self.image_reader.length,
        )
        if self.total_frames <= 0:
            raise ValueError(f"Could not determine frame count for {episode_dir}")

    @property
    def fps(self) -> int:
        return int(self.attrs.get("fps") or 30)

    @property
    def task(self) -> str:
        task = self.attrs.get("task_description") or self.attrs.get("task_name")
        return str(task or self.attrs.get("episode_id") or self.episode_dir.name)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (
            tuple(int(v) for v in self.attrs["features"][self.image_key]["shape"])
            if self.image_key in self.attrs.get("features", {})
            else (360, 640, 3)
        )

    def image(self, index: int) -> np.ndarray:
        return decode_jpeg_rgb(self.image_reader.get(index))

    def keypoints_126(self) -> np.ndarray:
        return np.concatenate(
            [
                self.left_keypoints[: self.total_frames],
                self.right_keypoints[: self.total_frames],
            ],
            axis=-1,
        ).astype(np.float32)

    def poses_14(self) -> np.ndarray:
        return np.concatenate(
            [
                self.left_pose[: self.total_frames],
                self.right_pose[: self.total_frames],
            ],
            axis=-1,
        ).astype(np.float32)


def convert_zarr_to_lerobot(
    input_zarr: Path,
    output_dir: Path,
    repo_id: str,
    *,
    chunk_size: int,
    encode_video: bool,
    overwrite: bool,
    image_key: str,
) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    episode = ProcessedMeckaZarr(input_zarr, image_key=image_key)
    features = build_lerobot_features(episode.image_shape, chunk_size, encode_video)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=episode.fps,
        root=output_dir,
        robot_type=str(episode.attrs.get("embodiment", "MECKA_BIMANUAL")),
        features=features,
        use_videos=encode_video,
    )

    keypoint_actions = prestack_future(episode.keypoints_126(), chunk_size)
    pose_actions = prestack_future(episode.poses_14(), chunk_size)
    embodiment = np.array([_embodiment_value()], dtype=np.int32)
    for frame_index in range(episode.total_frames):
        frame = {
            LEROBOT_IMAGE_KEY: episode.image(frame_index),
            "observations.state.ee_pose": episode.poses_14()[frame_index],
            "observations.state.head_pose": episode.head_pose[frame_index],
            "actions_ee_pose_world": pose_actions[frame_index],
            "actions_ee_keypoints_world": keypoint_actions[frame_index],
            "metadata.embodiment": embodiment,
            "timestamp": frame_index / episode.fps,
        }
        dataset.add_frame(frame)

    dataset.save_episode(task=episode.task, encode_videos=encode_video)
    dataset.consolidate(run_compute_stats=False)
    _write_mecka_metadata(output_dir, input_zarr, episode.attrs)
    LOGGER.info("Saved LeRobot dataset to %s", output_dir)
    return output_dir


def _write_mecka_metadata(
    output_dir: Path, input_zarr: Path, attrs: dict[str, Any]
) -> None:
    info_path = output_dir / "meta" / "info.json"
    if not info_path.exists():
        return
    info = load_json(info_path)
    info["mecka"] = {
        "source_zarr": str(input_zarr),
        "episode_id": attrs.get("episode_id"),
        "user_id": attrs.get("user_id"),
        "duration": attrs.get("duration"),
        "environment_id": attrs.get("environment_id"),
        "scene_id": attrs.get("scene_id"),
        "scene_desc": attrs.get("scene_desc"),
        "objects": attrs.get("objects", []),
        "intrinsics": attrs.get("intrinsics", {}),
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _mecka_to_opencv(keypoints: np.ndarray) -> np.ndarray:
    converted = np.zeros_like(keypoints, dtype=np.float64)
    converted[:, 0] = -keypoints[:, 1]
    converted[:, 1] = -keypoints[:, 2]
    converted[:, 2] = keypoints[:, 0]
    return converted


def _project_keypoints_opencv(
    keypoints_world: np.ndarray,
    *,
    focal_length: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    keypoints_cam = _mecka_to_opencv(
        np.asarray(keypoints_world, dtype=np.float64).reshape(21, 3)
    )
    pixels = np.full((21, 2), np.nan, dtype=np.float64)
    valid = keypoints_cam[:, 2] > 1e-6
    pixels[valid, 0] = (
        keypoints_cam[valid, 0] * focal_length / keypoints_cam[valid, 2] + cx
    )
    pixels[valid, 1] = (
        keypoints_cam[valid, 1] * focal_length / keypoints_cam[valid, 2] + cy
    )
    return pixels


def _draw_projected_hand(
    image_bgr: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
    *,
    vis_line: bool,
) -> None:
    h, w = image_bgr.shape[:2]

    def _in_bounds(point: np.ndarray) -> bool:
        return (
            not np.isnan(point[0])
            and not np.isnan(point[1])
            and 0 <= point[0] < w
            and 0 <= point[1] < h
        )

    if vis_line:
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = pixels[start_idx]
            end = pixels[end_idx]
            if _in_bounds(start) and _in_bounds(end):
                cv2.line(
                    image_bgr,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color,
                    2,
                    cv2.LINE_AA,
                )

    for idx, point in enumerate(pixels):
        if _in_bounds(point):
            radius = 5 if idx else 7
            cv2.circle(
                image_bgr,
                (int(point[0]), int(point[1])),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )


def render_keypoints_with_manual_projection(
    image_rgb: np.ndarray,
    left_world: np.ndarray,
    right_world: np.ndarray,
    *,
    vis_line: bool,
    intrinsics: dict[str, Any] | None = None,
) -> np.ndarray:
    image_bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    intrinsics = intrinsics or {}
    src_w = float(intrinsics.get("w") or w)
    scale = w / src_w if src_w else 1.0
    focal_length = (
        float(intrinsics.get("fl_x") or intrinsics.get("focal_length") or (0.7 * w))
        * scale
    )
    cx = (
        float(intrinsics.get("cx") or (w / 2.0)) * scale
        if "cx" in intrinsics
        else w / 2.0
    )
    cy = (
        float(intrinsics.get("cy") or (h / 2.0)) * scale
        if "cy" in intrinsics
        else h / 2.0
    )
    left_px = _project_keypoints_opencv(
        left_world, focal_length=focal_length, cx=cx, cy=cy
    )
    right_px = _project_keypoints_opencv(
        right_world, focal_length=focal_length, cx=cx, cy=cy
    )
    _draw_projected_hand(image_bgr, left_px, (0, 255, 0), vis_line=vis_line)
    _draw_projected_hand(image_bgr, right_px, (0, 0, 255), vis_line=vis_line)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _transform_keypoints_to_headframe(
    head_pose: np.ndarray,
    left_world: np.ndarray,
    right_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from egomimic.rldb.zarr.action_chunk_transforms import PoseCoordinateFrameTransform

    left_tf = PoseCoordinateFrameTransform(
        target_world="obs_head_pose",
        pose_world="left.obs_keypoints",
        transformed_key_name="left.obs_keypoints_headframe",
        mode="xyz",
    )
    right_tf = PoseCoordinateFrameTransform(
        target_world="obs_head_pose",
        pose_world="right.obs_keypoints",
        transformed_key_name="right.obs_keypoints_headframe",
        mode="xyz",
    )
    batch = {
        "obs_head_pose": np.asarray(head_pose, dtype=np.float64),
        "left.obs_keypoints": np.asarray(left_world, dtype=np.float64),
        "right.obs_keypoints": np.asarray(right_world, dtype=np.float64),
    }
    batch = left_tf.transform(batch)
    batch = right_tf.transform(batch)
    return batch["left.obs_keypoints_headframe"], batch["right.obs_keypoints_headframe"]


def render_keypoints_on_image(
    image_rgb: np.ndarray,
    left_world: np.ndarray,
    right_world: np.ndarray,
    head_pose: np.ndarray,
    *,
    vis_line: bool,
    intrinsics: dict[str, Any] | None = None,
) -> np.ndarray:
    global _WARNED_ARIA_FALLBACK
    try:
        from egomimic.rldb.embodiment.human import Aria

        left_head, right_head = _transform_keypoints_to_headframe(
            head_pose, left_world, right_world
        )
        viz_data = np.concatenate(
            [left_head.reshape(-1), right_head.reshape(-1)], axis=0
        )
        vis = Aria.viz(
            image=np.asarray(image_rgb, dtype=np.uint8),
            viz_data=viz_data,
            mode="keypoints",
            intrinsics_key="base",
            vis_line=vis_line,
        )
        return np.asarray(vis, dtype=np.uint8)
    except ModuleNotFoundError as exc:
        if not _WARNED_ARIA_FALLBACK:
            LOGGER.warning(
                "Falling back to OpenCV projection because Aria dependencies are unavailable: %s",
                exc,
            )
            _WARNED_ARIA_FALLBACK = True
        return render_keypoints_with_manual_projection(
            image_rgb,
            left_world,
            right_world,
            vis_line=vis_line,
            intrinsics=intrinsics,
        )


def visualize_zarr(
    input_zarr: Path,
    output_video: Path,
    *,
    image_key: str,
    max_frames: int | None,
    fps: int | None,
    frame_stride: int,
    vis_line: bool,
) -> Path:
    episode = ProcessedMeckaZarr(input_zarr, image_key=image_key)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    end = (
        episode.total_frames
        if max_frames is None
        else min(max_frames, episode.total_frames)
    )
    writer = imageio.get_writer(
        str(output_video), fps=fps or episode.fps, macro_block_size=1
    )
    written = 0
    try:
        for frame_index in range(0, end, frame_stride):
            frame = render_keypoints_on_image(
                episode.image(frame_index),
                episode.left_keypoints[frame_index].reshape(21, 3),
                episode.right_keypoints[frame_index].reshape(21, 3),
                episode.head_pose[frame_index],
                vis_line=vis_line,
                intrinsics=episode.attrs.get("intrinsics", {}),
            )
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    if written == 0:
        raise ValueError(f"No frames written for {input_zarr}")
    LOGGER.info("Saved overlay video to %s (%s frames)", output_video, written)
    return output_video


def _as_rgb_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def visualize_lerobot(
    lerobot_root: Path,
    repo_id: str,
    output_video: Path,
    *,
    max_frames: int | None,
    fps: int | None,
    frame_stride: int,
    vis_line: bool,
) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=lerobot_root, local_files_only=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    end = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    writer = imageio.get_writer(
        str(output_video), fps=fps or dataset.fps, macro_block_size=1
    )
    written = 0
    try:
        for frame_index in range(0, end, frame_stride):
            sample = dataset[frame_index]
            image = _as_rgb_uint8(sample[LEROBOT_IMAGE_KEY])
            keypoints = _as_numpy(sample["actions_ee_keypoints_world"][0]).reshape(
                2, 21, 3
            )
            head_pose = _as_numpy(sample["observations.state.head_pose"])
            frame = render_keypoints_on_image(
                image,
                keypoints[0],
                keypoints[1],
                head_pose,
                vis_line=vis_line,
            )
            writer.append_data(frame)
            written += 1
    finally:
        writer.close()
    if written == 0:
        raise ValueError(f"No frames written for {lerobot_root}")
    LOGGER.info("Saved overlay video to %s (%s frames)", output_video, written)
    return output_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert processed Mecka Zarr to LeRobot and render hand-keypoint overlay videos."
    )
    parser.add_argument("--input-zarr", type=Path, default=DEFAULT_INPUT_ZARR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="LeRobot output root. If set, conversion runs.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--video-encoding",
        action="store_true",
        help="Store LeRobot image stream as video.",
    )
    parser.add_argument("--image-key", type=str, default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--visualize-out", type=Path, default=None)
    parser.add_argument(
        "--visualize-source", choices=["zarr", "lerobot"], default="zarr"
    )
    parser.add_argument("--lerobot-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--vis-line", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.output_dir is None and args.visualize_out is None:
        raise ValueError(
            "Nothing to do: pass --output-dir for conversion and/or --visualize-out for video output."
        )

    if args.output_dir is not None:
        convert_zarr_to_lerobot(
            args.input_zarr,
            args.output_dir,
            args.repo_id,
            chunk_size=args.chunk_size,
            encode_video=args.video_encoding,
            overwrite=args.overwrite,
            image_key=args.image_key,
        )

    if args.visualize_out is not None:
        if args.visualize_source == "zarr":
            visualize_zarr(
                args.input_zarr,
                args.visualize_out,
                image_key=args.image_key,
                max_frames=args.max_frames,
                fps=args.fps,
                frame_stride=args.frame_stride,
                vis_line=args.vis_line,
            )
        else:
            lerobot_root = args.lerobot_root or args.output_dir
            if lerobot_root is None:
                raise ValueError(
                    "--visualize-source lerobot requires --lerobot-root or --output-dir"
                )
            visualize_lerobot(
                lerobot_root,
                args.repo_id,
                args.visualize_out,
                max_frames=args.max_frames,
                fps=args.fps,
                frame_stride=args.frame_stride,
                vis_line=args.vis_line,
            )


if __name__ == "__main__":
    main()
