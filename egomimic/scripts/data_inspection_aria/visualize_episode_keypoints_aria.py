#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import imageio_ffmpeg
import mediapy as mpy
import numpy as np
from numcodecs import CRC32C, VLenBytes, Zstd

# Allow direct execution without setting PYTHONPATH first.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.rldb.embodiment.human import Aria  # noqa: E402
from egomimic.rldb.zarr.action_chunk_transforms import (  # noqa: E402
    PoseCoordinateFrameTransform,
)
from egomimic.utils.egomimicUtils import (  # noqa: E402
    INTRINSICS,
    cam_frame_to_cam_pixels,
)
from egomimic.utils.pose_utils import _split_keypoints  # noqa: E402

mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


DEFAULT_EPISODE = Path(
    "/home/djy/EgoVerse/data/scale/flagship_scoop_granular/2026-05-02-19-09-32-836802"
)
DEFAULT_OUTPUT = Path("/home/djy/EgoVerse/data/keypoint_inspection/keypoints_traj.mp4")


def load_json(path: Path) -> dict:
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
    """Read numeric Zarr v3 sharding_indexed arrays (Mecka: logical shape may be smaller than outer chunk)."""
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
    if len(inner_shape) != 2:
        raise ValueError(f"Expected 2D inner shard chunks, got {inner_shape}")

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
    """Read JPEG bytes from Zarr v3 sharding_indexed variable-length image arrays (Mecka layout)."""

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


def extract_first_step_keypoints(
    actions_keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(actions_keypoints)
    while data.ndim > 1:
        data = data[0]

    if data.shape[-1] == 140:
        _, _, left_keypoints, _, _, right_keypoints = _split_keypoints(
            data, wrist_in_data=True, is_quat=True
        )
    elif data.shape[-1] == 138:
        _, _, left_keypoints, _, _, right_keypoints = _split_keypoints(
            data, wrist_in_data=True, is_quat=False
        )
    else:
        left_keypoints, right_keypoints = _split_keypoints(data, wrist_in_data=False)

    return left_keypoints.reshape(21, 3), right_keypoints.reshape(21, 3)


def annotate_keypoint_indices(
    image: np.ndarray,
    actions_keypoints: np.ndarray,
    intrinsics_key: str = "base",
) -> np.ndarray:
    vis = np.asarray(image, dtype=np.uint8).copy()
    intrinsics = INTRINSICS[intrinsics_key]
    h, w = vis.shape[:2]

    left_keypoints, right_keypoints = extract_first_step_keypoints(actions_keypoints)
    hands = {"L": left_keypoints, "R": right_keypoints}

    for hand_label, keypoints in hands.items():
        keypoints_px = cam_frame_to_cam_pixels(keypoints, intrinsics)
        valid = keypoints[:, 2] > 0.01
        valid &= (keypoints_px[:, 0] >= 0) & (keypoints_px[:, 0] < w)
        valid &= (keypoints_px[:, 1] >= 0) & (keypoints_px[:, 1] < h)

        for idx in range(21):
            if not valid[idx]:
                continue
            x = int(keypoints_px[idx, 0]) + 5
            y = int(keypoints_px[idx, 1]) - 5
            label = f"{hand_label}{idx}"
            cv2.putText(
                vis,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return vis


def transform_keypoints_to_headframe(
    head_pose: np.ndarray,
    left_world: np.ndarray,
    right_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


class ManualEpisode:
    def __init__(self, episode_path: Path):
        self.episode_path = episode_path
        meta = load_json(episode_path / "zarr.json")
        attrs = meta["attributes"]

        self.metadata_total_frames = int(attrs["total_frames"])
        self.fps = int(attrs.get("fps", 30))
        self.left_keypoints = decode_numeric_sharded_array(
            episode_path / "left.obs_keypoints"
        )
        self.right_keypoints = decode_numeric_sharded_array(
            episode_path / "right.obs_keypoints"
        )
        self.head_pose = decode_numeric_sharded_array(episode_path / "obs_head_pose")
        self.image_reader = ImageShardReader(episode_path / "images.front_1")

        self.total_frames = min(
            self.metadata_total_frames,
            self.left_keypoints.shape[0],
            self.right_keypoints.shape[0],
            self.head_pose.shape[0],
            self.image_reader.length,
        )

    def __len__(self) -> int:
        return self.total_frames

    def get_frame_visualization(
        self,
        frame_index: int,
        vis_index: bool,
        vis_line: bool,
    ) -> np.ndarray:
        image = decode_jpeg_rgb(self.image_reader.get(frame_index))
        left_world = self.left_keypoints[frame_index].reshape(21, 3)
        right_world = self.right_keypoints[frame_index].reshape(21, 3)
        head_pose = self.head_pose[frame_index]

        left_head, right_head = transform_keypoints_to_headframe(
            head_pose=head_pose,
            left_world=left_world,
            right_world=right_world,
        )
        viz_data = np.concatenate(
            [left_head.reshape(-1), right_head.reshape(-1)], axis=0
        )
        vis = Aria.viz(
            image=image,
            viz_data=viz_data,
            mode="keypoints",
            intrinsics_key="base",
            vis_line=vis_line,
        )
        if vis_index:
            vis = annotate_keypoint_indices(vis, viz_data, intrinsics_key="base")
        return np.asarray(vis, dtype=np.uint8)


def render_video(
    episode_path: Path,
    output_path: Path,
    max_frames: int | None,
    fps: int,
    vis_index: bool,
    vis_line: bool,
) -> None:
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}")

    episode = ManualEpisode(episode_path)
    limit = len(episode) if max_frames is None else min(max_frames, len(episode))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps)
    frames_written = 0
    try:
        for frame_idx in range(limit):
            vis = episode.get_frame_visualization(
                frame_idx,
                vis_index=vis_index,
                vis_line=vis_line,
            )
            writer.append_data(vis)
            frames_written += 1
    finally:
        writer.close()

    if frames_written == 0:
        raise ValueError(f"No visualization frames were produced for {episode_path}")

    print(f"Saved keypoint visualization to: {output_path}")
    print(f"Frames written: {frames_written}")


def render_frame(
    episode_path: Path,
    output_path: Path,
    frame_index: int,
    vis_index: bool,
    vis_line: bool,
) -> None:
    episode = ManualEpisode(episode_path)
    if frame_index < 0 or frame_index >= len(episode):
        raise IndexError(
            f"frame_index={frame_index} out of range for dataset of length {len(episode)}"
        )

    vis = episode.get_frame_visualization(
        frame_index,
        vis_index=vis_index,
        vis_line=vis_line,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), vis)
    print(f"Saved keypoint frame visualization to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize EgoVerse Aria hand keypoints over the episode RGB frames."
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=DEFAULT_EPISODE,
        help="Path to a local EgoVerse episode directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path. Use .mp4 for video mode or .png/.jpg for frame mode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="If set, render only the first N frames in video mode.",
    )
    parser.add_argument(
        "--vis-index",
        action="store_true",
        help="If set, overlay keypoint indices on the visualization.",
    )
    parser.add_argument(
        "--vis-line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to draw skeleton lines between keypoints. Use --no-vis-line to disable.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="If set, render only one frame and save an image instead of a video.",
    )
    args = parser.parse_args()

    episode_path = args.episode_dir
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_path}")

    if args.frame_index is not None:
        render_frame(
            episode_path=episode_path,
            output_path=args.out,
            frame_index=args.frame_index,
            vis_index=args.vis_index,
            vis_line=args.vis_line,
        )
        return

    render_video(
        episode_path=episode_path,
        output_path=args.out,
        max_frames=args.max_frames,
        fps=args.fps,
        vis_index=args.vis_index,
        vis_line=args.vis_line,
    )


if __name__ == "__main__":
    main()
