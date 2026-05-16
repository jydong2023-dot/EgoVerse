#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from numcodecs import CRC32C, Zstd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE = REPO_ROOT / "data" / "2025-09-20-17-47-54-000000"
DEFAULT_MAPPING_JSON = REPO_ROOT / "egomimic" / "scripts" / "egoverse_to_openego.json"
DEFAULT_OUTPUT_EPISODE = REPO_ROOT / "data" / "2025-09-20-17-47-54-000000_mapped"
DEFAULT_VIDEO_OUT = REPO_ROOT / "data_mapping.mp4"
VIS_SCRIPT = REPO_ROOT / "egomimic" / "scripts" / "visualize_episode_keypoints.py"
KEYPOINT_DIRS = ("left.obs_keypoints", "right.obs_keypoints")


def load_mapping(mapping_json: Path) -> list[int]:
    with mapping_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    mapping = payload["egoverse_to_openego"]
    original_index = mapping["original_index"]
    new_index = mapping["new_index"]
    if len(original_index) != len(new_index):
        raise ValueError("original_index and new_index must have the same length")

    output_size = len(new_index)
    reordered = [None] * output_size
    for src_idx, dst_idx in zip(original_index, new_index, strict=True):
        if dst_idx < 0 or dst_idx >= output_size:
            raise ValueError(f"new_index out of range: {dst_idx}")
        reordered[dst_idx] = src_idx

    if any(item is None for item in reordered):
        raise ValueError(f"Incomplete mapping in {mapping_json}: {reordered}")

    return [int(item) for item in reordered]


def transform_keypoints(
    data: np.ndarray, reordered_original_indices: list[int]
) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] != 63:
        raise ValueError(f"Expected keypoints shape (T, 63), got {tuple(data.shape)}")

    keypoints = data.reshape(data.shape[0], 21, 3)
    extra_keypoint = (keypoints[:, 5, :] + keypoints[:, 6, :]) / 2.0
    augmented = np.concatenate(
        [keypoints, extra_keypoint[:, None, :]], axis=1
    )  # (T, 22, 3)

    if max(reordered_original_indices) >= augmented.shape[1]:
        raise ValueError(
            "Mapping references an original index outside the augmented keypoint range: "
            f"max={max(reordered_original_indices)}, available={augmented.shape[1]}"
        )

    reordered = augmented[:, reordered_original_indices, :]
    return reordered.reshape(data.shape[0], 63).astype(data.dtype, copy=False)


def load_array_meta(array_dir: Path) -> dict:
    with (array_dir / "zarr.json").open("r", encoding="utf-8") as f:
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


def decode_sharded_array(array_dir: Path) -> np.ndarray:
    """Read numeric Zarr v3 sharding_indexed arrays (Mecka-style: logical shape can be smaller than outer chunk)."""
    meta = load_array_meta(array_dir)
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


def encode_sharded_array(array_dir: Path, data: np.ndarray) -> None:
    """Write Mecka-style sharding_indexed bytes: inner tiles cover each outer chunk; unused slots are sentinels."""
    meta = load_array_meta(array_dir)
    shape = tuple(int(v) for v in meta["shape"])
    outer_shape = tuple(
        int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]
    )
    inner_shape = tuple(
        int(v) for v in meta["codecs"][0]["configuration"]["chunk_shape"]
    )
    dtype = np.dtype(meta["data_type"]).newbyteorder("<")

    if tuple(data.shape) != tuple(shape):
        raise ValueError(
            f"Data shape mismatch: expected {shape}, got {tuple(data.shape)}"
        )

    if len(shape) != 2 or len(inner_shape) != 2:
        raise ValueError(
            f"Expected 2D array and 2D shard chunks, got {shape} and {inner_shape}"
        )

    outer_rows = _ceil_div(shape[0], outer_shape[0])
    outer_cols = _ceil_div(shape[1], outer_shape[1])
    max_u64 = np.iinfo(np.uint64).max
    zstd = Zstd()

    for outer_row in range(outer_rows):
        for outer_col in range(outer_cols):
            chunk_path = array_dir / "c" / str(outer_row) / str(outer_col)
            outer_row_start = outer_row * outer_shape[0]
            outer_col_start = outer_col * outer_shape[1]

            shard_rows = _ceil_div(outer_shape[0], inner_shape[0])
            shard_cols = _ceil_div(outer_shape[1], inner_shape[1])

            payloads: list[bytes] = []
            entries: list[tuple[int, int]] = []
            offset = 0

            for sr in range(shard_rows):
                for sc in range(shard_cols):
                    inner_r0 = sr * inner_shape[0]
                    inner_r1 = inner_r0 + inner_shape[0]
                    inner_c0 = sc * inner_shape[1]
                    inner_c1 = inner_c0 + inner_shape[1]

                    g_r0 = outer_row_start + inner_r0
                    g_r1 = outer_row_start + inner_r1
                    g_c0 = outer_col_start + inner_c0
                    g_c1 = outer_col_start + inner_c1

                    ir0 = max(0, g_r0)
                    ir1 = min(shape[0], g_r1)
                    ic0 = max(0, g_c0)
                    ic1 = min(shape[1], g_c1)

                    if ir0 >= ir1 or ic0 >= ic1:
                        entries.append((int(max_u64), int(max_u64)))
                        continue

                    buf = np.zeros(inner_shape, dtype=dtype)
                    loc_r0 = ir0 - g_r0
                    loc_r1 = loc_r0 + (ir1 - ir0)
                    loc_c0 = ic0 - g_c0
                    loc_c1 = loc_c0 + (ic1 - ic0)
                    buf[loc_r0:loc_r1, loc_c0:loc_c1] = np.asarray(
                        data[ir0:ir1, ic0:ic1], dtype=dtype
                    )

                    encoded = zstd.encode(np.ascontiguousarray(buf).tobytes(order="C"))
                    payloads.append(encoded)
                    entries.append((offset, len(encoded)))
                    offset += len(encoded)

            entries_arr = np.asarray(entries, dtype="<u8")
            index_payload = CRC32C().encode(entries_arr.tobytes(order="C"))
            index_bytes = (
                index_payload.tobytes()
                if isinstance(index_payload, np.ndarray)
                else bytes(index_payload)
            )
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.write_bytes(b"".join(payloads) + index_bytes)


def copy_episode_with_symlinks(
    src_episode: Path, dst_episode: Path, overwrite: bool
) -> None:
    if dst_episode.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output episode already exists: {dst_episode}. Use --overwrite to replace it."
            )
        if dst_episode.is_symlink() or dst_episode.is_file():
            dst_episode.unlink()
        else:
            shutil.rmtree(dst_episode)

    dst_episode.mkdir(parents=True, exist_ok=False)

    for child in src_episode.iterdir():
        dst_child = dst_episode / child.name
        if child.name in KEYPOINT_DIRS:
            shutil.copytree(child, dst_child)
        else:
            dst_child.symlink_to(child, target_is_directory=child.is_dir())


def write_transformed_keypoints(
    src_episode: Path,
    dst_episode: Path,
    reordered_original_indices: list[int],
) -> None:
    for key_dir in KEYPOINT_DIRS:
        src_data = decode_sharded_array(src_episode / key_dir).astype(
            np.float64, copy=False
        )
        transformed = transform_keypoints(src_data, reordered_original_indices)
        if transformed.shape != src_data.shape:
            raise ValueError(
                f"Transformed shape mismatch for {key_dir}: {transformed.shape} vs {src_data.shape}"
            )
        encode_sharded_array(dst_episode / key_dir, transformed)


def run_visualization(
    episode_dir: Path,
    video_out: Path,
    max_frames: int | None,
    fps: int,
    vis_index: bool,
) -> None:
    cmd = [
        sys.executable,
        str(VIS_SCRIPT),
        "--episode-dir",
        str(episode_dir),
        "--out",
        str(video_out),
        "--fps",
        str(fps),
    ]
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if vis_index:
        cmd.append("--vis-index")

    print("Running visualization command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map EgoVerse hand keypoints to OpenEgo order and render a visualization video."
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--output-episode", type=Path, default=DEFAULT_OUTPUT_EPISODE)
    parser.add_argument("--video-out", type=Path, default=DEFAULT_VIDEO_OUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--vis-index", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.episode_dir.exists():
        raise FileNotFoundError(f"Episode directory does not exist: {args.episode_dir}")
    if not args.mapping_json.exists():
        raise FileNotFoundError(f"Mapping json does not exist: {args.mapping_json}")
    if not VIS_SCRIPT.exists():
        raise FileNotFoundError(f"Visualization script does not exist: {VIS_SCRIPT}")

    reordered_original_indices = load_mapping(args.mapping_json)
    print(f"Loaded mapping from: {args.mapping_json}", flush=True)
    print(f"Reordered original indices: {reordered_original_indices}", flush=True)

    print(f"Creating mapped episode at: {args.output_episode}", flush=True)
    copy_episode_with_symlinks(
        src_episode=args.episode_dir,
        dst_episode=args.output_episode,
        overwrite=args.overwrite,
    )
    print("Episode skeleton created.", flush=True)

    print("Writing transformed left/right keypoints...", flush=True)
    write_transformed_keypoints(
        src_episode=args.episode_dir,
        dst_episode=args.output_episode,
        reordered_original_indices=reordered_original_indices,
    )
    print("Keypoint arrays written.", flush=True)

    print(f"Rendering visualization to: {args.video_out}", flush=True)
    run_visualization(
        episode_dir=args.output_episode,
        video_out=args.video_out,
        max_frames=args.max_frames,
        fps=args.fps,
        vis_index=args.vis_index,
    )

    print(f"Mapped episode written to: {args.output_episode}")
    print(f"Visualization saved to: {args.video_out}")


if __name__ == "__main__":
    main()
