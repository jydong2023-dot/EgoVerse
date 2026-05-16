#!/usr/bin/env python3
"""Split an EgoVerse zarr episode into segment-specific zarr episodes.

This script consumes a ``*_segments.json`` file with inclusive
``start_frame`` / ``end_frame`` values and writes one zarr folder per segment.

It intentionally reads/writes the simple zarr v3 layout used by EgoVerse data
directly, so it does not require the ``zarr`` Python package at runtime:

* one outer chunk per array
* ``sharding_indexed`` codec
* numeric arrays encoded as bytes + zstd
* jpeg/image arrays encoded as vlen-bytes + zstd
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs.vlen import VLenBytes
from numcodecs.zstd import Zstd

try:
    from numcodecs import CRC32C
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing CRC32C codec (needed for zarr sharding indexes). "
        "Install it with: pip install google-crc32c"
    ) from exc


DEFAULT_SEGMENTS_JSON = Path(
    "/home/djy/EgoVerse/data/2025-09-20-17-47-54-000000/"
    "2025-09-20-17-47-54-000000_segments.json"
)
DEFAULT_OUTPUT_ROOT = Path("/home/djy/EgoVerse/data/split_data")


@dataclass(frozen=True)
class SegmentRange:
    segment_id: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _array_meta(array_dir: Path) -> dict[str, Any]:
    return load_json(array_dir / "zarr.json")


def _dtype_from_meta(meta: dict[str, Any]) -> np.dtype:
    return np.dtype(meta["data_type"])


def _inner_chunk_shape(meta: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in meta["codecs"][0]["configuration"]["chunk_shape"])


def _grid_shape(
    shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if len(shape) != len(chunk_shape):
        raise ValueError(f"Shape rank {shape} does not match chunk rank {chunk_shape}")
    if not shape:
        return ()
    return tuple(
        (dim + chunk - 1) // chunk
        for dim, chunk in zip(shape, chunk_shape, strict=True)
    )


def _chunk_slices(
    chunk_index: tuple[int, ...], shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> tuple[slice, ...]:
    slices: list[slice] = []
    for idx, dim, chunk in zip(chunk_index, shape, chunk_shape, strict=True):
        start = idx * chunk
        stop = min(start + chunk, dim)
        slices.append(slice(start, stop))
    return tuple(slices)


def _single_shard_path(array_dir: Path, ndim: int) -> Path:
    shard = array_dir / "c"
    for _ in range(ndim):
        shard = shard / "0"
    if shard.is_file():
        return shard

    chunk_root = array_dir / "c"
    if not chunk_root.exists():
        return shard
    candidates = sorted(p for p in chunk_root.rglob("*") if p.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one shard file under {chunk_root}, found {len(candidates)}"
        )
    return candidates[0]


def _decode_shard_index(raw: bytes, num_entries: int) -> np.ndarray:
    if num_entries == 0:
        return np.empty((0, 2), dtype="<u8")
    index_nbytes = num_entries * 2 * 8
    index_payload = CRC32C().decode(raw[-(index_nbytes + 4) :])
    return np.frombuffer(index_payload, dtype="<u8").reshape(num_entries, 2)


def _encode_shard(chunks: list[bytes]) -> bytes:
    body = bytearray()
    entries: list[tuple[int, int]] = []
    for chunk in chunks:
        entries.append((len(body), len(chunk)))
        body.extend(chunk)
    index = np.asarray(entries, dtype="<u8").reshape(len(entries), 2).tobytes()
    body.extend(CRC32C().encode(index))
    return bytes(body)


def _read_numeric_array(array_dir: Path, meta: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(x) for x in meta["shape"])
    if any(dim == 0 for dim in shape):
        return np.empty(shape, dtype=_dtype_from_meta(meta))

    chunk_shape = _inner_chunk_shape(meta)
    grid_shape = _grid_shape(shape, chunk_shape)
    shard_path = _single_shard_path(array_dir, len(shape))
    raw = shard_path.read_bytes()
    entries = _decode_shard_index(raw, int(np.prod(grid_shape)))

    data = np.empty(shape, dtype=_dtype_from_meta(meta))
    zstd = Zstd()
    for flat_idx, chunk_index in enumerate(product(*[range(n) for n in grid_shape])):
        offset, nbytes = entries[flat_idx]
        payload = raw[int(offset) : int(offset + nbytes)]
        decoded = zstd.decode(payload)
        slices = _chunk_slices(tuple(chunk_index), shape, chunk_shape)
        chunk_shape_actual = tuple(s.stop - s.start for s in slices)
        chunk = np.frombuffer(decoded, dtype=data.dtype).reshape(chunk_shape_actual)
        data[slices] = chunk
    return data


def _read_vlen_bytes_array(array_dir: Path, meta: dict[str, Any]) -> list[bytes]:
    shape = tuple(int(x) for x in meta["shape"])
    if len(shape) != 1:
        raise ValueError(
            f"Only 1-D variable_length_bytes arrays are supported, got {shape}"
        )
    if shape[0] == 0:
        return []

    chunk_shape = _inner_chunk_shape(meta)
    grid_shape = _grid_shape(shape, chunk_shape)
    shard_path = _single_shard_path(array_dir, len(shape))
    raw = shard_path.read_bytes()
    entries = _decode_shard_index(raw, int(np.prod(grid_shape)))

    out: list[bytes] = []
    zstd = Zstd()
    vlen = VLenBytes()
    for flat_idx, chunk_index in enumerate(product(*[range(n) for n in grid_shape])):
        offset, nbytes = entries[flat_idx]
        payload = raw[int(offset) : int(offset + nbytes)]
        decoded = zstd.decode(payload)
        values = vlen.decode(decoded)
        out.extend(bytes(v) for v in values)
    return out[: shape[0]]


def read_sharded_array(array_dir: Path) -> np.ndarray | list[bytes]:
    """Read a supported EgoVerse zarr v3 array from disk."""
    meta = _array_meta(array_dir)
    if meta["data_type"] == "variable_length_bytes":
        return _read_vlen_bytes_array(array_dir, meta)
    return _read_numeric_array(array_dir, meta)


def _segment_numeric(data: np.ndarray, segment: SegmentRange) -> np.ndarray:
    return np.asarray(data[segment.start_frame : segment.end_frame + 1])


def _segment_vlen(values: list[bytes], segment: SegmentRange) -> list[bytes]:
    return values[segment.start_frame : segment.end_frame + 1]


def _write_shard(array_dir: Path, meta: dict[str, Any], chunks: list[bytes]) -> None:
    shape = tuple(int(x) for x in meta["shape"])
    if not shape or any(dim == 0 for dim in shape):
        return
    shard_path = _single_shard_path(array_dir, len(shape))
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(_encode_shard(chunks))


def _update_array_meta_for_segment(
    meta: dict[str, Any], shape: tuple[int, ...]
) -> dict[str, Any]:
    out = copy.deepcopy(meta)
    out["shape"] = list(shape)
    out["chunk_grid"]["configuration"]["chunk_shape"] = list(shape)
    inner = list(_inner_chunk_shape(out))
    if shape and inner:
        inner[0] = min(inner[0], shape[0]) if shape[0] > 0 else inner[0]
        out["codecs"][0]["configuration"]["chunk_shape"] = inner
    return out


def _write_numeric_array(
    array_dir: Path, source_meta: dict[str, Any], data: np.ndarray
) -> None:
    out_meta = _update_array_meta_for_segment(
        source_meta, tuple(int(x) for x in data.shape)
    )
    write_json(array_dir / "zarr.json", out_meta)
    if data.size == 0:
        return

    chunk_shape = _inner_chunk_shape(out_meta)
    grid_shape = _grid_shape(tuple(data.shape), chunk_shape)
    zstd = Zstd()
    chunks: list[bytes] = []
    for chunk_index in product(*[range(n) for n in grid_shape]):
        slices = _chunk_slices(tuple(chunk_index), tuple(data.shape), chunk_shape)
        chunk = np.ascontiguousarray(data[slices])
        chunks.append(zstd.encode(chunk.tobytes(order="C")))
    _write_shard(array_dir, out_meta, chunks)


def _write_vlen_bytes_array(
    array_dir: Path, source_meta: dict[str, Any], values: list[bytes]
) -> None:
    out_meta = _update_array_meta_for_segment(source_meta, (len(values),))
    write_json(array_dir / "zarr.json", out_meta)
    if not values:
        return

    chunk_shape = _inner_chunk_shape(out_meta)
    chunk_len = chunk_shape[0]
    zstd = Zstd()
    vlen = VLenBytes()
    chunks = [
        zstd.encode(vlen.encode(values[i : i + chunk_len]))
        for i in range(0, len(values), chunk_len)
    ]
    _write_shard(array_dir, out_meta, chunks)


def _copy_empty_or_non_frame_array(
    src_dir: Path, dst_dir: Path, source_meta: dict[str, Any]
) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    write_json(dst_dir / "zarr.json", source_meta)


def _frame_count_from_arrays(episode_dir: Path) -> int:
    max_first_dim = 0
    for child in episode_dir.iterdir():
        meta_path = child / "zarr.json"
        if not child.is_dir() or not meta_path.exists():
            continue
        meta = load_json(meta_path)
        shape = meta.get("shape") or []
        if shape:
            max_first_dim = max(max_first_dim, int(shape[0]))
    return max_first_dim


def _validate_segment(segment: SegmentRange, source_frames: int) -> None:
    if segment.start_frame < 0 or segment.end_frame < segment.start_frame:
        raise ValueError(f"Invalid segment range: {segment}")
    if segment.end_frame >= source_frames:
        raise ValueError(
            f"Segment {segment.segment_id} out of bounds: end_frame={segment.end_frame}, "
            f"source_frames={source_frames}"
        )


def _segment_from_dict(raw: dict[str, Any], fallback_id: int) -> SegmentRange:
    return SegmentRange(
        segment_id=int(raw.get("segment_id", fallback_id)),
        start_frame=int(raw["start_frame"]),
        end_frame=int(raw["end_frame"]),
    )


def _split_array(
    src_dir: Path, dst_dir: Path, segment: SegmentRange, source_frames: int
) -> None:
    meta = _array_meta(src_dir)
    shape = tuple(int(x) for x in meta["shape"])
    if not shape or shape[0] == 0:
        _copy_empty_or_non_frame_array(src_dir, dst_dir, meta)
        return
    if shape[0] != source_frames:
        # Not aligned to the episode timeline; preserve as-is instead of slicing the wrong axis.
        _copy_empty_or_non_frame_array(src_dir, dst_dir, meta)
        return
    if meta["data_type"] == "variable_length_bytes":
        values = _segment_vlen(_read_vlen_bytes_array(src_dir, meta), segment)
        _write_vlen_bytes_array(dst_dir, meta, values)
    else:
        data = _segment_numeric(_read_numeric_array(src_dir, meta), segment)
        _write_numeric_array(dst_dir, meta, data)


def _output_episode_dir(
    output_root: Path, episode_dir: Path, segment: SegmentRange
) -> Path:
    return (
        output_root
        / f"{episode_dir.name}_seg{segment.segment_id:03d}_{segment.start_frame:06d}_{segment.end_frame:06d}"
    )


def _write_group_meta(
    src_group_meta: dict[str, Any],
    dst_dir: Path,
    segment: SegmentRange,
    src_episode: Path,
) -> None:
    out_meta = copy.deepcopy(src_group_meta)
    attrs = out_meta.setdefault("attributes", {})
    attrs["total_frames"] = segment.frame_count
    attrs["source_episode_path"] = str(src_episode)
    attrs["source_segment"] = {
        "segment_id": segment.segment_id,
        "start_frame": segment.start_frame,
        "end_frame": segment.end_frame,
        "frame_count": segment.frame_count,
    }
    write_json(dst_dir / "zarr.json", out_meta)


def split_episode_from_segments(
    segments_json: Path, output_root: Path, overwrite: bool = False
) -> list[Path]:
    segments_json = segments_json.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    segments_payload = load_json(segments_json)

    episode_dir = (
        Path(segments_payload.get("episode_path") or segments_json.parent)
        .expanduser()
        .resolve()
    )
    if not (episode_dir / "zarr.json").is_file():
        raise FileNotFoundError(f"Episode zarr root not found: {episode_dir}")

    group_meta = load_json(episode_dir / "zarr.json")
    source_frames = max(
        int(segments_payload.get("total_frames") or 0),
        int(group_meta.get("attributes", {}).get("total_frames") or 0),
        _frame_count_from_arrays(episode_dir),
    )

    raw_segments = segments_payload.get("segments") or []
    if not raw_segments:
        raise ValueError(f"No segments found in {segments_json}")

    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for idx, raw_segment in enumerate(raw_segments):
        segment = _segment_from_dict(raw_segment, idx)
        _validate_segment(segment, source_frames)

        dst_dir = _output_episode_dir(output_root, episode_dir, segment)
        if dst_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output already exists: {dst_dir} (use --overwrite)"
                )
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True)

        _write_group_meta(group_meta, dst_dir, segment, episode_dir)
        write_json(dst_dir / "source_segment.json", raw_segment)

        for child in sorted(episode_dir.iterdir()):
            if not child.is_dir() or not (child / "zarr.json").is_file():
                continue
            _split_array(child, dst_dir / child.name, segment, source_frames)

        outputs.append(dst_dir)
        print(f"wrote {dst_dir} ({segment.frame_count} frames)")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split an EgoVerse zarr episode according to start_frame/end_frame segments."
    )
    parser.add_argument(
        "--segments-json",
        type=Path,
        default=DEFAULT_SEGMENTS_JSON,
        help="Path to *_segments.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where split zarr episode folders are written",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing split output folders"
    )
    args = parser.parse_args()

    outputs = split_episode_from_segments(
        args.segments_json, args.output_root, overwrite=args.overwrite
    )
    print(
        f"Done. Wrote {len(outputs)} split episode(s) to {args.output_root.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
