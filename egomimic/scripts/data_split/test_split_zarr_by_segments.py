from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numcodecs import CRC32C
from numcodecs.vlen import VLenBytes
from numcodecs.zstd import Zstd

from egomimic.scripts.data_split.split_zarr_by_segments import (
    read_sharded_array,
    split_episode_from_segments,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _shard_entries(chunks: list[bytes]) -> bytes:
    body = bytearray()
    entries = []
    for chunk in chunks:
        entries.append((len(body), len(chunk)))
        body.extend(chunk)
    index = np.asarray(entries, dtype="<u8").reshape(len(entries), 2).tobytes()
    body.extend(CRC32C().encode(index))
    return bytes(body)


def _write_numeric_array(
    root: Path, key: str, data: np.ndarray, inner_first_dim: int = 2
) -> None:
    arr_dir = root / key
    arr_dir.mkdir(parents=True, exist_ok=True)
    shape = list(data.shape)
    chunk_shape = [inner_first_dim, *shape[1:]]
    meta = {
        "shape": shape,
        "data_type": str(data.dtype),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shape}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk_shape,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {
                            "name": "zstd",
                            "configuration": {"level": 0, "checksum": False},
                        },
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    _write_json(arr_dir / "zarr.json", meta)
    chunks = [
        Zstd().encode(data[i : i + inner_first_dim].tobytes(order="C"))
        for i in range(0, shape[0], inner_first_dim)
    ]
    shard_path = arr_dir / "c" / "/".join(["0"] * len(shape))
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(_shard_entries(chunks))


def _write_bytes_array(root: Path, key: str, frames: list[bytes]) -> None:
    arr_dir = root / key
    arr_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "shape": [len(frames)],
        "data_type": "variable_length_bytes",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [len(frames)]},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": "",
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [1],
                    "codecs": [
                        {"name": "vlen-bytes", "configuration": {}},
                        {
                            "name": "zstd",
                            "configuration": {"level": 0, "checksum": False},
                        },
                    ],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array",
        "storage_transformers": [],
    }
    _write_json(arr_dir / "zarr.json", meta)
    vlen = VLenBytes()
    chunks = [Zstd().encode(vlen.encode([frame])) for frame in frames]
    shard_path = arr_dir / "c" / "0"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(_shard_entries(chunks))


def test_split_episode_from_segments_keeps_only_requested_frame_range(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    _write_json(
        episode / "zarr.json",
        {
            "attributes": {
                "total_frames": 6,
                "fps": 30,
                "features": {
                    "signal": {"dtype": "int64", "shape": [], "names": []},
                    "vec": {"dtype": "int64", "shape": [2], "names": ["dim_0"]},
                    "images.front_1": {
                        "dtype": "jpeg",
                        "shape": [1, 1, 3],
                        "names": ["height", "width", "channel"],
                    },
                },
            },
            "zarr_format": 3,
            "node_type": "group",
        },
    )
    _write_numeric_array(episode, "signal", np.arange(6, dtype=np.int64))
    _write_numeric_array(episode, "vec", np.arange(12, dtype=np.int64).reshape(6, 2))
    _write_bytes_array(
        episode, "images.front_1", [f"img{i}".encode() for i in range(6)]
    )

    segments_json = tmp_path / "segments.json"
    _write_json(
        segments_json,
        {
            "episode_path": str(episode),
            "total_frames": 6,
            "fps": 30,
            "num_segments": 1,
            "segments": [{"segment_id": 7, "start_frame": 2, "end_frame": 4}],
        },
    )

    out_root = tmp_path / "split"
    outputs = split_episode_from_segments(segments_json, out_root, overwrite=True)

    assert len(outputs) == 1
    split_dir = outputs[0]
    split_meta = json.loads((split_dir / "zarr.json").read_text(encoding="utf-8"))
    assert split_meta["attributes"]["total_frames"] == 3
    assert split_meta["attributes"]["source_segment"]["start_frame"] == 2
    assert json.loads((split_dir / "signal" / "zarr.json").read_text(encoding="utf-8"))[
        "shape"
    ] == [3]
    assert json.loads((split_dir / "vec" / "zarr.json").read_text(encoding="utf-8"))[
        "shape"
    ] == [3, 2]
    assert json.loads(
        (split_dir / "images.front_1" / "zarr.json").read_text(encoding="utf-8")
    )["shape"] == [3]

    assert np.array_equal(
        read_sharded_array(split_dir / "signal"), np.array([2, 3, 4], dtype=np.int64)
    )
    assert np.array_equal(
        read_sharded_array(split_dir / "vec"),
        np.array([[4, 5], [6, 7], [8, 9]], dtype=np.int64),
    )
    assert read_sharded_array(split_dir / "images.front_1") == [
        b"img2",
        b"img3",
        b"img4",
    ]


def test_split_episode_rejects_out_of_bounds_segment(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    _write_json(
        episode / "zarr.json",
        {
            "attributes": {"total_frames": 2, "features": {}},
            "zarr_format": 3,
            "node_type": "group",
        },
    )
    segments_json = tmp_path / "segments.json"
    _write_json(
        segments_json,
        {
            "episode_path": str(episode),
            "segments": [{"segment_id": 0, "start_frame": 1, "end_frame": 2}],
        },
    )

    with pytest.raises(ValueError, match="out of bounds"):
        split_episode_from_segments(segments_json, tmp_path / "split", overwrite=True)
