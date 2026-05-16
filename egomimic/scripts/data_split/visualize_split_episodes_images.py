#!/usr/bin/env python3
"""
Encode JPEG image streams from split EgoVerse episodes into MP4 files.

Inspired by tutorials/zarr_data_viz_local.py (imageio writer + FFmpeg path via imageio_ffmpeg),
but reads raw JPEG bytes from ``images.front_1`` style zarr arrays without ZarrDataset / Torch.

Expected layout per episode folder (same as ``split_data`` output):
    <episode>/zarr.json
    <episode>/images.<name>/zarr.json + c/...

Each episode may contain multiple ``images.*`` keys (dtype jpeg in group attributes).
By default MP4 files are named after the episode folder (``{episode_folder}.mp4``) and saved under
``/home/djy/EgoVerse/data/videos`` (override with ``--output-dir``). Multiple camera streams become
``{episode_folder}__{images_key}.mp4`` with dots in the key replaced by underscores.

Other layouts: encode next to each zarray dir or under the episode root (see ``--output-style``).

Default episode scan roots are unchanged unless you change ``--split-root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import imageio_ffmpeg
import numpy as np
from numcodecs.vlen import VLenBytes
from numcodecs.zstd import Zstd

try:
    from numcodecs import CRC32C
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing CRC32C codec (needed for sharded zarr index). "
        "Install: pip install google-crc32c"
    ) from exc

# Match zarr_data_viz_local.py: repo root on sys.path when needed downstream
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_sharded_entries(raw: bytes, num_subchunks: int) -> np.ndarray:
    index_nbytes = num_subchunks * 2 * 8
    index_payload = CRC32C().decode(raw[-(index_nbytes + 4) :])
    return np.frombuffer(index_payload, dtype="<u8").reshape(num_subchunks, 2)


def _shard_path_for_outer_1d(array_dir: Path, outer_idx: int) -> Path:
    return array_dir / "c" / str(outer_idx)


def _infer_outer_shard_paths(array_dir: Path, num_outer: int) -> list[Path]:
    """Resolve ``c/<i>`` files for i in 0..num_outer-1 (1-D storage layout)."""
    paths = [_shard_path_for_outer_1d(array_dir, i) for i in range(num_outer)]
    missing = [p for p in paths if not p.is_file()]
    if not missing:
        return paths

    cand = sorted(p for p in (array_dir / "c").rglob("*") if p.is_file())
    if len(cand) != num_outer:
        raise FileNotFoundError(
            "Missing shard file(s): "
            f"expected {num_outer} under {array_dir / 'c'}, "
            f"direct keys missing {missing!r}; found {len(cand)} candidate file(s) via rglob"
        )

    rel_base = array_dir / "c"

    def rel_key(p: Path) -> tuple:
        parts = p.relative_to(rel_base).parts
        return tuple(
            int(component) if component.isdigit() else component for component in parts
        )

    return sorted(cand, key=rel_key)


class ImageShardReader:
    """JPEG ``variable_length_bytes`` with ``sharding_indexed``, inner chunk one frame.

    Supports storage where ``chunk_grid.chunk_shape`` (outer slab size along time) differs from
    the logical ``shape[0]`` (e.g. mecka: 428 frames in one slab while outer chunk grid is 1000),
    and where multiple slabs ``c/0``, ``c/1``, ... exist for longer recordings.
    """

    def __init__(self, array_dir: Path):
        meta = load_json(array_dir / "zarr.json")
        self.meta = meta
        shape_tuple = tuple(int(x) for x in meta["shape"])
        if len(shape_tuple) != 1:
            raise ValueError(f"Expected 1-D image array, shape={shape_tuple}")
        self.length = shape_tuple[0]

        outer_cfg = tuple(
            int(x) for x in meta["chunk_grid"]["configuration"]["chunk_shape"]
        )
        if len(outer_cfg) != 1:
            raise ValueError(f"Expected 1-D chunk_grid chunk_shape, got {outer_cfg}")
        self.outer_chunk = outer_cfg[0]
        if self.outer_chunk <= 0:
            raise ValueError(f"Invalid outer chunk size {self.outer_chunk}")

        inner_chunk_shape = tuple(meta["codecs"][0]["configuration"]["chunk_shape"])
        if inner_chunk_shape != (1,):
            raise ValueError(
                f"Expected per-frame inner chunk_shape=(1,), got {inner_chunk_shape}"
            )

        self._outer_count = (
            0
            if self.length == 0
            else (self.length + self.outer_chunk - 1) // self.outer_chunk
        )
        self._shard_paths = _infer_outer_shard_paths(array_dir, self._outer_count)
        self._shard_cache: dict[int, tuple[bytes, np.ndarray]] = {}
        self._zstd = Zstd()
        self._vlen = VLenBytes()

    def _inner_count_for_outer(self, outer_idx: int) -> int:
        start = outer_idx * self.outer_chunk
        return max(0, min(self.outer_chunk, self.length - start))

    def _index_rows_for_shard(self, outer_idx: int) -> int:
        """Index table always lists ``outer_chunk`` rows per outer slab (unused rows are sentinel)."""
        del outer_idx  # uniform grid in current layout
        return self.outer_chunk

    def _load_outer(self, outer_idx: int) -> tuple[bytes, np.ndarray]:
        cached = self._shard_cache.get(outer_idx)
        if cached is not None:
            return cached

        n_rows = self._index_rows_for_shard(outer_idx)
        if n_rows <= 0:
            raise IndexError(f"outer_idx {outer_idx} has no slab")

        shard_path = self._shard_paths[outer_idx]
        raw = shard_path.read_bytes()
        entries = _decode_sharded_entries(raw, n_rows)
        self._shard_cache[outer_idx] = (raw, entries)
        return raw, entries

    def get(self, index: int) -> bytes:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        outer_idx = index // self.outer_chunk
        inner_idx = index - outer_idx * self.outer_chunk
        usable = self._inner_count_for_outer(outer_idx)
        if inner_idx >= usable:
            raise IndexError(
                f"frame {index} maps outside usable slab ({usable} frames in outer {outer_idx})"
            )

        raw, entries = self._load_outer(outer_idx)
        offset, nbytes = entries[inner_idx]
        payload = raw[int(offset) : int(offset + nbytes)]
        decoded = self._zstd.decode(payload)
        return self._vlen.decode(decoded)[0]


def decode_jpeg_rgb(jpeg_payload: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_payload, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG payload")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def list_episode_roots(split_root: Path) -> list[Path]:
    return sorted(
        p for p in split_root.iterdir() if p.is_dir() and (p / "zarr.json").is_file()
    )


def resolve_episodes(split_root: Path) -> list[Path]:
    """Either a bundle directory (many episode subdirs) or a single episode root."""
    if (split_root / "zarr.json").is_file():
        return [split_root.resolve()]
    return list_episode_roots(split_root)


def list_image_keys(features: dict) -> list[str]:
    out: list[str] = []
    for key, spec in features.items():
        if not key.startswith("images."):
            continue
        if spec.get("dtype") == "jpeg":
            out.append(key)
    return sorted(out)


def _mp4_filename_for_episode(
    episode_dir: Path, image_key: str, num_streams: int
) -> str:
    """One stream -> <folder>.mp4; multiple streams -> <folder>__<key_with_underscores>.mp4."""
    base = episode_dir.name
    if num_streams == 1:
        return f"{base}.mp4"
    safe_key = image_key.replace(".", "_")
    return f"{base}__{safe_key}.mp4"


def encode_episode_images(
    episode_dir: Path,
    fps: float,
    image_keys: list[str] | None,
    output_style: str,
    output_dir: Path | None,
    inner_mp4_name: str,
    max_frames: int | None,
    overwrite: bool,
) -> None:
    group_meta = load_json(episode_dir / "zarr.json")
    feats = group_meta.get("attributes", {}).get("features", {})
    task_fps = float(group_meta.get("attributes", {}).get("fps") or fps)

    keys = (
        list_image_keys(feats)
        if not image_keys
        else [k for k in image_keys if k in feats]
    )
    if not keys:
        raise ValueError(f"No jpeg image keys under {episode_dir}")

    if output_style == "videos_flat":
        if output_dir is None:
            raise ValueError("output_dir is required when output_style=videos_flat")
        output_dir.mkdir(parents=True, exist_ok=True)

    num_streams = len(keys)

    for key in keys:
        img_dir = episode_dir / key
        if not (img_dir / "zarr.json").exists():
            print(f"[skip] missing {img_dir}")
            continue

        reader = ImageShardReader(img_dir)
        n = reader.length if max_frames is None else min(reader.length, max_frames)

        if output_style == "videos_flat":
            out_path = output_dir / _mp4_filename_for_episode(
                episode_dir, key, num_streams
            )
        elif output_style == "inside_image_folder":
            out_path = img_dir / inner_mp4_name
        elif output_style == "sibling_under_episode":
            out_path = episode_dir / f"{key}.mp4"
        else:
            raise ValueError(f"Unknown output_style {output_style!r}")

        if out_path.exists() and not overwrite:
            print(f"[skip] exists {out_path}")
            continue

        writer = imageio.get_writer(
            str(out_path), fps=task_fps, codec="libx264", ffmpeg_log_level="error"
        )
        try:
            for i in range(n):
                jpeg = reader.get(i)
                if isinstance(jpeg, memoryview):
                    jpeg = jpeg.tobytes()
                frame = decode_jpeg_rgb(bytes(jpeg))
                writer.append_data(frame)
        finally:
            writer.close()

        print(f"wrote {out_path} ({n} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode split episode JPEG zarr arrays to MP4."
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("/home/djy/EgoVerse/data/mecka/adjusting/"),
        help="Directory containing split episode folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/djy/EgoVerse/videos/mecka/adjusting/"),
        help="Destination directory when --output-style videos_flat (created if missing)",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Fallback FPS if episode has none"
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=None,
        help="Specific keys like images.front_1 (default: all dtype=jpeg images.*)",
    )
    parser.add_argument(
        "--output-style",
        choices=("videos_flat", "inside_image_folder", "sibling_under_episode"),
        default="videos_flat",
        help="videos_flat: <output-dir>/<episode_name>.mp4; others keep files under episode tree",
    )
    parser.add_argument(
        "--inner-mp4-name",
        default="video.mp4",
        help="Used with inside_image_folder (default video.mp4)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing mp4 outputs"
    )
    args = parser.parse_args()

    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not exe.is_file():
        raise FileNotFoundError(f"No ffmpeg exe at {exe}")

    output_dir_resolved = args.output_dir.expanduser().resolve()

    split_root = args.split_root.expanduser().resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {split_root}")

    episodes = resolve_episodes(split_root)
    if not episodes:
        print(f"No episode folders with zarr.json under {split_root}")
        return

    print(f"FFmpeg exe: {exe}")
    print(f"Found {len(episodes)} episode(s).")
    if args.output_style == "videos_flat":
        print(f"Output directory: {output_dir_resolved}")

    for ep in episodes:
        try:
            encode_episode_images(
                episode_dir=ep,
                fps=args.fps,
                image_keys=list(args.image_keys) if args.image_keys else None,
                output_style=args.output_style,
                output_dir=output_dir_resolved
                if args.output_style == "videos_flat"
                else None,
                inner_mp4_name=args.inner_mp4_name,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"[fail] {ep.name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
