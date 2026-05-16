#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from port_egoverse_to_lerobot import EgoVerseEpisodeReader
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import EgoVerseEpisodeReader from port_egoverse_to_lerobot.py. "
        "Please ensure required deps are installed (e.g. numcodecs, pyarrow, lerobot)."
    ) from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return value


def _write_csv(array: np.ndarray, csv_path: Path) -> None:
    arr = np.asarray(array)
    if arr.ndim == 0:
        value = _jsonable(arr.item())
        csv_path.write_text(
            f"value\n{json.dumps(value, ensure_ascii=False)}\n", encoding="utf-8"
        )
        return

    if arr.size == 0:
        csv_path.write_text("value\n", encoding="utf-8")
        return

    if arr.dtype.kind in ("i", "u", "f", "b"):
        if arr.ndim == 1:
            np.savetxt(csv_path, arr, delimiter=",", fmt="%.18e")
        else:
            flat = arr.reshape(arr.shape[0], -1)
            np.savetxt(csv_path, flat, delimiter=",", fmt="%.18e")
        return

    # Object/string-like arrays: write one JSON value per line.
    rows = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    lines = ["value"]
    for row in rows:
        if row.size == 1:
            val = _jsonable(row[0])
        else:
            val = _jsonable(row.tolist())
        lines.append(json.dumps(val, ensure_ascii=False))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_episode(
    episode_path: Path, output_dir: Path, overwrite: bool = False
) -> None:
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode path does not exist: {episode_path}")
    if not (episode_path / "zarr.json").exists():
        raise FileNotFoundError(f"Missing zarr.json under: {episode_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    reader = EgoVerseEpisodeReader(episode_path)
    features = reader.attrs.get("features", {})

    converted: list[tuple[str, str, tuple[int, ...], Path, Path]] = []
    skipped: list[tuple[str, str]] = []

    for key in sorted(features.keys()):
        if key == output_dir.name:
            skipped.append((key, "output directory"))
            continue
        if key.startswith("images."):
            skipped.append((key, "image directory"))
            continue

        npy_path = output_dir / f"{key}.npy"
        csv_path = output_dir / f"{key}.csv"
        if not overwrite and (npy_path.exists() or csv_path.exists()):
            raise FileExistsError(
                f"Output exists for key '{key}'. Use --overwrite to replace existing files."
            )

        arr = np.asarray(reader.get_array(key))
        np.save(npy_path, arr, allow_pickle=True)
        _write_csv(arr, csv_path)
        converted.append((key, str(arr.dtype), tuple(arr.shape), npy_path, csv_path))

    summary_lines = [
        f"Output directory: {output_dir}",
        f"Converted arrays: {len(converted)}",
    ]
    for key, dtype, shape, npy_path, csv_path in converted:
        summary_lines.extend(
            [
                f"- {key}: dtype={dtype}, shape={shape}",
                f"  npy={npy_path}",
                f"  csv={csv_path}",
            ]
        )
    summary_lines.append(f"Skipped entries: {len(skipped)}")
    for key, reason in skipped:
        summary_lines.append(f"- {key}: {reason}")

    summary_path = output_dir / "_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    fields = [
        {"field": key, "dtype": dtype, "size": list(shape)}
        for key, dtype, shape, _, _ in converted
    ]
    info = {
        "source_summary": str(summary_path),
        "num_fields": len(fields),
        "fields": fields,
    }
    (output_dir / "fields_dtype_size.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export EgoVerse episode zarr arrays into readable_exports (*.npy/*.csv)."
    )
    parser.add_argument(
        "episode_path", type=Path, help="Path to episode folder containing zarr.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <episode_path>/readable_exports)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-key outputs if they already exist.",
    )
    args = parser.parse_args()

    episode_path = args.episode_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else episode_path / "readable_exports"
    )

    export_episode(
        episode_path=episode_path, output_dir=output_dir, overwrite=args.overwrite
    )
    print(f"Exported readable files to: {output_dir}")


if __name__ == "__main__":
    main()
