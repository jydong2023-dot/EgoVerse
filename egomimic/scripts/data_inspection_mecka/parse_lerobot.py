#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SUPPORTED_SUFFIXES = (
    "/data/chunk-",
    "/meta/episodes/chunk-",
    "/meta/tasks.parquet",
)


def detect_kind(path: Path) -> str:
    path_str = str(path)
    if "/data/chunk-" in path_str and path.name.endswith(".parquet"):
        return "data"
    if "/meta/episodes/chunk-" in path_str and path.name.endswith(".parquet"):
        return "meta_episodes"
    if path_str.endswith("/meta/tasks.parquet"):
        return "meta_tasks"
    raise ValueError(
        "Unsupported parquet path. Expected one of:\n"
        "  .../data/chunk-XXX/file-YYY.parquet\n"
        "  .../meta/episodes/chunk-XXX/file-YYY.parquet\n"
        "  .../meta/tasks.parquet"
    )


def arrow_type_summary(dtype: pa.DataType) -> str:
    if pa.types.is_fixed_size_list(dtype):
        return f"fixed_size_list[{dtype.list_size}]<{arrow_type_summary(dtype.value_type)}>"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return f"list<{arrow_type_summary(dtype.value_type)}>"
    if pa.types.is_struct(dtype):
        fields = ", ".join(
            f"{field.name}: {arrow_type_summary(field.type)}" for field in dtype
        )
        return f"struct<{fields}>"
    return str(dtype)


def build_column_tree(columns: list[str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for column in columns:
        parts = []
        for slash_part in column.split("/"):
            parts.extend(slash_part.split("."))
        node = root
        for part in parts:
            node = node.setdefault(part, {})
    return root


def format_tree(node: dict[str, Any], prefix: str = "") -> list[str]:
    lines = []
    keys = sorted(node.keys())
    for index, key in enumerate(keys):
        is_last = index == len(keys) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{key}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        lines.extend(format_tree(node[key], child_prefix))
    return lines


def summarize_value(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    if depth >= max_depth:
        if isinstance(value, list):
            return f"list(len={len(value)})"
        if isinstance(value, dict):
            return f"dict(keys={list(value.keys())})"
        return type(value).__name__

    if isinstance(value, list):
        if not value:
            return {"type": "list", "len": 0, "sample": []}
        return {
            "type": "list",
            "len": len(value),
            "sample": [summarize_value(value[0], depth + 1, max_depth)],
        }

    if isinstance(value, dict):
        return {
            key: summarize_value(subvalue, depth + 1, max_depth)
            for key, subvalue in value.items()
        }

    return value


def sample_rows(table: pa.Table, max_rows: int) -> list[dict[str, Any]]:
    pylist = table.slice(0, min(max_rows, table.num_rows)).to_pylist()
    return [
        {key: summarize_value(value) for key, value in row.items()} for row in pylist
    ]


def metadata_summary(schema: pa.Schema) -> dict[str, Any]:
    if not schema.metadata:
        return {}
    result = {}
    for key, value in schema.metadata.items():
        key_str = key.decode("utf-8", errors="replace")
        value_str = value.decode("utf-8", errors="replace")
        if len(value_str) > 300:
            value_str = value_str[:300] + "...(truncated)"
        result[key_str] = value_str
    return result


def first_scalar(table: pa.Table, column: str) -> Any:
    if table.num_rows == 0 or column not in table.column_names:
        return None
    return table[column][0].as_py()


def min_scalar(table: pa.Table, column: str) -> Any:
    if table.num_rows == 0 or column not in table.column_names:
        return None
    return pc.min(table[column]).as_py()


def max_scalar(table: pa.Table, column: str) -> Any:
    if table.num_rows == 0 or column not in table.column_names:
        return None
    return pc.max(table[column]).as_py()


def extract_first_episode_data(path: Path, kind: str, rows: int) -> dict[str, Any]:
    table = pq.read_table(path)
    if table.num_rows == 0:
        return {"kind": kind, "rows": []}

    if kind == "data":
        first_episode_index = first_scalar(table, "episode_index")
        first_episode_table = table.filter(
            pc.equal(table["episode_index"], pa.scalar(first_episode_index))
        )
        return {
            "kind": kind,
            "episode_index": first_episode_index,
            "num_rows": first_episode_table.num_rows,
            "frame_index_start": min_scalar(first_episode_table, "frame_index"),
            "frame_index_end": max_scalar(first_episode_table, "frame_index"),
            "timestamp_start": min_scalar(first_episode_table, "timestamp"),
            "timestamp_end": max_scalar(first_episode_table, "timestamp"),
            "rows": first_episode_table.to_pylist(),
            "rows_preview": sample_rows(first_episode_table, rows),
        }

    first_row = table.slice(0, 1).to_pylist()[0]
    return {
        "kind": kind,
        "num_rows": 1,
        "rows": [first_row],
        "rows_preview": [
            {key: summarize_value(value) for key, value in first_row.items()}
        ],
    }


def summarize_first_episode_data(first_episode_data: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in first_episode_data.items() if key != "rows"}
    if "rows_preview" not in summary and "rows" in first_episode_data:
        summary["rows_preview"] = [
            {key: summarize_value(value) for key, value in row.items()}
            for row in first_episode_data["rows"][:1]
        ]
    return summary


def first_episode_json_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.first_episode.json")


def save_first_episode_data(path: Path, first_episode_data: dict[str, Any]) -> Path:
    output_path = first_episode_json_path(path)
    output_path.write_text(
        json.dumps(first_episode_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_file_report(path: Path, rows: int) -> None:
    kind = detect_kind(path)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    columns = schema.names
    first_episode_data = extract_first_episode_data(path, kind, rows)
    json_output_path = save_first_episode_data(path, first_episode_data)

    print(f"path: {path}")
    print(f"kind: {kind}")
    print(f"rows: {parquet.metadata.num_rows}")
    print(f"row_groups: {parquet.num_row_groups}")
    print(f"columns: {len(columns)}")
    print(f"first_episode_json: {json_output_path}")
    print()

    print("column_tree:")
    tree_lines = format_tree(build_column_tree(columns))
    for line in tree_lines:
        print(line)
    print()

    print("schema:")
    for field in schema:
        print(f"- {field.name}: {arrow_type_summary(field.type)}")
    print()

    metadata = metadata_summary(schema)
    print("schema_metadata:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print()

    table = parquet.read_row_groups(list(range(min(parquet.num_row_groups, 1))))
    print(f"sample_rows_first_{min(rows, table.num_rows)}:")
    print(json.dumps(sample_rows(table, rows), ensure_ascii=False, indent=2))
    print()

    print("first_episode_data:")
    print(
        json.dumps(
            summarize_first_episode_data(first_episode_data),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse supported Lerobot parquet files and print their stored structure."
    )
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument(
        "--rows", type=int, default=2, help="Number of sample rows to print."
    )
    args = parser.parse_args()

    path = args.parquet_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.suffix != ".parquet":
        raise ValueError(f"Expected a .parquet file, got: {path}")

    print_file_report(path, rows=max(1, args.rows))


if __name__ == "__main__":
    main()
