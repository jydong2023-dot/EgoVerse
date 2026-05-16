#!/usr/bin/env python

import argparse
import base64
import binascii
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from numcodecs.vlen import VLenBytes
from numcodecs.zstd import Zstd
from tqdm.auto import tqdm

try:
    from lerobot.utils.constants import HF_LEROBOT_HOME
except ImportError:
    # Older lerobot: re-export lived on lerobot_dataset
    from lerobot.datasets.lerobot_dataset import (
        HF_LEROBOT_HOME,  # type: ignore[attr-defined]
    )

MANO_JOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

MAX_UINT64 = 2**64 - 1
_VLEN_BYTES = VLenBytes()


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return x


def _jpeg_payload_to_bytes(jpeg_payload: Any) -> bytes:
    if isinstance(jpeg_payload, np.ndarray):
        if jpeg_payload.dtype == object or jpeg_payload.ndim == 0:
            jpeg_payload = jpeg_payload.item()
        else:
            return jpeg_payload.tobytes()
    if isinstance(jpeg_payload, memoryview):
        return jpeg_payload.tobytes()
    if isinstance(jpeg_payload, bytearray):
        return bytes(jpeg_payload)
    if isinstance(jpeg_payload, bytes):
        return jpeg_payload
    if isinstance(jpeg_payload, str):
        if not jpeg_payload:
            return b""
        payload = jpeg_payload
        if payload.startswith("data:image"):
            payload = payload.split(",", 1)[-1]
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return payload.encode("latin1")
    raise TypeError(f"JPEG payload must be bytes-like, got {type(jpeg_payload)}")


def decode_jpeg_rgb(jpeg_payload: Any) -> np.ndarray:
    raw = _jpeg_payload_to_bytes(jpeg_payload)
    if len(raw) == 0:
        raise ValueError("JPEG payload is empty")
    try:
        import simplejpeg

        return simplejpeg.decode_jpeg(raw, colorspace="RGB")
    except Exception:
        pass
    try:
        from io import BytesIO

        from PIL import Image

        return np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except Exception:
        pass
    try:
        import cv2

        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"JPEG decode failed: {exc}") from exc


def empty_annotation(task=""):
    return {"task": task, "actions": []}


def load_episode_attrs(episode_dir: Path) -> dict[str, Any]:
    meta = json.loads((episode_dir / "zarr.json").read_text())
    return dict(meta.get("attributes", {}))


def coerce_annotation_entry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return coerce_annotation_entry(value.item())
    if isinstance(value, (bytes, bytearray)):
        try:
            return coerce_annotation_entry(json.loads(value.decode("utf-8")))
        except Exception:
            return None
    if isinstance(value, str):
        if not value:
            return None
        try:
            return coerce_annotation_entry(json.loads(value))
        except Exception:
            return None
    return None


def load_annotations_from_npy(npy_path: Path) -> list[dict[str, Any]]:
    arr = np.load(npy_path, allow_pickle=True)
    return [
        entry
        for entry in (coerce_annotation_entry(v) for v in arr)
        if entry is not None
    ]


class EgoVerseEpisodeReader:
    def __init__(self, episode_dir: Path):
        self.episode_dir = episode_dir
        self.attrs = load_episode_attrs(episode_dir)
        self.readable_dir = episode_dir / "readable_exports"
        self._cache: dict[str, np.ndarray] = {}

    def has_key(self, key: str) -> bool:
        if (self.readable_dir / f"{key}.npy").exists():
            return True
        features = self.attrs.get("features", {})
        return key in features

    def _key_dir(self, key: str) -> Path:
        return self.episode_dir / key

    def _load_key_meta(self, key: str) -> dict[str, Any]:
        return json.loads((self._key_dir(key) / "zarr.json").read_text())

    def _find_shard_file(self, key: str) -> Path:
        chunk_root = self._key_dir(key) / "c"
        files = sorted(p for p in chunk_root.rglob("*") if p.is_file())
        if not files:
            raise FileNotFoundError(
                f"No shard files found for {key} under {chunk_root}"
            )
        return files[0]

    def _decode_chunk(self, chunk_bytes: bytes, meta: dict[str, Any]) -> np.ndarray:
        codec_specs = meta["codecs"][0]["configuration"]["codecs"]
        payload: bytes | np.ndarray = chunk_bytes
        for spec in reversed(codec_specs):
            name = spec["name"]
            config = spec.get("configuration", {})
            if name == "zstd":
                payload = Zstd(
                    level=int(config.get("level", 0)),
                    checksum=bool(config.get("checksum", False)),
                ).decode(payload)  # type: ignore[arg-type]
            elif name == "bytes":
                continue
            elif name == "vlen-bytes":
                payload = _VLEN_BYTES.decode(np.frombuffer(payload, dtype=np.uint8))  # type: ignore[arg-type]
            else:
                raise NotImplementedError(
                    f"Unsupported codec {name} for {self.episode_dir.name}"
                )

        chunk_shape = tuple(meta["codecs"][0]["configuration"]["chunk_shape"])
        data_type = meta["data_type"]
        if isinstance(payload, np.ndarray):
            return payload.reshape(chunk_shape)

        dtype = np.dtype(data_type)
        return np.frombuffer(payload, dtype=dtype).reshape(chunk_shape)

    def _decode_sharded_array(self, key: str) -> np.ndarray:
        meta = self._load_key_meta(key)
        shape = tuple(meta["shape"])
        if 0 in shape:
            if meta["data_type"] == "variable_length_bytes":
                return np.empty(shape, dtype=object)
            return np.empty(shape, dtype=np.dtype(meta["data_type"]))

        shard_file = self._find_shard_file(key)
        shard_bytes = shard_file.read_bytes()
        inner_chunk_shape = tuple(meta["codecs"][0]["configuration"]["chunk_shape"])
        chunks_per_shard = tuple(
            s // c for s, c in zip(shape, inner_chunk_shape, strict=False)
        )
        index_entries = int(np.prod(chunks_per_shard))
        index_size = index_entries * 16 + 4
        if meta["codecs"][0]["configuration"].get("index_location", "end") != "end":
            raise NotImplementedError("Only end-located shard indices are supported")

        index = np.frombuffer(shard_bytes[-index_size:-4], dtype="<u8").reshape(
            chunks_per_shard + (2,)
        )

        if meta["data_type"] == "variable_length_bytes":
            fill_value = meta.get("fill_value", "")
            out = np.empty(shape, dtype=object)
            out.fill(fill_value)
        else:
            dtype = np.dtype(meta["data_type"])
            fill_value = meta.get("fill_value", 0)
            out = np.full(shape, fill_value, dtype=dtype)

        for chunk_coords in np.ndindex(*chunks_per_shard):
            offset, length = index[chunk_coords]
            if offset == MAX_UINT64 or length == MAX_UINT64:
                continue
            chunk = shard_bytes[int(offset) : int(offset + length)]
            decoded = self._decode_chunk(chunk, meta)
            slices = tuple(
                slice(coord * size, (coord + 1) * size)
                for coord, size in zip(chunk_coords, inner_chunk_shape, strict=False)
            )
            out[slices] = decoded
        return out

    def get_array(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]

        npy_path = self.readable_dir / f"{key}.npy"
        if npy_path.exists():
            arr = np.load(npy_path, allow_pickle=True)
        else:
            arr = self._decode_sharded_array(key)
        self._cache[key] = arr
        return arr

    def get_image(self, index: int, image_key: str = "images.front_1") -> np.ndarray:
        images = self.get_array(image_key)
        jpeg_payload = images[index]
        return decode_jpeg_rgb(jpeg_payload)

    def load_annotations(self) -> dict[str, Any]:
        task = str(
            self.attrs.get("task_description")
            or self.attrs.get("task_name")
            or self.episode_dir.name
        )

        npy_path = self.readable_dir / "annotations.npy"
        if npy_path.exists():
            return {"task": task, "actions": load_annotations_from_npy(npy_path)}

        if not self.has_key("annotations"):
            return empty_annotation(task=task)

        try:
            raw = self.get_array("annotations")
        except Exception as exc:
            logging.warning(
                f"Failed to read annotations for {self.episode_dir.name}: {exc}"
            )
            return empty_annotation(task=task)

        actions = [
            entry
            for entry in (coerce_annotation_entry(v) for v in raw)
            if entry is not None
        ]
        return {"task": task, "actions": actions}


def build_features(image_shape):
    hand_axes = [f"{j}_{a}" for j in MANO_JOINT_NAMES for a in ("x", "y", "z")]
    state_axes = [f"left_{a}" for a in hand_axes] + [f"right_{a}" for a in hand_axes]
    vis_axes = [f"left_{j}" for j in MANO_JOINT_NAMES] + [
        f"right_{j}" for j in MANO_JOINT_NAMES
    ]
    n_joints = len(MANO_JOINT_NAMES)
    return {
        "is_first": {"dtype": "bool", "shape": (1,), "names": None},
        "is_last": {"dtype": "bool", "shape": (1,), "names": None},
        "is_terminal": {"dtype": "bool", "shape": (1,), "names": None},
        "subtask": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_objects": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_actors": {"dtype": "string", "shape": (1,), "names": None},
        "observation.images.egocentric": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state.intrinsics": {
            "dtype": "float32",
            "shape": (3, 3),
            "names": None,
        },
        "observation.state.head_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["xyz_quat"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (n_joints * 6,),
            "names": {"axes": state_axes},
        },
        "observation.state.visibility": {
            "dtype": "int32",
            "shape": (n_joints * 2,),
            "names": {"axes": vis_axes},
        },
        "action": {
            "dtype": "float32",
            "shape": (n_joints * 6,),
            "names": {"axes": state_axes},
        },
        "action.visibility": {
            "dtype": "int32",
            "shape": (n_joints * 2,),
            "names": {"axes": vis_axes},
        },
    }


def empty_int(shape):
    return np.zeros(shape, dtype=np.int32)


def pad_or_trim(
    array: np.ndarray | None, n_frames: int, frame_shape: tuple[int, ...], dtype
) -> np.ndarray:
    if array is None:
        if np.issubdtype(np.dtype(dtype), np.integer):
            return np.zeros((n_frames,) + frame_shape, dtype=dtype)
        return np.full((n_frames,) + frame_shape, np.nan, dtype=dtype)

    arr = np.asarray(array)
    if arr.ndim == len(frame_shape):
        arr = np.broadcast_to(arr, (n_frames,) + frame_shape)
    if arr.ndim != len(frame_shape) + 1:
        if np.issubdtype(np.dtype(dtype), np.integer):
            return np.zeros((n_frames,) + frame_shape, dtype=dtype)
        return np.full((n_frames,) + frame_shape, np.nan, dtype=dtype)

    arr = arr.astype(dtype, copy=False)
    current = min(n_frames, arr.shape[0])
    out_shape = (n_frames,) + frame_shape
    if np.issubdtype(np.dtype(dtype), np.integer):
        out = np.zeros(out_shape, dtype=dtype)
    else:
        out = np.full(out_shape, np.nan, dtype=dtype)
    out[:current] = arr[:current]
    return out


def infer_visibility(joints: np.ndarray) -> np.ndarray:
    n_joints = len(MANO_JOINT_NAMES)
    reshaped = joints.reshape(joints.shape[0], n_joints, 3)
    return np.isfinite(reshaped).all(axis=-1).astype(np.int32)


def get_image_shape(attrs: dict[str, Any]) -> tuple[int, int, int]:
    image_meta = attrs.get("features", {}).get("images.front_1", {})
    shape = tuple(image_meta.get("shape", (480, 640, 3)))
    if len(shape) != 3:
        return (480, 640, 3)
    return shape  # type: ignore[return-value]


def _intrinsics_from_attrs_dict(mapping: dict[str, Any]) -> np.ndarray | None:
    """Parse camera matrix from episode root metadata (often Mecka-style dict)."""
    for key in (
        "K",
        "camera_matrix",
        "intrinsics_matrix",
        "matrix",
        "data",
        "cam_matrix",
    ):
        if key not in mapping:
            continue
        nested = coerce_intrinsics(mapping[key])
        if nested is not None:
            return nested

    fx_any = mapping.get("fl_x", mapping.get("fx"))
    fy_any = mapping.get("fl_y", mapping.get("fy"))
    focal_any = mapping.get("focal_length")

    if fx_any is None and focal_any is not None:
        fx_any = focal_any
    if fy_any is None and focal_any is not None:
        fy_any = focal_any
    # Square-pixel fallback when only one focal length axis is stored.
    if fx_any is None and fy_any is not None:
        fx_any = fy_any
    if fy_any is None and fx_any is not None:
        fy_any = fx_any

    cx_any = mapping.get("cx")
    cy_any = mapping.get("cy")
    if fx_any is None or fy_any is None or cx_any is None or cy_any is None:
        return None

    try:
        fx = float(fx_any)
        fy = float(fy_any)
        cx = float(cx_any)
        cy = float(cy_any)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fx) or not np.isfinite(fy) or fx == 0.0 or fy == 0.0:
        return None

    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def coerce_intrinsics(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if not value:
            return None
        return _intrinsics_from_attrs_dict(value)
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.dtype == np.dtype(object):
        return None
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


def get_episode_intrinsics(reader: EgoVerseEpisodeReader) -> np.ndarray:
    for key in ("intrinsics", "camera_intrinsics", "camera_matrix", "K"):
        intrinsics = coerce_intrinsics(reader.attrs.get(key))
        if intrinsics is not None:
            return intrinsics

    for key in (
        "intrinsics",
        "camera.intrinsics",
        "camera_intrinsics",
        "observation.state.intrinsics",
    ):
        if not reader.has_key(key):
            continue
        try:
            arr = reader.get_array(key)
        except Exception as exc:
            logging.warning(
                f"Failed to read intrinsics key '{key}' for {reader.episode_dir.name}: {exc}"
            )
            continue
        if arr.ndim >= 3 and arr.shape[-2:] == (3, 3):
            intrinsics = coerce_intrinsics(arr[0])
        else:
            intrinsics = coerce_intrinsics(arr)
        if intrinsics is not None:
            return intrinsics

    return np.eye(3, dtype=np.float32)


def get_episode_fps(
    attrs: dict[str, Any], timestamps_ns: np.ndarray | None = None
) -> float:
    fps = float(attrs.get("fps") or 0.0)
    if fps > 0:
        return fps
    if timestamps_ns is not None and timestamps_ns.size > 1:
        deltas = np.diff(timestamps_ns.astype(np.float64)) / 1e9
        deltas = deltas[deltas > 0]
        if deltas.size:
            return float(1.0 / np.median(deltas))
    return 30.0


def get_frame_times(
    timestamps_ns: np.ndarray | None, n_frames: int, fps: float
) -> np.ndarray:
    if timestamps_ns is not None and timestamps_ns.size >= n_frames:
        ts = timestamps_ns[:n_frames].astype(np.float64)
        return (ts - ts[0]) / 1e9
    if fps <= 0:
        return np.arange(n_frames, dtype=np.float64)
    return np.arange(n_frames, dtype=np.float64) / fps


def normalize_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        start_t = float(
            action.get("start_timestamp", action.get("start_time", 0.0)) or 0.0
        )
        end_t = float(
            action.get("end_timestamp", action.get("end_time", start_t)) or start_t
        )
        normalized.append(
            {
                **action,
                "start_timestamp": start_t,
                "end_timestamp": end_t,
                "label": str(action.get("label", action.get("name", "")) or ""),
                "objects": action.get("objects", []),
                "actors": action.get("actors", []),
            }
        )
    return sorted(normalized, key=lambda x: x["start_timestamp"])


def find_active_action(
    actions: list[dict[str, Any]], t: float, start_index: int
) -> tuple[dict[str, Any], int]:
    idx = start_index
    while idx < len(actions) and t >= float(actions[idx].get("end_timestamp", -1.0)):
        idx += 1
    active = {}
    if idx < len(actions):
        start_t = float(actions[idx].get("start_timestamp", np.inf))
        end_t = float(actions[idx].get("end_timestamp", -np.inf))
        if start_t <= t < end_t:
            active = actions[idx]
    return active, idx


def discover_episode_dirs(raw_dir: Path) -> list[Path]:
    episode_dirs = sorted(
        p for p in raw_dir.iterdir() if p.is_dir() and (p / "zarr.json").exists()
    )
    if not episode_dirs:
        raise ValueError(f"No EgoVerse episodes found under {raw_dir}")
    return episode_dirs


def port_egoverse(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    private: bool = False,
    root: Path | None = None,
    overwrite: bool = False,
):
    episode_dirs = discover_episode_dirs(raw_dir)
    first_reader = EgoVerseEpisodeReader(episode_dirs[0])
    first_shape = get_image_shape(first_reader.attrs)

    first_timestamps = None
    if first_reader.has_key("obs_rgb_timestamps_ns"):
        first_timestamps = np.asarray(first_reader.get_array("obs_rgb_timestamps_ns"))
    dataset_fps = (
        int(round(get_episode_fps(first_reader.attrs, first_timestamps))) or 30
    )
    features = build_features(first_shape)

    output_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output path already exists: {output_root}. "
                "Use --overwrite to remove it first, or set --root to a new path."
            )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="egoverse",
        fps=dataset_fps,
        features=features,
        root=output_root,
    )

    n_joints = len(MANO_JOINT_NAMES)
    empty_visibility = empty_int((n_joints * 2,))

    for episode_index, episode_dir in enumerate(
        tqdm(episode_dirs, desc="Episodes", unit="episode", dynamic_ncols=True)
    ):
        reader = EgoVerseEpisodeReader(episode_dir)
        annotation = reader.load_annotations()

        timestamps = None
        if reader.has_key("obs_rgb_timestamps_ns"):
            timestamps = np.asarray(reader.get_array("obs_rgb_timestamps_ns"))

        left_raw = (
            reader.get_array("left.obs_keypoints")
            if reader.has_key("left.obs_keypoints")
            else None
        )
        right_raw = (
            reader.get_array("right.obs_keypoints")
            if reader.has_key("right.obs_keypoints")
            else None
        )
        head_pose_raw = (
            reader.get_array("obs_head_pose")
            if reader.has_key("obs_head_pose")
            else None
        )

        intrinsics = get_episode_intrinsics(reader)

        candidate_lengths = [
            int(reader.attrs.get("total_frames") or 0),
            int(timestamps.shape[0]) if timestamps is not None else 0,
            int(left_raw.shape[0])
            if isinstance(left_raw, np.ndarray) and left_raw.ndim >= 2
            else 0,
            int(right_raw.shape[0])
            if isinstance(right_raw, np.ndarray) and right_raw.ndim >= 2
            else 0,
            int(head_pose_raw.shape[0])
            if isinstance(head_pose_raw, np.ndarray) and head_pose_raw.ndim >= 2
            else 0,
        ]
        n_frames = min((v for v in candidate_lengths if v > 0), default=0)
        if n_frames <= 0:
            logging.warning(
                f"Skipping {episode_dir.name}: cannot determine frame count"
            )
            continue

        left = pad_or_trim(left_raw, n_frames, (n_joints * 3,), np.float32)
        right = pad_or_trim(right_raw, n_frames, (n_joints * 3,), np.float32)
        head_pose = pad_or_trim(head_pose_raw, n_frames, (7,), np.float32)
        left_vis = (
            infer_visibility(left)
            if left_raw is not None
            else np.zeros((n_frames, n_joints), dtype=np.int32)
        )
        right_vis = (
            infer_visibility(right)
            if right_raw is not None
            else np.zeros((n_frames, n_joints), dtype=np.int32)
        )

        task = str(annotation.get("task", ""))
        actions = normalize_actions(annotation.get("actions", []))
        fps_value = get_episode_fps(reader.attrs, timestamps) or float(dataset_fps)
        frame_times = get_frame_times(timestamps, n_frames, fps_value)

        image_shape = get_image_shape(reader.attrs)
        if image_shape != first_shape:
            logging.warning(
                f"Image shape {image_shape} differs from dataset shape {first_shape} for {episode_dir.name}. "
                "Frames with mismatched shape will use an empty image."
            )
        empty_image = np.zeros(first_shape, dtype=np.uint8)

        logging.info(
            f"{episode_index + 1}/{len(episode_dirs)} {episode_dir.name} frames={n_frames}"
        )
        action_i = 0
        warned_image_decode_failure = False

        for frame_index in tqdm(
            range(n_frames),
            desc=episode_dir.name,
            unit="frame",
            leave=False,
            dynamic_ncols=True,
        ):
            t = float(frame_times[frame_index])
            active, action_i = find_active_action(actions, t, action_i)

            try:
                image = reader.get_image(frame_index)
            except Exception as exc:
                if not warned_image_decode_failure:
                    logging.warning(
                        f"Failed to decode image for {episode_dir.name} frame {frame_index}: {exc}. "
                        "Using an empty image for this and subsequent undecodable frames."
                    )
                    warned_image_decode_failure = True
                image = empty_image
            if image.shape != first_shape:
                logging.warning(
                    f"Image shape {image.shape} differs from dataset shape {first_shape} "
                    f"for {episode_dir.name} frame {frame_index}. Using an empty image."
                )
                image = empty_image

            left_state = left[frame_index].reshape(-1).astype(np.float32)
            right_state = right[frame_index].reshape(-1).astype(np.float32)
            state = np.concatenate([left_state, right_state]).astype(np.float32)

            if left_raw is None and right_raw is None:
                visibility = empty_visibility
            else:
                visibility = np.concatenate(
                    [left_vis[frame_index], right_vis[frame_index]]
                ).astype(np.int32)

            next_index = min(frame_index + 1, n_frames - 1)
            next_state = np.concatenate([left[next_index], right[next_index]]).astype(
                np.float32
            )
            next_visibility = (
                np.concatenate([left_vis[next_index], right_vis[next_index]]).astype(
                    np.int32
                )
                if (left_raw is not None or right_raw is not None)
                else empty_visibility
            )

            frame = {
                "is_first": np.array([frame_index == 0]),
                "is_last": np.array([frame_index == n_frames - 1]),
                "is_terminal": np.array([frame_index == n_frames - 1]),
                "task": task,
                "subtask": str(active.get("label", "")),
                "subtask_objects": json.dumps(
                    to_jsonable(active.get("objects", [])), ensure_ascii=False
                ),
                "subtask_actors": json.dumps(
                    to_jsonable(active.get("actors", [])), ensure_ascii=False
                ),
                "observation.images.egocentric": image,
                "observation.state.intrinsics": intrinsics,
                "observation.state.head_pose": head_pose[frame_index],
                "observation.state": state,
                "observation.state.visibility": visibility,
                "action": next_state,
                "action.visibility": next_visibility,
            }
            dataset.add_frame(frame)

        dataset.save_episode()

    dataset.finalize()
    if push_to_hub:
        dataset.push_to_hub(tags=["egoverse"], private=private)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    port_egoverse(**vars(args))


if __name__ == "__main__":
    main()
