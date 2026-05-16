#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from port_egoverse_to_lerobot import EgoVerseEpisodeReader  # noqa: E402


@dataclass
class SplitConfig:
    require_both_hands: bool = True
    min_visible_keypoint_ratio: float = 0.7
    motion_z_threshold: float = 0.9
    camera_weight: float = 0.4
    min_active_sec: float = 2.0
    max_gap_sec: float = 1.0
    merge_gap_sec: float = 0.6
    boundary_pad_sec: float = 0.3
    max_merge_camera_shift_m: float = 0.15


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        std = float(np.nanstd(x))
        scale = std if std > 1e-8 else 1.0
    return (x - med) / scale


def _quat_angle_diff_wxyz(quat: np.ndarray) -> np.ndarray:
    """
    quat: (T, 4), WXYZ.
    returns per-frame angle velocity proxy (rad), length T with first element 0.
    """
    if quat.shape[0] <= 1:
        return np.zeros(quat.shape[0], dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm < 1e-8] = 1.0
    q = q / q_norm
    dots = np.sum(q[1:] * q[:-1], axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return np.concatenate([np.zeros(1, dtype=np.float64), angles], axis=0)


def _fill_short_false_runs(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return mask
    out = mask.copy()
    n = out.shape[0]
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i  # [start, end)
        left_on = start > 0 and out[start - 1]
        right_on = end < n and out[end]
        if left_on and right_on and (end - start) <= max_gap_frames:
            out[start:end] = True
    return out


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i - 1
        runs.append((start, end))
    return runs


def _load_series(
    reader: EgoVerseEpisodeReader,
    key: str,
    fallback_shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    if not reader.has_key(key):
        if fallback_shape is None:
            return None
        return np.full(fallback_shape, np.nan, dtype=np.float64)
    return np.asarray(reader.get_array(key), dtype=np.float64)


def _build_signals(
    reader: EgoVerseEpisodeReader, cfg: SplitConfig
) -> dict[str, np.ndarray]:
    left_kp = np.asarray(
        reader.get_array("left.obs_keypoints"), dtype=np.float64
    ).reshape(-1, 21, 3)
    right_kp = np.asarray(
        reader.get_array("right.obs_keypoints"), dtype=np.float64
    ).reshape(-1, 21, 3)
    n = min(left_kp.shape[0], right_kp.shape[0])
    left_kp = left_kp[:n]
    right_kp = right_kp[:n]

    left_ee = _load_series(reader, "left.obs_ee_pose")
    right_ee = _load_series(reader, "right.obs_ee_pose")
    head_pose = _load_series(reader, "obs_head_pose")
    ts_ns = _load_series(reader, "obs_rgb_timestamps_ns")

    if left_ee is None:
        left_ee = np.concatenate(
            [left_kp[:, 0], np.full((n, 4), np.nan, dtype=np.float64)], axis=1
        )
    else:
        left_ee = left_ee[:n]
    if right_ee is None:
        right_ee = np.concatenate(
            [right_kp[:, 0], np.full((n, 4), np.nan, dtype=np.float64)], axis=1
        )
    else:
        right_ee = right_ee[:n]
    if head_pose is None:
        head_pose = np.full((n, 7), np.nan, dtype=np.float64)
    else:
        head_pose = head_pose[:n]
    if ts_ns is None:
        fps = float(reader.attrs.get("fps") or 30.0)
        times = np.arange(n, dtype=np.float64) / max(fps, 1e-6)
    else:
        ts = np.asarray(ts_ns[:n], dtype=np.float64)
        times = (ts - ts[0]) / 1e9 if ts.size else np.zeros(n, dtype=np.float64)

    left_vis_ratio = np.isfinite(left_kp).all(axis=-1).mean(axis=1)
    right_vis_ratio = np.isfinite(right_kp).all(axis=-1).mean(axis=1)
    left_visible = left_vis_ratio >= cfg.min_visible_keypoint_ratio
    right_visible = right_vis_ratio >= cfg.min_visible_keypoint_ratio
    both_visible = left_visible & right_visible
    any_visible = left_visible | right_visible

    left_pos = left_ee[:, :3]
    right_pos = right_ee[:, :3]
    hand_speed = np.zeros(n, dtype=np.float64)
    if n > 1:
        left_v = np.linalg.norm(np.diff(left_pos, axis=0), axis=1)
        right_v = np.linalg.norm(np.diff(right_pos, axis=0), axis=1)
        hand_speed[1:] = left_v + right_v

    hand_rot = _quat_angle_diff_wxyz(left_ee[:, 3:7]) + _quat_angle_diff_wxyz(
        right_ee[:, 3:7]
    )

    hand_distance = np.linalg.norm(left_pos - right_pos, axis=1)
    hand_distance_delta = np.zeros(n, dtype=np.float64)
    if n > 1:
        hand_distance_delta[1:] = np.abs(np.diff(hand_distance))

    camera_speed = np.zeros(n, dtype=np.float64)
    if n > 1:
        camera_speed[1:] = np.linalg.norm(np.diff(head_pose[:, :3], axis=0), axis=1)

    raw_motion = (
        hand_speed
        + 0.25 * hand_rot
        + 0.35 * hand_distance_delta
        + cfg.camera_weight * camera_speed
    )
    motion = _smooth(raw_motion, window=7)
    motion_z = _robust_zscore(motion)

    return {
        "times": times,
        "left_visible": left_visible,
        "right_visible": right_visible,
        "both_visible": both_visible,
        "any_visible": any_visible,
        "motion": motion,
        "motion_z": motion_z,
        "camera_speed": camera_speed,
        "head_pos": head_pose[:, :3],
    }


def split_episode(reader: EgoVerseEpisodeReader, cfg: SplitConfig) -> dict[str, Any]:
    sig = _build_signals(reader, cfg)
    n = sig["motion"].shape[0]
    fps = float(reader.attrs.get("fps") or 30.0)
    frame_dt = 1.0 / max(fps, 1e-6)

    visibility_gate = (
        sig["both_visible"] if cfg.require_both_hands else sig["any_visible"]
    )
    active_raw = (sig["motion_z"] >= cfg.motion_z_threshold) & visibility_gate

    max_gap_frames = max(1, int(round(cfg.max_gap_sec / frame_dt)))
    min_active_frames = max(1, int(round(cfg.min_active_sec / frame_dt)))
    merge_gap_frames = max(1, int(round(cfg.merge_gap_sec / frame_dt)))
    pad_frames = max(0, int(round(cfg.boundary_pad_sec / frame_dt)))

    active = _fill_short_false_runs(active_raw, max_gap_frames=max_gap_frames)
    runs = [(s, e) for s, e in _true_runs(active) if (e - s + 1) >= min_active_frames]

    # Merge close segments unless camera moved significantly between segments.
    merged: list[tuple[int, int]] = []
    for seg in runs:
        if not merged:
            merged.append(seg)
            continue
        prev_s, prev_e = merged[-1]
        cur_s, cur_e = seg
        gap = cur_s - prev_e - 1
        if gap > merge_gap_frames:
            merged.append(seg)
            continue
        cam_shift = float(
            np.linalg.norm(sig["head_pos"][cur_s] - sig["head_pos"][prev_e])
        )
        if np.isnan(cam_shift) or cam_shift <= cfg.max_merge_camera_shift_m:
            merged[-1] = (prev_s, cur_e)
        else:
            merged.append(seg)

    padded: list[tuple[int, int]] = []
    for s, e in merged:
        s2 = max(0, s - pad_frames)
        e2 = min(n - 1, e + pad_frames)
        if padded and s2 <= padded[-1][1] + 1:
            padded[-1] = (padded[-1][0], max(padded[-1][1], e2))
        else:
            padded.append((s2, e2))

    segments = []
    for i, (s, e) in enumerate(padded):
        seg_motion = sig["motion"][s : e + 1]
        both_ratio = float(sig["both_visible"][s : e + 1].mean())
        seg = {
            "segment_id": i,
            "start_frame": int(s),
            "end_frame": int(e),
            "start_time_sec": float(sig["times"][s]),
            "end_time_sec": float(sig["times"][e]),
            "duration_sec": float(sig["times"][e] - sig["times"][s] + frame_dt),
            "mean_motion": float(np.nanmean(seg_motion)),
            "mean_motion_z": float(np.nanmean(sig["motion_z"][s : e + 1])),
            "both_hands_visible_ratio": both_ratio,
        }
        segments.append(seg)

    return {
        "episode_path": str(reader.episode_dir),
        "task_name": str(reader.attrs.get("task_name", "")),
        "total_frames": int(n),
        "fps": float(fps),
        "config": asdict(cfg),
        "num_segments": len(segments),
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split one EgoVerse episode into multiple task executions using motion/visibility/camera heuristics."
    )
    parser.add_argument(
        "episode_path", type=Path, help="Episode directory containing zarr.json"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <episode>/split_segments.json)",
    )
    parser.add_argument("--require-both-hands", action="store_true", default=True)
    parser.add_argument("--allow-single-hand", action="store_true")
    parser.add_argument("--min-visible-keypoint-ratio", type=float, default=0.7)
    parser.add_argument("--motion-z-threshold", type=float, default=0.9)
    parser.add_argument("--camera-weight", type=float, default=0.4)
    parser.add_argument("--min-active-sec", type=float, default=2.0)
    parser.add_argument("--max-gap-sec", type=float, default=1.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.6)
    parser.add_argument("--boundary-pad-sec", type=float, default=0.3)
    parser.add_argument("--max-merge-camera-shift-m", type=float, default=0.15)
    args = parser.parse_args()

    episode_path = args.episode_path.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else episode_path / "split_segments.json"
    )

    cfg = SplitConfig(
        require_both_hands=not args.allow_single_hand,
        min_visible_keypoint_ratio=args.min_visible_keypoint_ratio,
        motion_z_threshold=args.motion_z_threshold,
        camera_weight=args.camera_weight,
        min_active_sec=args.min_active_sec,
        max_gap_sec=args.max_gap_sec,
        merge_gap_sec=args.merge_gap_sec,
        boundary_pad_sec=args.boundary_pad_sec,
        max_merge_camera_shift_m=args.max_merge_camera_shift_m,
    )

    reader = EgoVerseEpisodeReader(episode_path)
    result = split_episode(reader, cfg)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved split result: {output_json}")
    print(f"Segments: {result['num_segments']}")


if __name__ == "__main__":
    main()
