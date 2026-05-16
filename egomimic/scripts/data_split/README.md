# EgoVerse 单条 Episode 切分方案（一次任务执行一段）

本文针对如下数据形态：

- 目录：`/home/djy/EgoVerse/data/2025-09-20-17-47-54-000000/`
- `task_name=fold_clothes`
- 单条 episode 内可能重复执行同一任务多次（无显式分段标注）

目标：把一条 episode 自动切成多个 segment，每个 segment 尽量对应“一次完整任务执行”。

---

## 1. 为什么不能只看一个信号

只用单一规则（例如“手出现就算开始”）通常会误切：

- 手短暂离开画面（遮挡）会把同一次执行切成多段；
- 相机/head 有明显移动时，可能意味着一次执行结束后重置到下一次；
- 两手都可见但几乎不动，往往是准备/停顿，不应当算执行段。

因此需要联合三个信号：

1. **双手可见度**（来自 `left/right.obs_keypoints` 是否有限值）
2. **手部运动强度**（`left/right.obs_ee_pose` 的平移 + 旋转变化）
3. **相机运动**（`obs_head_pose` 的位置变化，用于区分相邻段是否应合并）

---

## 2. 采用的切分逻辑（启发式）

脚本：`split_egoverse_episode.py`

### 2.1 逐帧构造信号

- `left/right_visible_ratio`：21 个 keypoints 中有效点比例
- `both_visible`：左右手都达到可见阈值（默认 `>=0.7`）
- `hand_speed`：左右手 `obs_ee_pose[:3]` 帧间位移速度和
- `hand_rot`：四元数（WXYZ）帧间角度变化
- `hand_distance_delta`：双手间距离变化速度
- `camera_speed`：`obs_head_pose[:3]` 帧间位移

综合运动分数：

`motion = smooth(hand_speed + 0.25*hand_rot + 0.35*hand_distance_delta + camera_weight*camera_speed)`

再做 robust z-score 得到 `motion_z`。

### 2.2 初始活动区间检测

- `active_raw = (motion_z >= threshold) AND visibility_gate`
- 默认 `visibility_gate=both_visible`（可通过 `--allow-single-hand` 放宽）

### 2.3 去碎片/连通处理

- 填补短空洞（`max_gap_sec`）以避免遮挡造成断裂
- 过滤过短段（`min_active_sec`）
- 对相邻短间隔段尝试合并：若间隔短且 head 位移小于阈值，则合并；否则保留分割边界
- 最后做边界 padding（`boundary_pad_sec`）得到更完整的执行段

---

## 3. 运行方式

```bash
python /home/djy/EgoVerse/egomimic/scripts/data_split/split_egoverse_episode.py \
  /home/djy/EgoVerse/data/2025-09-20-17-47-54-000000 \
  --output-json /home/djy/EgoVerse/egomimic/scripts/data_split/2025-09-20-17-47-54-000000_segments.json
```

输出 JSON 包含：

- `num_segments`
- 每段 `start_frame/end_frame`
- `start_time_sec/end_time_sec`
- `duration_sec`
- `mean_motion_z`
- `both_hands_visible_ratio`

---

## 4. 建议的调参顺序

如果段数太多（过切）：

1. 提高 `--motion-z-threshold`（例如 0.9 -> 1.2）
2. 提高 `--min-active-sec`
3. 增大 `--max-gap-sec` / `--merge-gap-sec`

如果段数太少（欠切）：

1. 降低 `--motion-z-threshold`
2. 降低 `--max-gap-sec`
3. 降低 `--max-merge-camera-shift-m`（让相机变化更容易触发分段）

---

## 5. 当前方案的边界

- 它是启发式，不是语义级动作理解；面对极慢动作或频繁遮挡仍可能误差。
- 若后续有标注数据，建议把本脚本输出当作初始 proposal，再做人工修订或训练一个段边界模型。

