# 检查 EgoVerse Episode 中的 `obs_keypoints`

本文档说明如何检查 `/home/djy/EgoVerse/data/2025-09-20-17-47-54-000000/` 下的：

- `right.obs_keypoints`
- `left.obs_keypoints`

重点关注这两个节点中是否存在字面意义上的 `keypoints` 键值，以及它们实际存储的数据结构是什么。

## 结论

对该 episode 的检查结果如下：

- `left.obs_keypoints` 和 `right.obs_keypoints` 都存在。
- 它们不是带有内层 `keypoints` 键的对象或字典。
- 这两个节点本身就是 Zarr v3 的数组节点。
- 两个节点的元数据都显示：
  - `node_type = array`
  - `shape = [2900, 63]`
  - `data_type = float64`
- 因此，这里的 keypoints 数据不是保存在某个 `keypoints` 字段下，而是直接作为数组值存储。
- 每一帧是一行长度为 `63` 的向量，通常表示 `21 * 3` 的手部关键点展平结果。

也就是说：

```text
left.obs_keypoints
right.obs_keypoints
```

本身就是“关键点数据”的键，不存在额外一层：

```text
keypoints
```

这样的内层字段。

## 相关脚本

已新增检查脚本：

`/home/djy/EgoVerse/egomimic/scripts/inspect_episode_keypoints.py`

脚本功能：

1. 读取 episode 根目录的 `zarr.json`
2. 检查 `features` 中是否声明了 `left.obs_keypoints` 和 `right.obs_keypoints`
3. 读取两个数组节点各自的 `zarr.json`
4. 判断它们是否是 `array`，以及是否存在字面上的 `keypoints` 键
5. 如果环境中安装了 `zarr`，可进一步尝试读取样本值

## 运行方式

在当前机器上可直接运行：

```bash
python /home/djy/EgoVerse/egomimic/scripts/inspect_episode_keypoints.py \
  --episode-dir /home/djy/EgoVerse/data/2025-09-20-17-47-54-000000
```

如果你想检查别的 episode，只需要替换 `--episode-dir`：

```bash
python /home/djy/EgoVerse/egomimic/scripts/inspect_episode_keypoints.py \
  --episode-dir /path/to/another_episode
```

## 当前这次运行得到的核心输出

脚本对 `/home/djy/EgoVerse/data/2025-09-20-17-47-54-000000` 的检查结论可概括为：

```text
left.obs_keypoints: exists=yes
right.obs_keypoints: exists=yes
literal top-level feature named 'keypoints': no

left.obs_keypoints:
  node_type=array
  shape=[2900, 63]
  has attributes.keypoints=no
  has top-level 'keypoints' field in node metadata=no

right.obs_keypoints:
  node_type=array
  shape=[2900, 63]
  has attributes.keypoints=no
  has top-level 'keypoints' field in node metadata=no
```

## 备注

- 当前环境里没有可直接导入的 `zarr` Python 包，因此这次检查主要基于 `zarr.json` 元数据完成。
- 即便不读取底层 chunk 数值，也已经可以明确回答“是否存在 `keypoints` 键值”这个问题：不存在。
- 如果后续你希望继续读取实际数值样本，可以在安装了 `zarr` 的环境中再次运行同一个脚本。

