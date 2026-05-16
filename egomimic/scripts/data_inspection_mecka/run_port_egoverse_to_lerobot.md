# 运行 `data_inspection_mecka/port_egoverse_to_lerobot.py`

# !!!原始.zarr数据中的'task_name'为'debug', 因此现有代码转换lerobot后的'task_name'均为'debug'，需要修改为，从下载的父文件夹（以'task_name'命名）提取相关信息。

脚本路径：`egomimic/scripts/data_inspection_mecka/port_egoverse_to_lerobot.py`  
作用：把 EgoVerse（含 Mecka 处理后的）episode 目录（每层子文件夹内需有 `zarr.json`）转换为 LeRobot Dataset。  


## 环境与依赖

- Python 环境中需已安装：**`lerobot`**、**`numpy`**、**`tqdm`**、**`numcodecs`**。  
  解码 JPEG 帧时，建议至少安装 **`simplejpeg`**、**`Pillow`** 或 **`opencv-python`** 之一。
- `--push-to-hub` 需要：`huggingface_hub` 登录态（例如 `HF_TOKEN` 环境变量或 `huggingface-cli login`）。

## 输入目录要求

`--raw-dir` 指向**数据集根目录**，其下一层每个子文件夹若包含 `zarr.json`，即视为一条 episode。

以下为仓库内示例数据根路径（按需替换为你的数据路径）：

- **示例输入根目录：** `/home/djy/EgoVerse/data/mecka/example_data/`

## Mecka 示例数据 — 最小运行指令

使用 LeRobot 默认输出位置（`HF_LEROBOT_HOME/repo_id`）：

```bash
python /home/djy/EgoVerse/egomimic/scripts/data_inspection_mecka/port_egoverse_to_lerobot.py \
  --raw-dir /home/djy/EgoVerse/data/mecka/example_data \
  --repo-id local/mecka_example_data \
  
```

## 指定本地输出路径（推荐）

```bash
python /home/djy/EgoVerse/egomimic/scripts/data_inspection_mecka/port_egoverse_to_lerobot.py \
  --raw-dir /home/djy/EgoVerse/data/mecka/example_data \
  --repo-id local/mecka_example_data \
  --root /home/djy/EgoVerse/lerobot_output/mecka_example_data
```

不写 `--root` 时，输出目录默认为 **`HF_LEROBOT_HOME/repo_id`**（与 LeRobot 配置一致）。

## 覆盖已有输出

若 `--root` 或默认输出路径已存在，需加上 `--overwrite` 才会删除后重写：

```bash
python /home/djy/EgoVerse/egomimic/scripts/data_inspection_mecka/port_egoverse_to_lerobot.py \
  --raw-dir /home/djy/EgoVerse/data/mecka/example_data \
  --repo-id local/mecka_example_data \
  --root /home/djy/EgoVerse/lerobot_output/mecka_example_data \
  --overwrite
```

## 上传到 Hugging Face Hub

```bash
python /home/djy/EgoVerse/egomimic/scripts/data_inspection_mecka/port_egoverse_to_lerobot.py \
  --raw-dir /home/djy/EgoVerse/data/mecka/example_data \
  --repo-id your_username/your_dataset_name \
  --push-to-hub \
  --private
```

如需公开仓库，去掉 `--private`。

## 命令行参数一览

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--raw-dir` | 是 | EgoVerse episodes 的根目录 |
| `--repo-id` | 是 | LeRobot 数据集标识（常与 Hub repo 名对应） |
| `--root` | 否 | 本地输出根路径；默认为 `HF_LEROBOT_HOME/repo_id` |
| `--overwrite` | 否 | 输出路径已存在时先删除再写 |
| `--push-to-hub` | 否 | 转换完成后推送到 Hugging Face |
| `--private` | 否 | 配合 `--push-to-hub`，创建私有仓库 |

## 常见问题

1. **`ModuleNotFoundError: lerobot`**：在激活的 venv / conda 中安装 `lerobot`，或改用该环境的 `python` 完整路径执行。  
2. **JPEG 解码失败**：脚本会告警并可能对单帧使用空图；安装 `simplejpeg` 通常最省事。  
3. **权限或路径**：确保对 `--raw-dir` 可读、对 `--root` 或默认输出路径可写。
