# 运行 `parse_lerobot.py`（以 `egoverse_lerobot` 为例）

脚本路径：`/home/djy/EgoVerse/egomimic/scripts/parse_lerobot.py`  

作用：读取 **LeRobot 数据集目录下符合约定的单个 `.parquet` 文件**，在终端打印 schema、列树、样例行等；同时在**同目录**自动生成 `*.first_episode.json`（与输入 parquet Stem 同名），便于查看首条 episode 相关摘要。

数据集根目录示例：`/home/djy/EgoVerse/lerobot_output/egoverse_lerobot/`  

对应本仓库导出的 parquet 通常为：

| 用途 | 路径 |
|------|------|
| 帧级数据（observations / actions 等） | `data/chunk-000/file-000.parquet` |
| episode 级元数据 | `meta/episodes/chunk-000/file-000.parquet` |
| 任务列表 | `meta/tasks.parquet` |

## 环境与依赖

- 需安装 **`pyarrow`**（脚本通过 `pyarrow.parquet` 读取文件）。

```bash
pip install pyarrow
```

## 重要说明

- 第一个参数是 **单个 parquet 文件的绝对路径或相对路径**，不能传数据集根目录。
- 仅支持三类路径约定（不满足会报错）：`.../data/chunk-*/*.parquet`、`.../meta/episodes/chunk-*/*.parquet`、`.../meta/tasks.parquet`。
- **`--rows`**：打印前几条样例行摘要，默认 `2`，至少为 `1`。

## 示例：解析 `egoverse_lerobot` 三套 parquet

在终端执行（任选其一或依次执行）：

**1）帧数据（最常看）**

```bash
python /home/djy/EgoVerse/egomimic/scripts/parse_lerobot.py \
  /home/djy/EgoVerse/lerobot_output/egoverse_lerobot/data/chunk-000/file-000.parquet \
  --rows 2
```

**2）episode 元数据**

```bash
python /home/djy/EgoVerse/egomimic/scripts/parse_lerobot.py \
  /home/djy/EgoVerse/lerobot_output/egoverse_lerobot/meta/episodes/chunk-000/file-000.parquet \
  --rows 2
```

**3）tasks**

```bash
python /home/djy/EgoVerse/egomimic/scripts/parse_lerobot.py \
  /home/djy/EgoVerse/lerobot_output/egoverse_lerobot/meta/tasks.parquet \
  --rows 2
```

若在项目根目录下，也可用相对路径，例如：

```bash
cd /home/djy/EgoVerse
python egomimic/scripts/parse_lerobot.py \
  lerobot_output/egoverse_lerobot/data/chunk-000/file-000.parquet
```

## 补充：导出 episode 的 `readable_exports`

若你想把原始 episode zarr 数据导出为便于查看的 `*.npy/*.csv`，可运行：

```bash
python /home/djy/EgoVerse/egomimic/scripts/export_readable_exports.py \
  /home/djy/EgoVerse/data/2025-09-20-17-42-51-000000 \
  --overwrite
```

默认输出目录为：`<episode_path>/readable_exports`。  
也可自定义输出目录：

```bash
python /home/djy/EgoVerse/egomimic/scripts/export_readable_exports.py \
  /home/djy/EgoVerse/data/2025-09-20-17-42-51-000000 \
  --output-dir /home/djy/EgoVerse/data/2025-09-20-17-42-51-000000/readable_exports_new
```

## 输出文件位置

对上述 `file-000.parquet`，脚本会额外写入（示例）：

- `.../file-000.first_episode.json`（与 `file-000.parquet` 同目录）
- `meta/tasks.parquet` 则会生成 `.../tasks.first_episode.json`

## 常见问题

1. **`Expected a .parquet file`**：请把路径写到具体 `.parquet` 文件，不要只写到 `egoverse_lerobot/`。  
2. **`Unsupported parquet path`**：文件名或路径需符合上述三类约定；若数据集有多个 `chunk-*` / `file-*`，请把路径里的编号改成实际存在的文件。  
