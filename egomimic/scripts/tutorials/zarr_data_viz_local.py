from __future__ import annotations

import sys
from pathlib import Path

import imageio
import imageio_ffmpeg
import mediapy as mpy
import numpy as np
import torch

# Allow running this script directly without setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.rldb.embodiment.human import Aria  # noqa: E402
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset  # noqa: E402

# Ensure mediapy can find ffmpeg in this environment.
mpy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

# Local episode to visualize.
EPISODE_PATH = Path("/home/djy/EgoVerse/data/mecka/692e6de65aae241ad236e7e3/")
OUTPUT_MP4 = Path("/home/djy/EgoVerse/sample_traj_local1.mp4")
MAX_FRAMES = 3000
FPS = 30


def main() -> None:
    key_map = Aria.get_keymap(keymap_mode="cartesian")
    transform_list = Aria.get_transform_list(mode="cartesian")

    dataset = ZarrDataset(
        Episode_path=EPISODE_PATH,
        key_map=key_map,
        transform_list=transform_list,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    images = []
    for i, batch in enumerate(loader):
        vis = Aria.viz_transformed_batch(batch, mode="traj")
        images.append(vis)
        if i >= MAX_FRAMES:
            break

    if not images:
        raise ValueError(f"No frames produced from local episode: {EPISODE_PATH}")

    frames = np.stack(images, axis=0)
    writer = imageio.get_writer(str(OUTPUT_MP4), fps=FPS)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved visualization to: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
