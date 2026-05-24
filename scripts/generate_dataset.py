"""Generate the synthetic PCB training or validation dataset.

Usage:
    python scripts/generate_dataset.py --split train
    python scripts/generate_dataset.py --split val --config path/to/config.json

Reads all paths and generation parameters from ``config.json``.
Runs both generation phases in sequence:

  Phase 1: compose full-size images from templates → ``save_dir/``
  Phase 2: extract and resize random square crops   → ``crop_save_dir/``

See ``config.json`` (``generator.train_data`` / ``generator.val_data``) for
the full parameter reference.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aoi_pcb_ssd.config_loader import Config
from aoi_pcb_ssd.data.dataset_generator import PCBDatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PCB dataset.")
    parser.add_argument("--config", default="config.json", help="Path to config.json.")
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="Which data split to generate.")
    args = parser.parse_args()

    cfg = Config(args.config)
    split_cfg = cfg.generator.train_data if args.split == "train" else cfg.generator.val_data

    generator = PCBDatasetGenerator(
        templates_dir=cfg.generator.templates_dir,
        save_dir=split_cfg.save_dir,
        crop_save_dir=split_cfg.crop_save_dir,
        rotation_range=split_cfg.rotation_range,
        img_size=split_cfg.img_size,
        dataset_size=split_cfg.dataset_size,
        placement_offset_x=split_cfg.placement_offset_x,
        placement_offset_y=split_cfg.placement_offset_y,
        seed=split_cfg.seed,
    )

    print(f"Generating {args.split} dataset ({split_cfg.dataset_size} images)...")
    generator.generate(version="uncropped")

    print(f"Generating {args.split} crops ({split_cfg.dataset_size} crops)...")
    generator.generate(version="cropped")

    print("Done.")


if __name__ == "__main__":
    main()
