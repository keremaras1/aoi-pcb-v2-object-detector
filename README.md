# Automated Optical Inspection for PCBA Quality Control

Deep learning-based AOI system for detecting IC component defects in printed circuit board assembly lines, developed as a case study for Point of Sale device production. Built on a custom Single Shot MultiBox Detector (SSD) architecture that simultaneously classifies IC components and localises their corner points.

## Architecture

The custom model uses a 6-block CNN feature extractor feeding a three-branch detection head:

- **Classification branch** — softmax confidence scores per grid cell (IC vs background)
- **Localisation branch** — 4 × (x, y) corner offset pairs per grid cell
- **Grid centre branch** — anchor point coordinates for each feature map cell

The first two convolutional blocks use (5×5) kernels; the remaining four use (3×3). The feature extractor produces 8×8 feature maps fed directly to all three branches.

A second architecture using MobileNetV2 as a frozen or fine-tuned backbone is also included.

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the Dataset

```bash
# 1. Mark IC locations on a PCB template image
python implementations/realistic_data_generator/template_generator/template_generator.py

# 2. Generate the synthetic training dataset
python implementations/realistic_data_generator/dataset_generator/dataset_generator.py
```

Training and inference are run through the notebooks in `implementations/ssd_models/custom_architecture/`.

## License

To be specified.
