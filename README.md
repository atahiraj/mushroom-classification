# Mushroom Classification

## About

This repository contains an image-classification project for mushroom
photographs. It provides dataset splitting and preprocessing utilities,
PyTorch data pipelines, image augmentation, and transfer-learning experiments
with ResNet architectures. The project was developed for the University of
Liège INFO8010-1 course.

The images come from the
[2018 FGVCx Fungi Classification Challenge on Kaggle](https://www.kaggle.com/c/fungi-challenge-fgvc-2018).
The benchmark contains 85,578 training images and 4,182 validation images from
1,394 wild mushroom species documented by the Danish Svampe Atlas. The
[official challenge repository](https://github.com/visipedia/fgvcx_fungi_comp)
provides the dataset description, annotations, checksums, and download links.

The code is an archival experiment built around older PyTorch and TorchVision
APIs. Dependency versions were not recorded, so modern releases may require
small compatibility changes.

## Setup

Create and activate a Python environment, then install the imported runtime
dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision scikit-learn matplotlib numpy scikit-image pillow
```

Download the training and validation images from the
[Kaggle competition data page](https://www.kaggle.com/c/fungi-challenge-fgvc-2018/data)
or the official challenge repository. Accept the dataset terms before
downloading, then place the extracted `images/` directory under `data/`. The
source archive uses one directory per species, including a numeric taxonomy
identifier:

```text
data/
└── images/
    ├── 10056_Agaricus_arvensis/
    │   └── photo.jpg
    └── 10057_Agaricus_augustus/
        └── photo.jpg
```

The split operation creates the `data/train/` and `data/test/` directories
consumed by the training pipeline.

## Usage

Split the source dataset into training and test directories:

```sh
python train.py --path data --split true
```

Compute per-channel statistics for an existing split:

```sh
python train.py --path data --stats true
```

Run the model pipeline with a ResNet architecture:

```sh
python train.py --path data --train true --model resnet18 --gpu n
```

Supported model names include `resnet18`, `resnet34`, `resnet50`, `resnet101`,
and `resnet152`. Set `--gpu y` to use CUDA when available. The current archival
pipeline expects a `tut5-model.pt` checkpoint when it reaches evaluation.

## License

This project's source code is licensed under the MIT License. See
[LICENSE](LICENSE) for the complete terms. The image dataset is not included
under the MIT License: its challenge terms restrict it to non-commercial
research and educational use and prohibit redistribution of the images.
