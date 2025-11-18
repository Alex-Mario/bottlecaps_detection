# Bottle Caps Detection 🍾

Real-time bottle-cap detection and color classification (light blue / dark blue / others) using **YOLOv8n**.
Includes a reproducible ML pipeline, CLI (`bsort`), Docker, and CI templates.

## Table of Contents

 1. [Project Summary](#project-summary)

 2. [Key Features](#key-features)

 3. [Dataset](#dataset)

 4. [Modeling Approach](#modeling-approach)

 5. [Results Summary & Limitations](#results-summary--limitations)

 6. [Repo Structure](#repo-structure)

 7. [Installation](#installation)

 8. [Configuration](#configuration)

 9. [Usage](#usage)

10. [Docker](#docker)

11. [CI/CD](#cicd-pipeline)

12. [License](#license)

## Project Summary

This project aims to build a real-time computer vision model to detect bottle caps and classify them into:

* light blue

* dark blue

* others

The model must run on Raspberry Pi 5 at 5–10 ms/frame, so a small model (YOLOv8n) is used.
A complete ML engineering pipeline is provided:

* Model training

* Experiment tracking with Weights & Biases

* Python CLI tool (`bsort`)

* Dockerized environment

* Automated linting, formatting, testing & Docker build

## Key Features

* ✅ **YOLOv8n-based** object detector

* ✅ **Full reproducible ML pipeline**

* ✅ **Public Weights & Biases** experiment tracking

* ✅ **Configurable** training & inference via `settings.yaml`

* ✅ **Python CLI (`bsort`)**

* ✅ **Dockerfile** for deployment

* ✅ **GitHub Actions CI/CD**:

  * `pylint`

  * `black`

  * `isort`

  * `pytest`

  * `Docker image build`

## Dataset

**Dataset Characteristics:**

* **12 images total** (Extremely small dataset for demonstration)

* Labels provided in YOLO format

* Manual color relabeling required for:

  * Light Blue

  * Dark Blue

  * Others

* **Class imbalance:**

  * "Others" dominates

  * Light Blue & Dark Blue contain only 1–2 samples each

**Train/val/test split:**

| Split | Images | 
 | ----- | ----- | 
| Train | 30 (w/ Aug) | 
| Valid | 1 | 
| Test | 1 | 

*Due to the extremely small dataset, results primarily serve demonstration purposes.*

## Modeling Approach

* **Base model:** YOLOv8n (lightest & fastest)

* **Input size:** 224 × 224

* **Optimizer:** Adam

* **Learning rate:** 1e-4

* **Epochs:** 100

* **Augmentation:**

  * Flip (H/V)

  * Rotation ±15°

  * Exposure & brightness changes

  * Blur

* **W&B logging** enabled

Training command example:

bsort train --config configs/settings.yaml

## Results Summary & Limitations

Because validation and test sets contain only one image, numerical metrics like mAP/precision/recall are not statistically meaningful.

**Qualitative results:**

* Model successfully detects bottle caps

* "Others" is predicted reliably

* Light Blue & Dark Blue classification is inconsistent due to limited samples

**Limitations:**

* **Tiny dataset** → heavy overfitting

* **Severe class imbalance** → model biased toward “Others”

* **Stretch resizing** may distort object shapes

*A larger and more balanced dataset is necessary for real-world performance.*

**Inference speed (i5 12400f):**
**6–8 ms/frame** → meets Raspberry Pi 5 requirement.

## Repo Structure

```
.
├── .github/workflows/    # CI/CD (ci.yaml)
├── src/bsort/                # Source code for the CLI and Logic
├── configs/              # Configuration files (settings.yaml)
├── data/                 # Dataset folder
├── models/               # Saved models
├── tests/                # Unit tests
├── Dockerfile            # Docker configuration
└── README.md
```


## Installation

**1. Clone the repo**

git clone https://github.com/Alex-Mario/bottlecaps_detection.git cd bottlecaps_detection


**2. Install dependencies**

Install package (standard)
pip install .

Or install in editable mode (for development)
pip install -e .


## Configuration

All parameters are stored in: `configs/settings.yaml`

**Example:**
```
dataset_path: "data/"
epochs: 100
batch_size: 16
learning_rate: 0.0001
imgsz: 224
optimizer: "Adam"
dropout: 0.15
patience: 20
wandb_project: "Bottle-Caps-Detection"
run_name: "run1"
```

## Usage

List available commands:

`bsort --help`


### Training

`bsort train --config configs/settings.yaml`


**Training outputs:**

* W&B logs

* Model weights under `models/`

* Convert the model to ONNX format

### Inference

bsort infer --config configs/settings.yaml --image sample.jpg


The model will:

1. Load the trained YOLOv8n weights

2. Run detection

3. Draw boxes + predicted color class

4. Show the result

## Docker

**Build image:**

docker build -t bsort .


**Run container:**

`docker run -it bsort python -m bsort --help`


## CI/CD Pipeline

GitHub Action (`.github/workflows/ci.yaml`) includes:

* `pylint` → code quality check

* `black` → format enforcement

* `isort` → import sorting

* `pytest` → run unit tests

* `Build Docker image automatically`

Pipeline runs on every:

* `git push`

* `pull request`

## License

This project is licensed under the MIT License.
