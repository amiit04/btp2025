# Face De-Identification & Image Processing Pipeline


This repository contains the complete pipeline for image dataset preparation, processing (with and without smoothing), and evaluation using metrics such as SSIM and cosine similarity and FID score. It is designed for experiments related to face de-identification and image-quality analysis.

---

## 🔧 Features

* **Dataset Preparation**
  Automatic renaming (001.jpg → 100.jpg), folder organization, augmentation, and dataset generation.

* **Processing Pipeline**
  Runs your custom image-processing / de-identification logic, producing:

  * Output images **with smoothing**
  * Output images **without smoothing**

* **Evaluation Module**
  Computes SSIM, cosine similarity, and FID metrics between two image sets.
---

## Repository Structure

```
.
├── imagedata_generator.py   # Creates dataset, renaming, preprocessing
├── work.py                  # Main processing pipeline (with/without smoothing)
├── evaluate.py              # Evaluation metrics (SSIM, cosine similarity, FID)
└── README.md
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install numpy opencv-python pillow scikit-image tqdm
# Install torch/torchvision if using embedding-based metrics
```

---

## Usage

### 1. **Generate / Prepare Images**

```bash
python imagedata_generator.py
```

This step typically:

* renames images to `001.jpg`, `002.jpg`, …
* prepares clean dataset folder
* applies any preprocessing steps configured inside the script

---

### 2. **Run Processing Pipeline**

Creates *two* separate output folders:

* `*_with_smoothing.jpg`
* `*_without_smoothing.jpg`

Example:

```bash
python work.py --src_dir ./images --out_without ./outputs_without_smoothing \
  --out_with ./outputs_with_smoothing --start 1 --end 100 --padding 3 --min_area 1000

```

---

### 3. **Run Evaluation (SSIM, Cosine Similarity, etc.)**

```bash
python evaluate.py --orig_dir ./input --deid_dirs ./no_smooth,./with_smooth --out my_eval --iou_thresh 0.20
```

Outputs include:

* Per-image metric values
* Average SSIM & cosine similarity
* Optional logs/visual outputs (depending on script)
---