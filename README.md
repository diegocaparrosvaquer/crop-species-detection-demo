# 🌾 Crop Species Detection Demo

A deep-learning demo for hierarchical crop species classification using **DINOv2 ViT-B/14**.

The model uses a two-stage classification pipeline:

1. **Stage 1:** Determines whether an image contains cropland or no cropland.
2. **Stage 2:** Classifies cropland images into one of **9 crop species/classes**.

The demo is designed to run directly in **Google Colab** using the trained model checkpoints stored in this repository with Git LFS.

---

## 🌱 Supported Classes

### Stage 1

- `no cropland`
- `cropland`

### Stage 2

- `banana`
- `maize`
- `millets`
- `rapeseed`
- `soya`
- `sorghum`
- `sunflower`
- `vineyard`
- `wheat type crop`

---

## 🧠 Model Architecture

The classifier uses:

**DINOv2 ViT-B/14 → Classification Head**

The DINOv2 backbone provides a 768-dimensional image representation, followed by a multilayer classification head.

The trained models are:

```text
models/
├── stage1_best.pt
└── stage2_best.pt
```

The models are stored using Git Large File Storage (Git LFS) because each checkpoint is several hundred MB.

## 📊 Test Performance

The complete hierarchical pipeline achieved:

| Metric | Score |
|---|---|
| Accuracy | 93.72% |
| Balanced Accuracy | 93.34% |
| Macro F1 | 93.18% |
| Weighted F1 | 93.70% |

### Stage 1 — Cropland Detection

| Metric | Score |
|---|---|
| Accuracy | 95.19% |
| Balanced Accuracy | 95.49% |
| Macro F1 | 94.75% |

### Stage 2 — Crop Classification

Evaluated on the 4,414 true cropland test images:

| Metric | Score |
|---|---|
| Accuracy | 97.10% |
| Balanced Accuracy | 95.88% |
| Macro F1 | 95.04% |
| Weighted F1 | 97.12% |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No cropland | 0.90 | 0.96 | 0.93 |
| Banana | 0.97 | 0.98 | 0.98 |
| Maize | 0.95 | 0.93 | 0.94 |
| Millets | 0.83 | 0.86 | 0.85 |
| Rapeseed | 0.94 | 0.98 | 0.96 |
| Soya | 0.85 | 0.92 | 0.89 |
| Sorghum | 0.94 | 0.89 | 0.91 |
| Sunflower | 0.98 | 0.98 | 0.98 |
| Vineyard | 0.99 | 0.97 | 0.98 |
| Wheat type crop | 0.96 | 0.86 | 0.91 |

## 🚀 Running the Demo in Google Colab

The easiest way to use the model is through Google Colab.

### 1. Open the Colab notebook

Open the demo notebook in Google Colab.

The notebook will:

- Clone this repository.
- Install the required Python packages.
- Download the Git LFS model checkpoints.
- Load the DINOv2 models.
- Allow you to upload an image.
- Run Stage 1.
- Run Stage 2 when the image is classified as cropland.
- Display the final prediction and confidence scores.

## 📦 Repository Structure

```text
crop-species-detection-demo/
│
├── models/
│   ├── stage1_best.pt
│   └── stage2_best.pt
│
├── model.py
├── requirements.txt
├── README.md
└── demo.ipynb
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/diegocaparrosvaquer/crop-species-detection-demo.git
cd crop-species-detection-demo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If using Google Colab, the notebook performs these steps automatically.

## 🔬 Inference Pipeline

For every input image:

```text
                 Input Image
                      │
                      ▼
              ┌───────────────┐
              │   DINOv2      │
              │   Stage 1     │
              └───────┬───────┘
                      │
             ┌────────┴────────┐
             │                 │
       No Cropland          Cropland
             │                 │
             ▼                 ▼
      Final Prediction   ┌───────────────┐
                         │   DINOv2      │
                         │   Stage 2     │
                         └───────┬───────┘
                                 │
                                 ▼
                         Crop Species
```

Stage 2 is only used when Stage 1 predicts:

```
cropland
```

If Stage 1 predicts:

```
no cropland
```

the final prediction is immediately:

```
no cropland
```

## 🖼️ Example Usage

The inference code can be used with a local image:

```python
from PIL import Image

image = Image.open("example.jpg").convert("RGB")

prediction = predict(image)

print(prediction)
```

The demo notebook provides a complete implementation including preprocessing, model loading, inference, and visualization.

## 💾 Git LFS

The model checkpoints are tracked using Git LFS.

To clone the repository with the model files:

```bash
git lfs install
git clone https://github.com/diegocaparrosvaquer/crop-species-detection-demo.git
```

If the repository has already been cloned without downloading the LFS files:

```bash
git lfs pull
```

You can verify the checkpoints with:

```bash
git lfs ls-files
```

You should see:

```text
models/stage1_best.pt
models/stage2_best.pt
```

## 🖥️ Hardware

The models were trained and evaluated using an NVIDIA L40S GPU.

The demo can run on a GPU-enabled Google Colab runtime.

For best inference performance, enable:

```
Runtime → Change runtime type → T4 GPU
```

A CPU runtime can also be used, but inference will be considerably slower.

## 🔧 Input Preprocessing

Images are converted to RGB and resized to:

```
224 × 224
```

The same ImageNet normalization used during model evaluation is applied:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## 📈 Why Hierarchical Classification?

Instead of directly predicting all classes with one classifier, the system separates the task into two decisions:

**Stage 1**
Is this cropland?

**Stage 2**
If it is cropland, which crop is it?

This makes it possible to explicitly separate non-cropland imagery from crop-specific classification.

It also allows Stage 2 to specialize specifically in distinguishing between crop species.

## ⚠️ Important Notes

The reported performance is based on the project's held-out test dataset.

Performance on images from completely different geographic regions, cameras, seasons, lighting conditions, or acquisition systems may differ.

In particular, the classes with fewer test examples should be interpreted carefully:

- Millets: 156 test images
- Soya: 89 test images

The overall accuracy therefore should be considered together with balanced accuracy, macro F1, and the per-class metrics.

## 📚 Technologies

- Python
- PyTorch
- Torchvision
- DINOv2
- NumPy
- Pandas
- scikit-learn
- Pillow
- Google Colab
- Git LFS

## 👤 Author

**Diego Caparros Vaquer**

GitHub:

https://github.com/diegocaparrosvaquer

Repository:

https://github.com/diegocaparrosvaquer/crop-species-detection-demo

## 📄 License

This repository is intended as a demonstration of the trained crop-species classification models and inference pipeline.

Please check the licensing terms of the underlying DINOv2 model and datasets before using the system commercially or redistributing derived components.
