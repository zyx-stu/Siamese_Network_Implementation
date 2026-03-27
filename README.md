# Siamese_Network_Implementation

Basic implementation of Siamese Network with EfficientNet with Person-Re-Id-Dataset

🔗 Siamese Network — Person Re-Identification

A production-ready Siamese Network (APN — Anchor-Positive-Negative) for Person Re-Identification, built on top of a pretrained EfficientNet-B0 backbone. This repo demonstrates three high-impact improvements over a standard baseline:

1. ImageNet Normalization — unlocks the full power of pretrained weights
2. Data Augmentation — prevents overfitting on small person-identity datasets
3. Vectorized Inference — 10–100× faster gallery search via matrix operations

---

## 🚀 What Is a Siamese Network?

A Siamese Network learns to compare images by mapping them into an embedding space where:

* Similar images (same person) → embeddings are close
* Different images (different persons) → embeddings are far apart

It uses Triplet Loss on (Anchor, Positive, Negative) image groups:

```
Anchor ──►┐
           ├──► EfficientNet Encoder ──► Embedding ──► Triplet Loss
Positive ─►┘                                              ▲
                                                          │
Negative ──────────────────────────────────────────────────
```

---

## Dataset

Uses the Market-1501 Person Re-Identification dataset from Kaggle.

```
Person-Re-Id-Dataset/
├── train/
│   ├── person_001_cam1.jpg
│   ├── person_001_cam2.jpg
│   └── ...
└── train.csv   ← triplet pairs: Anchor | Positive | Negative
```

### CSV Preview (as shown in notebook)

| Anchor                  | Positive                | Negative                |
| ----------------------- | ----------------------- | ----------------------- |
| 0001_c1s1_001051_00.jpg | 0001_c2s1_001051_00.jpg | 0002_c1s1_000451_00.jpg |
| ...                     | ...                     | ...                     |

---

## Architecture

```
Input Images (3 streams: Anchor, Positive, Negative)
        │
        ▼
┌──────────────────────────────────────┐
│  EfficientNet-B0 (pretrained)        │
│  + Custom Linear Head: 1280 → 512   │  ← shared weights across all 3 streams
└──────────────────────────────────────┘
        │
        ▼
  512-dim Embeddings
        │
        ▼
  Triplet Margin Loss
```

---

## ⚡ Key Improvements Over Baseline

### 1. 📐 ImageNet Normalization

| Baseline    | This Repo                                             |
| ----------- | ----------------------------------------------------- |
| Pixel range | [0.0, 1.0] (raw /255)                                 |
| mean/std    | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

**Effect**

* Baseline → Misaligned inputs → degraded features
* This Repo → Correct alignment → pretrained weights work as intended

**Why it matters:**
EfficientNet-B0 was trained on ImageNet with specific pixel statistics. Without matching those stats, the first layers receive out-of-distribution inputs — effectively wasting the pretrained knowledge.

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std =[0.229, 0.224, 0.225])
```

---

### 2. Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),      
    transforms.RandomRotation(degrees=10),       
    transforms.ColorJitter(brightness=0.3, ...),  
    transforms.ToTensor(),
    transforms.Normalize(...)                      
])
```

**Why it matters:**
Triplet datasets overfit easily — the model may memorize exact pixel patterns rather than learning view-invariant identity features. Augmentation forces generalization.

---

### 3. ⚡ Vectorized Inference

| Baseline (Original) | This Repo (Optimized)                                   |
| ------------------- | ------------------------------------------------------- |
| Approach            | for i in range(N): euclidean_dist(query, gallery[i])    |
| Optimized           | diff = query - gallery then sqrt((diff**2).sum(axis=1)) |
| Complexity          | O(N) sequential Python iterations                       |
| Optimized           | O(1) single NumPy broadcast op                          |
| Speed (1000 images) | ~2–5 seconds                                            |
| Optimized           | ~20ms                                                   |

Also: database is built in batches (not image-by-image) via `get_encoding_csv()`.

---

## 📁 Project Structure

```
siamese-person-reid/
├── siamese_network.py
├── database.csv
├── best_model.pt
├── sample_triplet.png
├── augmented_triplet.png
├── training_curves.png
├── top5_matches.png
└── README.md
```

---

## ⚙️ Quickstart

### 1. Install Dependencies

```bash
pip install torch torchvision timm scikit-image scikit-learn pandas numpy matplotlib tqdm
```

### 2. Get the Dataset

```bash
# Kaggle dataset
# https://www.kaggle.com/pengcw1/market-1501

# Or use the lightweight training split:
git clone https://github.com/parth1620/Person-Re-Id-Dataset
```

### 3. Run the Full Pipeline

```bash
python siamese_network.py
```

This will:

* 📋 Print the CSV head (first 5 rows)
* 🖼️ Show sample raw triplets
* 🎨 Show augmented triplets
* 🏋️ Train for 15 epochs with progress bars
* 📉 Plot training/validation loss curves
* 🗄️ Build database.csv (image name + embeddings)
* 🔍 Run vectorized query and show top-5 matches

---

## 📊 Outputs Explained

### database.csv — Encoding Database

After training, every anchor image is encoded into a 512-dimensional vector and saved:

| Anchor        | 0      | 1      | 2     | ... | 511   |
| ------------- | ------ | ------ | ----- | --- | ----- |
| 0001_c1s1.jpg | 0.231  | -0.045 | 0.812 | ... | 0.003 |
| 0002_c1s1.jpg | -0.112 | 0.334  | 0.091 | ... | 0.441 |

```python
df_enc.head()
print(df_enc.shape)
```

---

## 🔧 Hyperparameter Guide

| Parameter            | Default | Notes                           |
| -------------------- | ------- | ------------------------------- |
| BATCH_SIZE           | 32      | Larger → more stable gradients  |
| LR                   | 0.001   | Lower to 5e-4 for finetuning    |
| EPOCHS               | 15      | 20–30 for harder datasets       |
| emb_size             | 512     | Try 256 for smaller datasets    |
| margin (TripletLoss) | 1.0     | Increase if embeddings collapse |

---

## 📚 References

* FaceNet: Triplet Loss for Face Recognition — Schroff et al., 2015
* EfficientNet: Rethinking Model Scaling for CNNs — Tan & Le, 2019
* Market-1501 Dataset — Zheng et al., 2015
* Person Re-Id Dataset (Training Split)

---

## 💡 Tip

Open this notebook in Google Colab for free GPU access. The full pipeline runs in ~15 minutes on a T4 GPU.
