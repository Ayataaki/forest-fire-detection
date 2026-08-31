# 🔥 Forest Fire Detection & Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-ff4b4b?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**A full-stack AI system for wildfire detection and fire intensity prediction using satellite imagery and NASA FIRMS data.**

![App Screenshot](https://raw.githubusercontent.com/mtgfz/forest-fire-prediction/main/assets/app1.png)


[🚀 Live App](https://forest-fire-prediction-ptjmpe27xqngkm47jdsq4h.streamlit.app) · [📓 Kaggle Notebook](https://www.kaggle.com/code/fatimaezzahraelemt/notebookcb8a562633) · [📊 Dataset 1](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) · [📊 Dataset 2](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)


</div>

---

## 📌 Project Overview

This project combines **deep learning image classification** with **tabular machine learning** to tackle wildfire detection from two angles:

| Pipeline | Input | Models | Output |
|---|---|---|---|
| 🛰️ Image Classification | Satellite images | EfficientNet-B2 + GeM | Fire / No Fire |
| 📡 Intensity Prediction | NASA FIRMS CSV | XGBoost, KNN, SVM, Decision Tree | Low / Medium / High intensity |

The two pipelines complement each other: the image model detects **whether** a fire exists, while the tabular models predict **how intense** it is.

---

## 🎯 Results

### Deep Learning — EfficientNet-B2

| Metric | Value |
|---|---|
| Test Accuracy | **99.31%** |
| AUC-ROC | **0.9992** |
| F1 Score | **0.9931** |
| OOD AUC (RGB dataset) | 0.4153 *(expected — different domain)* |

### Classical ML — NASA FIRMS Tabular Data

| Model | Accuracy | Task |
|---|---|---|
| XGBoost | **77.0%** | Fire intensity classification |
| SVM | 71.2% | Fire intensity classification |
| Decision Tree | 71.2% | Fire intensity classification |
| KNN | 68.8% | Fire intensity classification |
| XGBoost Regressor | MAE 3.8 MW | FRP regression |

---

## 🏗️ Model Architecture

```
Input (224×224 RGB satellite image)
           ↓
EfficientNet-B2 backbone (pretrained on ImageNet)
           ↓
Generalised Mean Pooling — GeM (p ≈ 3.0)
           ↓
BatchNorm1d → Dropout(0.3) → Linear(1408 → 512) → SiLU
           ↓
BatchNorm1d → Dropout(0.15) → Linear(512 → 2)
           ↓
Softmax → P(nowildfire),  P(wildfire)
```

**Why EfficientNet-B2 + GeM?**
- EfficientNet scales depth, width and resolution jointly → better features per FLOP than ResNet
- GeM pooling is more robust to outlier activations than standard average pooling, improving fine-grained visual retrieval

---

## 📦 Datasets

| Dataset | Source | Size | Role |
|---|---|---|---|
| Wildfire Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) | 30,250 satellite images | Primary training corpus |
| The Wildfire Dataset | [Kaggle](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) | 2,500 images | Out-of-distribution evaluation |
| NASA FIRMS VIIRS | [NASA API](https://firms.modaps.eosdis.nasa.gov/api/) | 263,831 records (5 days) | Tabular fire intensity model |

---

## 🚀 Live Application

The app is deployed on Streamlit Cloud and has 4 tabs:

- **🗺️ Live Fire Map** — Interactive Folium map showing fire hotspots by region (Morocco, California, Canada, etc.)
- **📷 Image Analysis** — Upload a satellite image → EfficientNet-B2 predicts fire probability with a gauge chart
- **📊 FIRMS Risk Prediction** — Enter lat/lon/brightness/FRP → all 4 ML models predict Low/Medium/High intensity
- **📈 Model Performance** — Training curves, AUC-ROC history, model comparison charts

👉 **[Open the app](https://forest-fire-prediction-ptjmpe27xqngkm47jdsq4h.streamlit.app)**

---

## 🗂️ Repository Structure

```
forest-fire-prediction/
├── src/
│   ├── dashboard/
│   │   └── app.py              ← Streamlit app (all 4 tabs)
│   ├── models/
│   │   ├── cnn_classifier.py
│   │   ├── resnet_transfer.py
│   │   └── lstm_timeseries.py
│   ├── ingestion/              ← Data download scripts
│   ├── processing/             ← Preprocessing pipeline
│   └── training/               ← Training utilities
├── models/                     ← Saved model weights & encoders
│   ├── xgboost_classifier.json
│   ├── xgboost_frp.json
│   ├── knn_firms.pkl
│   ├── svm_firms.pkl
│   ├── decision_tree_firms.pkl
│   └── label_encoder.pkl
├── best_model.pth              ← EfficientNet-B2 best checkpoint
├── notebooks/                  ← Kaggle training notebook
├── data/
│   └── training_history.csv   ← Epoch-by-epoch metrics
├── assets/                     ← Evaluation plots & visualisations
├── requirements.txt
└── README.md
```

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Model | EfficientNet-B2 |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| LR (head) | 3e-4 |
| LR (backbone) | 3e-5 *(discriminative LR)* |
| Scheduler | Cosine annealing + warm restart |
| Regularisation | Mixup (α=0.4) + Label smoothing (0.1) |
| Augmentation | RandomFlip, Rotation ±20°, ColorJitter, GaussianBlur |
| Epochs | 21 *(early stopping, patience=6)* |
| GPU | Kaggle P100 |

---

## 🔬 Key Findings

- **Val accuracy jumped from 94% → 99%+ in just 3 epochs** — EfficientNet's ImageNet pretraining transfers extremely well to satellite imagery
- **Val loss consistently lower than train loss** — expected behaviour with Mixup + label smoothing, which artificially inflates training loss
- **OOD degradation (AUC 0.41 on RGB photos) is expected and documented** — the model learned satellite-specific features (burn scars, smoke plumes from above) that don't transfer to ground-level photography
- **Grad-CAM confirms the model attends to correct regions** — activation maps highlight smoke and burned vegetation, not roads or water

---

## 🛠️ Run Locally

```bash
# Clone the repo
git clone https://github.com/mtgfz/forest-fire-prediction.git
cd forest-fire-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/dashboard/app.py
```

The app opens at `http://localhost:8501`

> **Note:** `best_model.pth` must be in the repo root. Download it from the [Kaggle notebook output](https://www.kaggle.com/code/fatimaezzahraelemt/notebookcb8a562633) if needed.

---

## 👥 Team

Aya Taki & Fatima-Ezzahrae Lemtougui

*Applied AI Project 2025-2026*

---

## 📄 License

MIT License — feel free to use, modify and share with attribution.
