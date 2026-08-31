# ECG Heartbeat Anomaly Detection (Two-Stage Deep Learning)

A two-stage deep learning pipeline for detecting and characterizing abnormal
heartbeats in ECG signals, built on the [MIT-BIH Arrhythmia dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

## Overview

Most heartbeats are normal, and abnormal ones are rare and diverse — a classic
anomaly detection setting. Instead of training a single multi-class classifier,
this project splits the problem into two stages:

1. **Stage 1 — Unsupervised anomaly detection (1D-CNN Autoencoder).**
   A convolutional autoencoder is trained *only* on normal heartbeats (Class 0).
   At inference time, any heartbeat the autoencoder struggles to reconstruct well
   (high reconstruction error) is flagged as a potential anomaly. This means the
   model never needs labeled anomalies to learn what "normal" looks like.

2. **Stage 2 — Supervised sub-type classification.**
   Heartbeats flagged as anomalies in Stage 1 are passed to a second classifier
   trained on the four abnormal heartbeat classes (1–4), so the pipeline doesn't
   just say "abnormal" — it predicts *which kind* of abnormality it likely is.

## Dataset

- **Source:** [MIT-BIH Arrhythmia Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- 187 timesteps per heartbeat, 5 classes: `0` (Normal) and `1–4` (different arrhythmia types)
- Heavily imbalanced: ~83% of the training set is Class 0

| Class | Count (train) |
|-------|---------------|
| 0 (Normal) | 72,471 |
| 1 | 2,223 |
| 2 | 5,788 |
| 3 | 641 |
| 4 | 6,431 |

## Model Architecture

**Stage 1 — 1D-CNN Autoencoder**
```
Input (187, 1)
 → Conv1D(32) → MaxPool
 → Conv1D(16) → MaxPool         [bottleneck]
 → Conv1D(16) → UpSampling
 → Conv1D(32) → UpSampling → Cropping
 → Conv1D(1)                    [reconstruction]
```
Trained with MSE loss on normal heartbeats only. Anomaly threshold is set from
a percentile of the reconstruction error on normal training data.

**Stage 2 — Anomaly sub-type classifier**
Trained on heartbeats from classes 1–4 only, with class weighting to handle the
imbalance between sub-types (e.g. Class 3 has ~10x fewer samples than Class 4).

## Results

**Stage 1 — Binary anomaly detection (Normal vs. Anomaly)** on the held-out test set:

| Metric | Normal (0) | Anomaly (1–4) |
|---|---|---|
| Precision | 0.99 | 0.98 |
| Recall | 1.00 | 0.95 |
| F1-score | 0.99 | 0.96 |

Overall accuracy: **99%** | Macro F1: **0.98**

**Stage 2 — Anomaly sub-type classification**, evaluated in isolation on true
anomalies only (i.e. assuming Stage 1 routed them correctly):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 1 | 0.97 | 0.95 | 0.96 | 556 |
| 2 | 0.98 | 0.95 | 0.96 | 1,448 |
| 3 | 0.66 | 0.96 | 0.78 | 162 |
| 4 | 0.99 | 0.99 | 0.99 | 1,608 |

Overall accuracy: **96%** | Macro F1: **0.92**

Class 3 (the smallest and most imbalanced class, ~641 training samples) has
noticeably lower precision despite high recall — the class-weighted loss
pushes the model to catch nearly all true Class 3 beats, at the cost of some
false positives from other classes.

**Complete end-to-end pipeline** — Stage 1 routes each beat, and only the ones
flagged as anomalies are passed to Stage 2, so errors in Stage 1 routing
propagate through:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Normal (0) | 0.99 | 1.00 | 0.99 | 18,118 |
| 1 | 0.88 | 0.79 | 0.83 | 556 |
| 2 | 0.98 | 0.93 | 0.95 | 1,448 |
| 3 | 0.67 | 0.81 | 0.74 | 162 |
| 4 | 0.99 | 0.98 | 0.99 | 1,608 |

Overall accuracy: **98%** | Macro F1: **0.90** | Weighted F1: **0.98**

The gap between Stage 2's isolated performance (96% acc) and the full
pipeline's performance (98% overall, but lower per-class recall on 1 and 3)
shows the compounding effect of Stage 1 misrouting a small number of true
anomalies as normal before they ever reach Stage 2.

## Project Structure

```
.
├── anomaly_2models.ipynb   # Main notebook: data prep, both models, evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone and install dependencies
```bash
git clone https://github.com/<your-username>/ecg-anomaly-detection.git
cd ecg-anomaly-detection
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials
This project downloads data via the Kaggle API. Set up your credentials
following the [official Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials)
(either a `~/.kaggle/kaggle.json` file or `KAGGLE_USERNAME` / `KAGGLE_KEY`
environment variables). **Never commit your API key to the repo.**

### 3. Run the notebook
```bash
jupyter notebook anomaly_2models.ipynb
```

## Future Work

- Try a Deep SVDD / latent-space distance approach as an alternative to
  reconstruction-error thresholding (scaffolded in the notebook)
- Compare against a One-Class SVM baseline (scaffolded in the notebook)
- Explore a residual 1D-CNN architecture for Stage 2
- Threshold tuning via precision-recall curve instead of a fixed percentile

## License

MIT — see [LICENSE](LICENSE).
