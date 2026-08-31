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

Stage 1 binary anomaly detection (Normal vs. Anomaly) on the held-out test set:

| Metric | Normal (0) | Anomaly (1–4) |
|---|---|---|
| Precision | 0.91 | 0.39 |
| Recall | 0.80 | 0.61 |
| F1-score | 0.85 | 0.47 |

Overall accuracy: **77%** | Macro F1: **0.66**

```
Confusion Matrix
[[14494  3624]
 [ 1474  2300]]
```

The reconstruction-error threshold trades off precision and recall — lowering
it catches more true anomalies (better recall) at the cost of more false
positives on normal beats. See the notebook for the full sweep and Stage 2
sub-type classification report.

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
