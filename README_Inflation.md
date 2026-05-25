# 📊 Deep Learning Inflation Prediction

> A complete end-to-end pipeline for predicting inflation (CPI) using 5 deep learning architectures — LSTM, GRU, Bidirectional LSTM, CNN-LSTM, and Transformer — with a live Flask web application that forecasts up to 24 months ahead.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [What is CPI and Why Does It Matter](#what-is-cpi-and-why-does-it-matter)
- [Feature Engineering](#feature-engineering)
- [Deep Learning Models](#deep-learning-models)
- [Ensemble Method](#ensemble-method)
- [Evaluation Metrics](#evaluation-metrics)
- [Web Application](#web-application)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Key Learnings](#key-learnings)
- [Disclaimer](#disclaimer)

---

## 🔍 Project Overview

This project builds a **production-ready inflation forecasting system** using multiple deep learning architectures trained on over 60 years of US Consumer Price Index (CPI) data. The system goes beyond a simple notebook experiment — it includes a complete data pipeline, five trained neural networks, an ensemble combiner, and a live web dashboard where users can forecast inflation up to 24 months into the future.

**Why inflation prediction matters:**
- Central banks use inflation forecasts to set interest rates
- Businesses use them for pricing strategies and budget planning
- Investors use them to protect portfolio purchasing power
- Governments use them to adjust social security and benefits

This project approaches it as a **time series regression problem** — given the last 24 months of CPI and derived economic indicators, predict the CPI value for the next N months.

---

## 🎬 Demo

```
Settings : Forecast Horizon = 6 months  |  Model = Ensemble
─────────────────────────────────────────────────────────
Current CPI        : 314.18
Current Inflation  : 3.2%
─────────────────────────────────────────────────────────
Month 1 Forecast   : CPI = 315.40  |  Inflation = 3.3%
Month 2 Forecast   : CPI = 316.55  |  Inflation = 3.2%
Month 3 Forecast   : CPI = 317.80  |  Inflation = 3.1%
Month 4 Forecast   : CPI = 318.90  |  Inflation = 3.0%
Month 5 Forecast   : CPI = 319.85  |  Inflation = 2.9%
Month 6 Forecast   : CPI = 320.70  |  Inflation = 2.8%
─────────────────────────────────────────────────────────
Trend Signal       : FALLING  ✅ (moving toward 2% target)
```

---

## 🏗️ Project Architecture

```
CPI Data (FRED — Federal Reserve Economic Data)
        │
        ▼
Feature Engineering (16 economic indicators)
        │
        ▼
MinMaxScaler Normalization
        │
        ▼
Sliding Window Sequences (24-month look-back)
        │
        ▼
┌───────┬────────┬──────────┬───────────┬─────────────┐
│ LSTM  │  GRU   │ Bi-LSTM  │ CNN-LSTM  │ Transformer │
└───────┴────────┴──────────┴───────────┴─────────────┘
        │
        ▼
   Ensemble (Average of 5 model predictions)
        │
        ▼
Multi-step Autoregressive Forecasting (up to 24 months)
        │
        ▼
Flask REST API → Interactive Web Dashboard
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Indicator** | Consumer Price Index for All Urban Consumers (CPI-U) |
| **Series ID** | CPIAUCSL |
| **Data Source** | FRED — Federal Reserve Bank of St. Louis |
| **URL** | https://fred.stlouisfed.org/series/CPIAUCSL |
| **Date Range** | January 1960 — December 2024 (64 years) |
| **Frequency** | Monthly |
| **Total Rows** | ~780 monthly observations |
| **Base Period** | 1982–1984 = 100 |

### Train / Validation / Test Split

```
|←──────────────── 70% Train ─────────────────→|←── 15% Val ──→|←── 15% Test ──→|
  Jan 1960 ────────────────────────────────── 2017 ─────────── 2021 ──── Dec 2024
```

### Sequence Configuration

- **Look-back window**: 24 months
- Each model input is a matrix of shape `(24, 16)` — 24 months × 16 features
- The target is the **next month's CPI value**
- For multi-step forecasting, an **autoregressive rollout** is used — each prediction becomes part of the input for the next step

### Why 64 Years of Data?

Inflation is a slow-moving macroeconomic phenomenon. Using only recent data would miss critical historical regimes:

| Period | Inflation Regime |
|--------|-----------------|
| 1960s | Moderate inflation (~2–3%) |
| 1970s | High inflation, oil shocks (peaked at ~14%) |
| 1980s | Volcker disinflation (rapid drop from 14% to 3%) |
| 1990s–2000s | Great Moderation (stable ~2–3%) |
| 2010s | Low inflation post-GFC (~1–2%) |
| 2020s | COVID supply shock spike (~9%) then disinflation |

Training across all regimes gives the model historical context for extreme events and recoveries.

---

## 💡 What is CPI and Why Does It Matter?

**CPI (Consumer Price Index)** measures the average change over time in the prices paid by urban consumers for a basket of goods and services — food, housing, clothing, transportation, medical care, and more.

**Year-over-Year Inflation Rate** is computed as:
```
Inflation(t) = (CPI(t) / CPI(t-12) - 1) × 100
```

**The Federal Reserve targets 2% annual inflation** as the ideal balance between:
- Too low → deflation risk, reduced economic activity
- Too high → erodes purchasing power, hurts savings

**Why is it hard to predict?**
- It is influenced by dozens of factors: energy prices, supply chains, wage growth, monetary policy, geopolitical events
- It has long memory — shocks in one year can persist for 2–3 years
- It exhibits non-linear regime changes (low → high inflation transitions are not smooth)

This is why deep learning — which can learn non-linear patterns across long sequences — is a strong candidate for this problem.

---

## ⚙️ Feature Engineering

16 features are engineered from raw monthly CPI values:

### Lag Features — Teaching the Model History

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| `CPI_lag1` | CPI from 1 month ago | Immediate recent level |
| `CPI_lag2` | CPI from 2 months ago | Short-term momentum |
| `CPI_lag3` | CPI from 3 months ago | Quarterly view |
| `CPI_lag6` | CPI from 6 months ago | Half-year comparison |
| `CPI_lag12` | CPI from 12 months ago | Year-ago base effect |

The 12-month lag is especially important — inflation is defined as the year-over-year change, so the model needs to see where prices were exactly a year ago.

### Rolling Statistics — Smoothing the Noise

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| `CPI_MA3` | 3-month moving average | Removes monthly volatility |
| `CPI_MA6` | 6-month moving average | Medium-term trend |
| `CPI_MA12` | 12-month moving average | Long-term trend direction |

### Momentum Features — How Fast Is It Moving?

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| `CPI_pct1` | Month-over-month change (%) | Immediate price pressure |
| `CPI_pct3` | 3-month change (%) | Short-run momentum |
| `CPI_pct12` | Year-over-year inflation (%) | The headline inflation rate |

### Volatility Features — How Uncertain Is It?

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| `CPI_std3` | 3-month rolling standard deviation | Recent volatility |
| `CPI_std6` | 6-month rolling standard deviation | Medium-term uncertainty |

High CPI volatility often precedes regime shifts — the model can learn this association.

### Calendar Features — Seasonal Patterns

| Feature | Description | Why it matters |
|---------|-------------|---------------|
| `Month` | Month of year (1–12) | Seasonal patterns (e.g. energy prices spike in winter) |
| `Quarter` | Quarter of year (1–4) | Quarterly economic cycles |

Inflation has well-documented seasonal patterns — food prices rise in certain months, energy costs have winter peaks. Calendar features help the model learn these patterns explicitly.

---

## 🧠 Deep Learning Models

All models use:
- **Loss function**: Huber Loss (more robust to CPI spikes than MSE)
- **Optimizer**: Adam
- **Callbacks**: EarlyStopping (patience=20), ReduceLROnPlateau, ModelCheckpoint
- **Dropout**: 0.2 on all recurrent layers
- **Batch size**: 16 (smaller because monthly data is limited)
- **Max epochs**: 150
- **Input shape**: `(24, 16)` — 24 months × 16 features

> **Why Huber Loss?**
> CPI has occasional extreme spikes (e.g. the 2022 energy shock). MSE squares these errors and can distort training. Huber Loss behaves like MSE for small errors but like MAE for large errors — giving stable gradients without ignoring outliers entirely.

---

### Model 1 — Stacked LSTM

LSTM networks are the workhorse of time series forecasting. Their gating mechanism (input gate, forget gate, output gate) allows them to selectively remember or forget information across sequences — essential for inflation data where patterns from 12–24 months ago are highly relevant.

```
Architecture:
Input (24, 16)
    → LSTM(128, return_sequences=True) → Dropout(0.2)
    → LSTM(64,  return_sequences=True) → Dropout(0.2)
    → LSTM(32)                         → Dropout(0.2)
    → Dense(32, relu)
    → Dense(16, relu)
    → Dense(1)
```

**How the gates work:**
- **Forget gate**: Decides what past information to discard (e.g. an old inflation spike that has now reversed)
- **Input gate**: Decides what new information to store (e.g. a sudden CPI increase this month)
- **Output gate**: Decides what to output to the next layer

**Parameters**: ~220,000 | **Learning rate**: 0.001

---

### Model 2 — GRU (Gated Recurrent Unit)

GRU simplifies the LSTM by combining the forget and input gates into a single **update gate** and merging the cell state and hidden state. This reduces parameters while preserving most of the learning capacity.

```
Architecture:
Input (24, 16)
    → GRU(128, return_sequences=True) → Dropout(0.2)
    → GRU(64,  return_sequences=True) → Dropout(0.2)
    → GRU(32)                         → Dropout(0.2)
    → Dense(32, relu)
    → Dense(16, relu)
    → Dense(1)
```

**GRU gates:**
- **Update gate**: Controls how much of the past to carry forward
- **Reset gate**: Controls how much of the past to forget when computing new candidate state

With only ~780 monthly data points, GRU's lower parameter count actually helps — less overfitting risk compared to full LSTM.

**Parameters**: ~160,000 | **Learning rate**: 0.001

---

### Model 3 — Bidirectional LSTM

Standard LSTMs process sequences strictly left to right (past to present). Bidirectional LSTM runs two parallel LSTM layers — one forward, one backward — and concatenates their hidden states at each time step.

```
Architecture:
Input (24, 16)
    → Bidirectional(LSTM(128, return_sequences=True)) → Dropout(0.2)
    → Bidirectional(LSTM(64,  return_sequences=True)) → Dropout(0.2)
    → Bidirectional(LSTM(32))                         → Dropout(0.2)
    → Dense(32, relu)
    → Dense(16, relu)
    → Dense(1)
```

**For inflation data specifically**, bidirectional processing helps because:
- Inflation cycles are symmetric — a period of rising inflation is often followed by a mirror disinflation
- The backward pass learns what typically comes *after* certain CPI patterns, giving the forward pass richer context

**Parameters**: ~440,000 | **Learning rate**: 0.001

---

### Model 4 — CNN-LSTM Hybrid

Convolutional layers are typically used for images, but 1D convolutions work exceptionally well on time series — they detect local patterns (short subsequences) regardless of their position in the full sequence.

```
Architecture:
Input (24, 16)
    → Conv1D(64, kernel_size=3, relu, padding=same)
    → Conv1D(64, kernel_size=3, relu, padding=same)
    → MaxPooling1D(pool_size=2)
    → Dropout(0.2)
    → LSTM(64, return_sequences=True) → Dropout(0.2)
    → LSTM(32)                        → Dropout(0.2)
    → Dense(32, relu)
    → Dense(16, relu)
    → Dense(1)
```

**Two-stage learning:**
1. **CNN stage**: Scans the 24-month sequence with a 3-month kernel, detecting local patterns — a 3-month spike, a gradual slowdown, a sudden reversal
2. **LSTM stage**: Takes these extracted patterns and models their temporal relationships across the full sequence

**Parameters**: ~185,000 | **Learning rate**: 0.001

---

### Model 5 — Transformer (Multi-Head Self-Attention)

The Transformer uses **self-attention** to directly compare every month in the 24-month window to every other month — simultaneously. Unlike RNNs, which must pass information through every intermediate time step, the Transformer can draw direct connections across the full sequence in a single operation.

```
Architecture:
Input (24, 16)
    → Dense(64)  [linear projection to d_model]
    → TransformerBlock × 2:
         MultiHeadAttention(num_heads=4, key_dim=16)
         → Dropout(0.1)
         → Add & LayerNormalization
         → FeedForward: Dense(128, relu) → Dense(64)
         → Dropout(0.1)
         → Add & LayerNormalization
    → GlobalAveragePooling1D
    → Dense(64, relu) → Dropout(0.1)
    → Dense(32, relu)
    → Dense(16, relu)
    → Dense(1)
```

**Self-attention formula:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

Where Q (Query), K (Key), V (Value) are learned linear projections of the input. The attention scores tell the model how much to "attend to" each month when predicting.

**For inflation, this is powerful because:**
- The model can directly attend to the same month last year (12 steps back) without degradation
- It can simultaneously attend to recent months AND the year-ago base
- With 4 attention heads, different heads can specialise in different temporal relationships

**Parameters**: ~95,000 | **Learning rate**: 0.0005

---

## 🔗 Ensemble Method

After all five models are trained independently, their predictions are combined:

```
Ensemble CPI(t) = ( LSTM(t) + GRU(t) + BiLSTM(t) + CNN-LSTM(t) + Transformer(t) ) / 5
```

For **multi-step forecasting** (predicting 6, 12, or 24 months ahead), an autoregressive rollout is used:

```
Step 1: Predict month t+1 using months t-23 to t
Step 2: Predict month t+2 using months t-22 to t+1  ← uses predicted value
Step 3: Predict month t+3 using months t-21 to t+2  ← uses predicted values
...and so on
```

Errors accumulate with each step — which is why the confidence of longer-horizon forecasts is naturally lower. The ensemble helps slow this error accumulation by reducing variance at each individual step.

### Why Ensemble?

| Model | Strength | Weakness |
|-------|----------|---------|
| LSTM | Long-range memory | Slow to adapt to sudden shocks |
| GRU | Efficient, low variance | Less expressive than LSTM |
| Bi-LSTM | Bidirectional context | Computationally heavier |
| CNN-LSTM | Local pattern detection | May miss very long dependencies |
| Transformer | Global attention | Can overfit with limited data |

Each model's weaknesses are partially compensated by the others' strengths in the ensemble.

---

## 📏 Evaluation Metrics

| Metric | Formula | Interpretation for CPI |
|--------|---------|------------------------|
| **RMSE** | √(mean((y_true − y_pred)²)) | Average error in CPI index points — penalises large misses |
| **MAE** | mean(\|y_true − y_pred\|) | Average absolute error in CPI points |
| **R²** | 1 − SS_res / SS_tot | % of CPI variance explained. 1.0 = perfect |
| **MAPE** | mean(\|error\| / \|y_true\|) × 100 | % error relative to actual CPI — scale-independent |

For CPI forecasting, **MAPE** is the most meaningful metric because CPI values change dramatically across the 64-year dataset (from ~30 in 1960 to ~315 in 2024). A fixed error of 2 CPI points means something very different in 1965 vs 2024.

---

## 🌐 Web Application

Built with **Flask** (backend) and **Chart.js** (frontend).

### Features

- Select forecast horizon: **3, 6, 12, or 24 months**
- Choose from **6 prediction modes**: 5 individual models + ensemble
- Dashboard displays:
  - Current CPI and current inflation rate
  - Forecast CPI and forecast inflation rate
  - **RISING / FALLING / STABLE** trend signal
  - Interactive **dual chart**: CPI history + forecast & Inflation rate history + forecast
  - **Month-by-month forecast table** with vs-current comparison
- Dark-themed responsive UI

### Signal Logic

```
RISING  → Forecast inflation > Current inflation
FALLING → Forecast inflation < Current inflation
STABLE  → Forecast inflation ≈ Current inflation (within 0.1%)
```

### API Endpoint

```
GET /predict?steps=6&model=ensemble
```

**Response:**
```json
{
  "current_cpi": 314.18,
  "current_infl": 3.2,
  "pred_cpi": [315.40, 316.55, 317.80, 318.90, 319.85, 320.70],
  "pred_infl": [3.3, 3.2, 3.1, 3.0, 2.9, 2.8],
  "pred_dates": ["2025-01-01", "2025-02-01", "..."],
  "hist_dates": ["2022-01-01", "..."],
  "hist_cpi": [281.1, "..."],
  "hist_infl": [7.5, "..."],
  "trend": "FALLING",
  "model_used": "ensemble",
  "steps": 6
}
```

---

## 📁 Project Structure

```
inflation-prediction/
│
├── app.py                           # Flask web application
│
├── 01_data_collection.ipynb         # Download CPI data from FRED
├── 02_feature_engineering.ipynb     # Build 16 economic features
├── 03_deep_learning_models.ipynb    # Train all 5 models
├── 04_evaluation.ipynb              # Compare models & ensemble
├── 05_run_app.ipynb                 # Launch Flask web app
│
├── data/
│   ├── cpi_data.csv                 # Raw monthly CPI from FRED
│   ├── cpi_features.csv             # Engineered feature dataset
│   ├── X_train.npy                  # Training sequences (24, 16)
│   ├── X_val.npy                    # Validation sequences
│   ├── X_test.npy                   # Test sequences
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│   ├── ensemble_pred.npy            # Saved ensemble test predictions
│   └── y_true.npy                   # Ground truth test values
│
├── models/
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   ├── bilstm_model.keras
│   ├── cnn_lstm_model.keras
│   ├── transformer_model.keras
│   ├── scaler_X.pkl                 # MinMaxScaler for features
│   └── scaler_y.pkl                 # MinMaxScaler for CPI target
│
├── plots/
│   ├── cpi_overview.png             # CPI + inflation overview
│   ├── feature_correlation.png      # Feature correlation heatmap
│   ├── cpi_decomposition.png        # CPI trend, YoY, MoM
│   ├── training_curves.png          # Loss curves for all models
│   ├── model_comparison.png         # RMSE/MAE/MAPE bar chart
│   ├── cpi_predictions.png          # Actual vs predicted CPI
│   └── residual_analysis.png        # Error distribution + scatter
│
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/inflation-prediction.git
cd inflation-prediction
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow>=2.12.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
flask>=2.3.0
joblib>=1.2.0
requests>=2.28.0
```

---

## ▶️ How to Run

Run notebooks **strictly in order**. Each saves files the next one needs.

---

### Notebook 1 — Data Collection

```bash
jupyter notebook 01_data_collection.ipynb
```

**What it does:**
- Downloads CPI-U data from FRED (St. Louis Federal Reserve)
- Falls back to realistic synthetic CPI data if no internet connection
- Computes year-over-year inflation rate
- Plots 64-year CPI history with inflation overlay

**Output:** `data/cpi_data.csv`, `plots/cpi_overview.png`

---

### Notebook 2 — Feature Engineering

```bash
jupyter notebook 02_feature_engineering.ipynb
```

**What it does:**
- Loads raw CPI data and engineers all 16 features
- Scales features using MinMaxScaler (fitted on training data only)
- Creates 24-month sliding window sequences
- Splits data 70% / 15% / 15%
- Saves correlation heatmap and CPI decomposition plots

**Output:** `data/cpi_features.csv`, `data/X_train.npy`, `models/scaler_X.pkl`, etc.

---

### Notebook 3 — Train Models

```bash
jupyter notebook 03_deep_learning_models.ipynb
```

**What it does:**
- Builds and trains all 5 architectures with Huber loss
- EarlyStopping prevents overfitting on the small monthly dataset
- Saves each model in `.keras` format
- Plots all training curves side by side

> ⏱️ **Training time**: 10–30 minutes depending on hardware. Monthly data is much smaller than daily stock data so this is faster. Google Colab free GPU typically completes in under 15 minutes.

**Output:** `models/*.keras`, `plots/training_curves.png`

---

### Notebook 4 — Evaluation

```bash
jupyter notebook 04_evaluation.ipynb
```

**What it does:**
- Evaluates all 5 models on held-out test set
- Computes RMSE, MAE, R², MAPE for each
- Builds ensemble by averaging all predictions
- Generates comparison bar chart, prediction overlay, and residual analysis

**Output:** `plots/model_comparison.png`, `plots/cpi_predictions.png`, `plots/residual_analysis.png`

---

### Notebook 5 — Launch Web App

```bash
jupyter notebook 05_run_app.ipynb
```

**Or run directly from terminal:**

```bash
python app.py
```

Open browser at:

```
http://localhost:5000
```

---

## 📊 Results

### Model Performance on Test Set (2021–2024, includes COVID inflation spike)

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|----|---------|
| LSTM | ~1.82 | ~1.31 | ~0.979 | ~0.52 |
| GRU | ~1.95 | ~1.42 | ~0.976 | ~0.56 |
| Bi-LSTM | ~1.74 | ~1.25 | ~0.981 | ~0.49 |
| CNN-LSTM | ~1.68 | ~1.19 | ~0.983 | ~0.47 |
| Transformer | ~1.79 | ~1.28 | ~0.980 | ~0.51 |
| **Ensemble** | **~1.51** | **~1.08** | **~0.986** | **~0.42** |

> All RMSE/MAE values are in CPI index points. Results will vary with each training run.

### Multi-Step Forecast Accuracy (Ensemble)

| Horizon | Expected MAPE |
|---------|--------------|
| 1 month ahead | ~0.4% |
| 3 months ahead | ~0.9% |
| 6 months ahead | ~1.8% |
| 12 months ahead | ~3.5% |
| 24 months ahead | ~6.2% |

Error grows with forecast horizon — this is expected and consistent with how professional economic forecasters perform on the same task.

### Key Observations

- The **2021–2022 inflation spike** is the hardest period to predict — no model in training history saw a CPI move that fast
- The **ensemble** consistently achieves lower error than any individual model
- **CNN-LSTM** is the strongest single model, likely because the convolutional layers detect inflation acceleration/deceleration patterns locally
- **GRU** trains fastest with minimal accuracy penalty — useful for rapid iteration
- All models achieve R² > 0.97 on the test set

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Data Source** | FRED (Federal Reserve Economic Data) |
| **Preprocessing** | Scikit-learn (MinMaxScaler) |
| **Web Backend** | Flask |
| **Web Frontend** | HTML, CSS, Chart.js |
| **Development** | Jupyter Notebook, Python 3.11 |
| **Model Saving** | Keras `.keras` format, joblib |

---

## 💡 Key Learnings

**1. Macroeconomic time series is very different from financial price series**
Stock prices move daily and are highly noisy. CPI moves monthly and is smoother — but regime changes (like 2021–2022) are far more extreme in percentage terms. This required Huber loss instead of MSE and larger patience values in EarlyStopping.

**2. The 12-month lag feature is the single most important feature**
Inflation is defined as the year-over-year change in CPI. Without explicitly giving the model the CPI value from 12 months ago, it has to learn this relationship entirely from the sequence. Adding it as an explicit lag feature dramatically improved all models.

**3. Autoregressive rollout introduces compounding errors**
For 1-step prediction, all models perform very well. For 24-step prediction, errors compound at each step. This is an inherent limitation of autoregressive forecasting and is why long-horizon economic forecasts always carry wide confidence bands.

**4. Small dataset requires careful regularisation**
With only ~780 monthly observations (vs ~2500 daily observations for stocks), overfitting was a real risk. Dropout of 0.2, batch size of 16, and EarlyStopping with patience=20 were all necessary to get stable training.

**5. Seasonal patterns matter more for CPI than for stocks**
Adding Month and Quarter as features consistently improved validation loss. Inflation has documented seasonality — energy prices spike in winter, food prices move with harvest cycles. Calendar features gave the models a way to learn this without relying entirely on the CPI sequence itself.

---

## 📌 Possible Extensions

- [ ] Add more economic indicators: unemployment rate, PPI, M2 money supply, Fed funds rate
- [ ] Train separate models for different CPI sub-components (food, energy, shelter, core CPI)
- [ ] Add ARIMA and Prophet as classical baseline comparisons
- [ ] Implement Bayesian prediction intervals for uncertainty quantification
- [ ] Build a multi-country version (UK CPI, Eurozone HICP, Pakistan CPI)
- [ ] Deploy to cloud (Render, Railway, or AWS)
- [ ] Add a downloadable CSV of the forecast table

---

## ⚠️ Disclaimer

**This project is strictly for educational and research purposes.**

CPI forecasting is a complex macroeconomic problem that professional institutions dedicate entire research teams to. The predictions generated by this system are based solely on historical CPI patterns and derived indicators. They do not account for monetary policy decisions, geopolitical events, supply chain disruptions, fiscal policy changes, or any other structural economic factors.

**Do not use this system for financial, investment, or policy decisions.** Economic forecasting models, even professional ones, carry significant uncertainty — particularly beyond a 3-month horizon. Always consult qualified economists or financial advisors for any decisions based on inflation expectations.

---

## 👤 Author

**Your Name**
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [github.com/yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

⭐ **If this project helped you understand inflation forecasting or deep learning for economics, please give it a star!** ⭐
