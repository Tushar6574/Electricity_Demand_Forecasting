# TimesFM (Time Series Foundation Model) — Technical Report

## Overview

TimesFM is a **decoder-only foundation model for time-series forecasting** developed by Google Research. Unlike traditional models that are trained from scratch for each dataset, TimesFM is pre-trained on a large corpus of diverse time-series data and can be used for **zero-shot forecasting** — making predictions on new datasets without any fine-tuning.

- **Paper**: "A Decoder-Only Foundation Model for Time-Series Forecasting" (ICML 2024)
- **Model Versions**:

| Version | Parameters | Context Length | Release |
|---------|-----------|---------------|---------|
| 1.0     | 200M      | 512           | 2024    |
| 2.0     | 500M      | 2,048         | 2024    |
| 2.5     | 200M      | 16,384        | 2025    |

---

## Architecture

TimesFM uses a **stacked transformer decoder** architecture:

1. **Input Patching** — The input time series is divided into fixed-length patches (e.g., 32 time steps per patch), reducing sequence length and computational cost.
2. **Decoder-Only Transformer** — Processes the patched sequence using causal self-attention, similar to GPT-style language models.
3. **Output Projection** — Each patch position predicts multiple future time steps (output patch length = 128), enabling multi-horizon forecasting in a single forward pass.
4. **Residual Connections + LayerNorm** — Standard transformer building blocks for stable training.

```
Input:  [t0, t1, ..., tN]
         │
    ┌────┴────┐
    │ Patching │  (patch_len=32)
    └────┬────┘
         │
    ┌────┴────┐
    │  Transformer Decoder × L layers │
    └────┬────┘
         │
    ┌────┴────┐
    │  Output Projection │  (output_patch_len=128)
    └────┬────┘
         │
Output: [pred_t0, ..., pred_t127]
```

### Pre-Training

TimesFM was pre-trained on **~100 billion time points** from:
- Google Trends
- WikiPage views
- Synthetic time series
- Public benchmark datasets (M3, M4, M5, etc.)
- Sensor data
- Financial and economic indicators

This diverse pre-training enables **zero-shot generalization** — the model can forecast patterns it has never seen before.

---

## Benefits Over Traditional Models

### vs. Statistical Models (ARIMA, ETS, Theta)

| Aspect | ARIMA / ETS | TimesFM |
|--------|------------|---------|
| **Setup effort** | Manual differencing, seasonality detection, order selection (p,d,q) | Zero-shot — just pass the series |
| **Multiple seasonality** | Requires manual decomposition (e.g., TBATS) | Handles multiple frequencies natively |
| **External regressors** | Complex to incorporate | Dynamic numerical/categorical covariates through `forecast_with_covariates()` |
| **Uncertainty** | Prediction intervals require assumptions | Quantile heads provide calibrated distributions |
| **Data requirement** | Works on short series | Works on any length (automatic padding/truncation) |

### vs. Machine Learning Models (XGBoost, Random Forest)

| Aspect | XGBoost / RF | TimesFM |
|--------|-------------|---------|
| **Feature engineering** | Must create lag features, rolling windows, differencing manually | Learned from pre-training — no manual feature engineering needed |
| **Time-awareness** | No inherent temporal structure; treats time as arbitrary features | Built-in causal attention captures sequential dependencies |
| **Training** | Requires per-dataset training and hyperparameter tuning | Pre-trained; zero-shot inference on new data |
| **Long-range dependencies** | Limited by manual lag selection (e.g., lag_24, lag_168) | Attention mechanism can look back up to 16,384 steps (v2.5) |
| **Generalization** | Dataset-specific; retrains needed for distribution shifts | Foundation model robust to distribution shifts from pre-training |

### vs. Deep Learning (LSTM, DeepAR, Prophet)

| Aspect | LSTM / DeepAR / Prophet | TimesFM |
|--------|------------------------|---------|
| **Training cost** | Requires GPU-hours per dataset | Pre-trained once; inference only on CPU/GPU |
| **Scalability** | Must train separate model per time series | Single model handles thousands of series |
| **Covariates** | Supported via feature channels | Supported via `forecast_with_covariates()` (dynamic + static) |
| **Quantile forecasting** | Often requires separate models | Built-in continuous quantile head |

---

## Concrete Advantages for Electricity Demand Forecasting

### 1. Zero-Shot Capability
TimesFM was pre-trained on energy demand patterns. It can forecast your electricity load **without any training on your dataset**. This is valuable when:
- You have limited historical data
- You need quick baseline forecasts
- You want to compare against specialized models

### 2. Automatic Seasonality Handling
Electricity demand exhibits daily (24h), weekly (168h), and seasonal patterns. TimesFM's attention mechanism captures these natively — no manual specification of seasonal periods.

### 3. Covariate Support (v2.5+)
`forecast_with_covariates()` accepts:
- **Dynamic numerical covariates** — temperature, humidity, economic indicators
- **Dynamic categorical covariates** — holidays, special events
- **Static categorical covariates** — region, customer type

### 4. Distributional Output
The quantile head provides 10th–90th percentile forecasts, enabling:
- Uncertainty quantification for grid planning
- Risk-aware decision making (e.g., reserve margin estimation)
- Confidence intervals for peak load prediction

### 5. Scalability
TimesFM can forecast hundreds of time series in a single batch call, making it suitable for:
- Substation-level forecasting
- Customer-level load prediction
- Regional grid management

---

## Limitations

| Limitation | Explanation |
|-----------|------------|
| **Context length** | v1.0: 512, v2.0: 2048, v2.5: 16384 — older versions may not capture very long histories |
| **Forecast horizon** | Per-call horizon limited (128/256 steps); requires rolling window for long test periods |
| **Cold start** | Zero-shot performance lags behind trained models on domain-specific data |
| **Inference speed** | 200M–500M parameter model; slower than ARIMA or lightweight ML on CPU |
| **Interpretability** | Black-box transformer; no feature importance or coefficient analysis |
| **Installation** | PyPI version (1.3.0) lags behind GitHub (v2.5 requires repo clone) |

---

## When to Use TimesFM vs. XGBoost

| Scenario | Recommended Model |
|----------|-----------------|
| No training data available | TimesFM (zero-shot) |
| High-accuracy production system | XGBoost (trained on your data) |
| Need uncertainty intervals | TimesFM (quantile head) |
| Need feature importance | XGBoost (built-in importance scores) |
| Large-scale multi-series forecasting | TimesFM (batch inference) |
| Limited compute / deployment on edge | XGBoost (lightweight model) |
| Rapid prototyping | TimesFM (no training required) |
| Interpretability required | XGBoost (SHAP, feature importance) |

---

## References

- Google Research Blog: [A Decoder-Only Foundation Model for Time-Series Forecasting](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- GitHub: [github.com/google-research/timesfm](https://github.com/google-research/timesfm)
- HuggingFace: [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- Paper: Das et al., "A Decoder-Only Foundation Model for Time-Series Forecasting" (ICML 2024)
