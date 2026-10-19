# SmartPlant AI — Project Evaluation Report
**DATA DIVERGENCE Hackathon Final Submission**  
**Track:** Machine Learning & Energy Analytics  
**Date:** October 2026  
**Target Variable:** `next_interval_Usage_kWh` (15-Minute Forward Forecasting)

---

## 1. Executive Summary
SmartPlant AI is a machine-learning energy forecasting and anomaly detection system engineered for industrial steel manufacturing facilities. Industrial utility electricity tariffs rely on 15-minute maximum demand windows; sudden consumption surges trigger disproportionate peak penalty charges across monthly billing cycles.

To mitigate this, SmartPlant AI forecasts electricity consumption for the next 15-minute interval using past consumption momentum and real-time electrical telemetry. Once the actual meter observation arrives, a statistical anomaly detector evaluates forecast errors using robust Median Absolute Deviation (MAD) thresholds to detect operating irregularities (e.g., secondary auxiliary equipment left running, sudden power factor degradation, or unscheduled furnace changeovers) without triggering alert fatigue.

---

## 2. Dataset & Integrity Audit
The system was developed and validated on genuine SCADA telemetry from an active steel fabrication plant:
- **Total Observations:** 28,031 sequential 15-minute readings spanning January 1, 2018 to October 19, 2018 (~9.6 months).
- **Data Quality:** Zero missing values across all 12 columns, zero duplicate timestamps, and strictly continuous 15-minute resolution across all day transitions.
- **Target Alignment:** Confirmed that for all intervals $t \in [0, 28029]$, `next_interval_Usage_kWh[t] == Usage_kWh[t+1]` with zero index shift errors or future lookahead leakage.
- **Dataset Preservation:** The original `train.csv` remains untouched; the audited dataset is saved as `data/cleaned_dataset.csv` (2.47 MB).

---

## 3. Modeling Methodology

### 3.1 Delta Target Formulation
Rather than predicting raw energy consumption directly—which tends to lag sudden load changes—the champion model predicts the 15-minute change:
$$\Delta = y_{t+1} - y_t$$
The predicted energy is then calculated as:
$$\hat{y}_{t+1} = \max(0, y_t + \hat{\Delta})$$

### 3.2 37 Domain-Engineered Features
At time $t$, the system constructs 37 features using strictly historical information:
1. **Lags:** $y_{t-1}, y_{t-2}, y_{t-4}, y_{t-8}, y_{t-16}, y_{t-32}, y_{t-96}$ (24-hour seasonal lag), and $y_{t-192}$ (48-hour lag).
2. **Rolling Statistics:** 1-hour, 4-hour, and 24-hour rolling means, standard deviations, minimums, and maximums (excluding current observation $y_t$).
3. **Momentum & Ramps:** First difference ($y_{t-1} - y_{t-2}$), ramp acceleration, and offsets from yesterday and last week.
4. **Electrical Telemetry:** Apparent power $S = \sqrt{P^2 + Q^2}$ (in kVAh), reactive ratio $Q / (P + 10^{-3})$, and lagging power factor interaction.
5. **Cyclical Calendar:** Sine and cosine encodings of hour of day and day of week, weekend indicator, and tariff transition markers.

### 3.3 Champion Architecture
The champion model is a weighted ensemble combining:
- **55% XGBoost Regressor**
- **45% LightGBM Regressor**
- Preprocessed via a scikit-learn `ColumnTransformer` with median imputation, standard scaling, and one-hot encoding.
- Compact artifact size: **2.34 MB** (`energy_model_breakthrough.joblib`).

---

## 4. Evaluation Results

### 4.1 Chronological Benchmark (80% Train / 20% Test)
Evaluated across **5,607 out-of-sample test intervals** (August 22, 2018 to October 19, 2018; ~1.9 months):

| Model Architecture | Test MAE (kWh) | Test RMSE (kWh) | Error Reduction vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Naive Persistence Baseline ($y_{t+1} = y_t$)** | 5.2769 | 11.9448 | Baseline |
| Seasonal 24-Hour Lag ($y_{t+1} = y_{t-95}$) | 21.6039 | 32.1676 | +169.3% (Shift Divergence) |
| Rollback Tuned Random Forest (`RF_eng`) | 4.8872 | 10.4357 | -12.63% |
| **SmartPlant AI Champion (55% XGB + 45% LGB)** | **3.6774** | **8.0606** | **-32.52%** |

### 4.2 Inference Latency
- **Cold-Start (First Interval):** ~28–45 ms (thread pool initialization).
- **Warm Steady-State:** **10.02 ms average** (hardware performance counter), easily meeting real-time 15-minute dispatch timelines.

---

## 5. Statistical Anomaly Detection & Alert Engine

### 5.1 Robust Z-MAD Formulation
Anomaly detection is evaluated **strictly after the actual reading arrives at time $t+1$**:
$$e = y_{t+1} - \hat{y}_{t+1}$$
$$\text{MAD} = \text{median}(|e - \text{median}(e)|) = 0.50\text{ kWh}$$
$$Z_{\text{MAD}} = 0.6745 \times \frac{e - \text{median}(e)}{\text{MAD}}$$

### 5.2 Severity & Alert Deduplication
- **Severity Tiers:** Low ($|Z| \ge 2.0$), Medium ($|Z| \ge 3.0$), High ($|Z| \ge 4.5$ or $|e| \ge 35.24\text{ kWh}$), Critical ($|Z| \ge 6.0$ or $|e| \ge 61.32\text{ kWh}$, or sensor data faults).
- **Deduplication Cooldown:** Consecutive anomalies of the same type within 2 intervals update occurrence counts without dispatching duplicate alarms.
- **Operator Advisory:** Anomalies flag statistical deviations and degraded power factor (<70%), prompting operational reviews (e.g., checking capacitor steps or unscheduled furnace starts).

---

## 6. Limitations & Operational Scope
1. **Absence of Ground-Truth Incident Logs:** Sensor telemetry does not include work-order maintenance tickets; anomalies represent statistical outliers rather than confirmed physical failures.
2. **Facility-Specific Baseline:** Model weights and threshold quantiles reflect this steel plant; adapting to discrete manufacturing requires retraining on local baseline load curves.
3. **Unscheduled Shift Jumps:** Heavy equipment power-ups occurring without prior electrical precursor signals account for the majority of large squared errors. Integrating digital MES shift schedules would further improve transition accuracy.

---

## 7. Submission Verification
- **Automated Test Suite:** 39 / 39 tests passing in ~1.0 second (`tests/test_smartplant.py`, `tests/test_anomaly_detection.py`).
- **Interactive Notebook:** `SmartPlant_AI.ipynb` verified 100% executable top-to-bottom with all plots and outputs pre-rendered.
- **Terminal Demonstration:** `run_demo.py` verifies causal 15-minute forecasting, actual error computation, and alert lifecycle.
