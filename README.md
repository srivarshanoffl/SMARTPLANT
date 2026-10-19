# SmartPlant AI — Industrial Energy Forecasting & Anomaly Detection

A machine-learning forecasting and statistical anomaly detection system for 15-minute industrial electricity consumption in a steel manufacturing plant.

---

## 1. Project Title & Description
**SmartPlant AI** is a Python-based machine-learning system designed to forecast industrial electricity consumption for the next 15-minute interval and detect abnormal energy deviations. Built using genuine SCADA telemetry from a steel manufacturing facility, it combines a gradient-boosted delta ensemble with robust statistical anomaly detection (Z-MAD) to support industrial peak-demand management and operational monitoring.

---

## 2. Problem Statement
Industrial electricity tariffs in steel manufacturing are heavily structured around 15-minute maximum demand windows. If electricity consumption surges unexpectedly during a single 15-minute interval, utility companies levy significant peak demand penalty charges across the entire monthly billing period.

Operating facilities face two primary challenges:
1. **Unplanned Demand Surges:** Plant engineers lack short-term (15-minute ahead) foresight to adjust controllable secondary loads before a new peak is registered.
2. **Unexplained Energy Deviations:** Sudden discrepancies between expected energy consumption and actual meter readings often indicate operating inefficiencies (e.g., secondary cooling pumps left running during idling, degraded power factor, or unscheduled furnace changeover delays).

---

## 3. Project Objectives
1. **Accurate Short-Term Forecasting:** Predict active electricity usage (`Usage_kWh`) for the next 15-minute interval using past consumption history and electrical telemetry.
2. **Statistical Anomaly Detection:** Once actual meter readings arrive, compare actual versus predicted consumption to identify anomalous deviations using robust Median Absolute Deviation (MAD) statistics.
3. **Operational Decision Support:** Provide actionable operator advisories (e.g., power factor warnings, changeover ramp notices) without generating alert fatigue.
4. **Reproducibility:** Provide clean code, an executed Jupyter Notebook, a verified test suite, and a terminal demonstration runnable on standard hardware.

---

## 4. Dataset Description & Source
- **Origin:** Real supervisory control and data acquisition (SCADA) telemetry from a steel manufacturing plant.
- **File:** `data/cleaned_dataset.csv` (derived and audited from original `train.csv`).
- **Observations:** 28,031 sequential 15-minute intervals spanning January 1, 2018 to October 19, 2018 (~9.6 months).
- **Sampling Interval:** Strictly continuous 15-minute intervals (96 intervals per 24-hour day).
- **Data Quality:** Zero missing values, zero duplicates, and validated target alignment across midnight boundaries.

### Features
| Column Name | Units / Type | Description |
| :--- | :--- | :--- |
| `date` | Timestamp String | Date and 15-minute interval time (`DD/MM/YYYY HH:MM`) |
| `Usage_kWh` | kWh (Float) | Active electricity consumption during the current 15-minute interval |
| `Lagging_Current_Reactive.Power_kVarh` | kVarh (Float) | Reactive energy from inductive equipment (motors, transformers) |
| `Leading_Current_Reactive_Power_kVarh` | kVarh (Float) | Reactive energy from capacitive equipment |
| `CO2(tCO2)` | Metric Tons (Float) | Carbon emissions associated with energy consumed |
| `Lagging_Current_Power_Factor` | % (0 to 100) | Ratio of active power to apparent power (lagging) |
| `Leading_Current_Power_Factor` | % (0 to 100) | Ratio of active power to apparent power (leading) |
| `NSM` | Seconds (Integer) | Number of seconds elapsed from midnight (0 to 86,400) |
| `WeekStatus` | Categorical | `Weekday` or `Weekend` |
| `Day_of_week` | Categorical | Day of the week (`Monday` through `Sunday`) |
| `Load_Type` | Categorical | Tariff period: `Light_Load`, `Medium_Load`, or `Maximum_Load` |
| `next_interval_Usage_kWh` | kWh (Float) | Target variable $y_{t+1}$: active electricity consumption in next interval |

---

## 5. Technologies & Libraries
- **Language:** Python 3.10+ (tested on Python 3.14 on Windows AMD64)
- **Data Processing:** `pandas>=2.2.0`, `numpy>=2.0.0`, `scipy>=1.14.0`
- **Machine Learning:** `scikit-learn>=1.5.0`, `xgboost>=2.0.0`, `lightgbm>=4.0.0`, `joblib>=1.4.0`
- **Visualization:** `matplotlib>=3.8.0`
- **Notebook Environment:** `nbformat>=5.9.0`, `nbclient>=0.11.0`, `ipykernel>=6.29.0`
- **Testing Framework:** Standard Python `unittest`

---

## 6. Project Folder Structure
```text
DATA_DIVERGENCE_SUBMISSION/
├── README.md                           # Comprehensive project documentation
├── requirements.txt                     # Pinned project dependencies
├── SmartPlant_AI.ipynb                 # Completed 10-section Jupyter Notebook
├── run_demo.py                         # Standalone terminal demonstration script
├── data/
│   └── cleaned_dataset.csv             # Audited 28,031-row SCADA dataset
├── src/
│   ├── __init__.py                     # Package initializer
│   ├── predict.py                      # Production model loading & 37-feature inference
│   ├── anomaly_detector.py             # Robust Z-MAD statistical anomaly detector
│   ├── alert_engine.py                 # Alert lifecycle & deduplication cooldown engine
│   ├── monitoring_pipeline.py          # Sequential tracking & historical replay pipeline
│   ├── asset_dataset_adapter.py        # Adapter for external multi-asset datasets
│   └── experiment_and_train.py         # Lag feature engineering reference
├── models/
│   ├── energy_model_breakthrough.joblib # 2.34 MB Champion Delta Ensemble (55% XGB + 45% LGB)
│   └── anomaly_detector_calibration.json # Pre-computed residual median, MAD, and quantiles
├── smartplant_output/
│   ├── energy_model_breakthrough.joblib # Mirror directory for backward compatibility
│   └── anomaly_detector_calibration.json
├── results/
│   ├── final_validation_report.json    # Complete audit and verification report (JSON)
│   ├── final_validation_report.txt     # Human-readable validation summary
│   ├── model_demo_output.txt           # Captured terminal demonstration log
│   └── original_model_metrics.json     # Model training benchmark record
└── tests/
    ├── __init__.py                     # Test suite initializer
    ├── test_smartplant.py              # 19 tests for inference, features, and alignment
    └── test_anomaly_detection.py       # 20 tests for anomalies, alerts, and latency
```

---

## 7. Model Approach

### Delta Formulation ($\Delta = y_{t+1} - y_t$)
Direct forecasting of raw consumption often lags behind sudden industrial load transitions. SmartPlant AI predicts the **consumption change** $\Delta$ rather than the raw value:
$$\hat{\Delta} = f(\mathbf{x}_t)$$
$$\hat{y}_{t+1} = \max(0, y_t + \hat{\Delta})$$

### 37 Domain-Engineered Features
At interval $t$, the system constructs 37 features without lookahead leakage:
1. **Lags:** $y_{t-1}, y_{t-2}, y_{t-4}, y_{t-8}, y_{t-16}, y_{t-32}, y_{t-96}$ (24 hours prior), $y_{t-192}$ (48 hours prior).
2. **Rolling Statistics (excluding $y_t$):** 4-interval, 16-interval, and 96-interval rolling means, standard deviations, max, and min.
3. **Momentum:** First-difference delta ($y_{t-1} - y_{t-2}$), multi-step difference ($y_{t-1} - y_{t-5}$), current ramp rate ($y_t - y_{t-1}$), and ramp acceleration.
4. **Electrical Telemetry:** Apparent power $S = \sqrt{P^2 + Q^2}$, reactive ratio $Q / (P + 10^{-3})$, and lagging power factor interaction.
5. **Cyclical Calendar:** Sine and cosine transformations of hour of day and day of week, weekend indicator, and tariff transition markers.

### Champion Weighted Ensemble
- **55% XGBoost Regressor**
- **45% LightGBM Regressor**
- Features preprocessed via `ColumnTransformer` with `SimpleImputer(strategy='median')`, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')`.

---

## 8. Anomaly Detection Approach

Anomaly detection operates **strictly after the actual reading arrives at time $t+1$**.

### 1. Residual Computation
$$e = y_{t+1} - \hat{y}_{t+1}$$

### 2. Robust Z-MAD Metric
Standard deviations are susceptible to industrial outliers. We calculate robust Z-scores using the Median Absolute Deviation fit on training residuals:
$$\text{MAD} = \text{median}(|e - \text{median}(e)|)$$
$$Z_{\text{MAD}} = 0.6745 \times \frac{e - \text{median}(e)}{\text{MAD}}$$

### 3. Severity Categorization
- **Low:** $|Z_{\text{MAD}}| \ge 2.0$ (minor deviation within normal variance)
- **Medium:** $|Z_{\text{MAD}}| \ge 3.0$ (moderate deviation)
- **High:** $|Z_{\text{MAD}}| \ge 4.5$ or $|e| \ge P_{95}$ ($35.24\text{ kWh}$)
- **Critical:** $|Z_{\text{MAD}}| \ge 6.0$ or $|e| \ge P_{99}$ ($61.32\text{ kWh}$), or invalid sensor data (NaN, Inf, negative usage)

### 4. Alert Lifecycle & Deduplication
To prevent alarm fatigue, consecutive alerts of the same category within 2 intervals are deduplicated, updating the occurrence count while enforcing a cooldown period. Active alerts can be transitioned: `New` $\rightarrow$ `Acknowledged` $\rightarrow$ `Resolved`.

---

## 9. Verified Evaluation Results

### Chronological Holdout (80% Train / 20% Test)
- **Training Set:** First 22,424 observations (January 1, 2018 to August 22, 2018)
- **Held-Out Test Set:** Final 5,607 observations (August 22, 2018 to October 19, 2018; ~1.9 months)

### Benchmark Comparison
| Metric | Naive Persistence Baseline ($y_{t+1} = y_t$) | SmartPlant AI Champion Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Test MAE** | 5.2769 kWh | **3.6774 kWh** | **-30.31%** error reduction |
| **Test RMSE** | 11.9448 kWh | **8.0606 kWh** | **-32.52%** error reduction |
| **Evaluated Intervals** | 5,607 intervals | 5,607 intervals | Full chronological test split |

### Latency Performance
- **Cold-Start Latency:** ~45 ms (initial C++ thread pool spin-up)
- **Warm Steady-State Latency:** **10 to 12 ms average** (comfortably within the 15-minute operational budget)

---

## 10. Installation Instructions

### Step 1: Clone or Navigate to the Directory
```bash
cd DATA_DIVERGENCE_SUBMISSION
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. How to Execute the Notebook
The notebook `SmartPlant_AI.ipynb` is pre-executed with all cell outputs, charts, and tables visible.

To open and run the notebook interactively:
```bash
# Using JupyterLab / Notebook:
jupyter notebook SmartPlant_AI.ipynb

# Or open DATA_DIVERGENCE_SUBMISSION directly in VS Code and select your Python kernel.
```
In Jupyter or VS Code, select **Kernel** $\rightarrow$ **Restart & Run All**. Every cell executes sequentially from top to bottom without errors.

---

## 12. How to Run the Terminal Demonstration
To run the interactive step-by-step terminal demonstration over a real operational sequence:
```bash
python run_demo.py
```
This script loads the champion model, evaluates a 10-interval (2.5-hour) morning shift ramp from the test set, demonstrates forecast-error calculations, triggers anomaly alerts, and writes a log to `results/model_demo_output.txt`.

---

## 13. How to Run the Test Suite
To execute all 39 automated unit and regression tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
**Expected Output:**
```text
Ran 39 tests in ~1.0s
OK
```

---

## 14. Known Limitations
1. **Absence of Maintenance Incident Logs:** The dataset records electrical measurements and tariff categories, but does not include equipment maintenance repair tickets. Statistical anomalies flag operational deviations, not confirmed physical mechanical breakdowns.
2. **Facility-Specific Baselines:** The model and threshold distributions are calibrated to this specific steel fabrication plant. Transferring to other manufacturing industries (e.g., pharmaceutical or electronics assembly) requires re-fitting on site-specific baseline load curves.
3. **Unscheduled Shift Jumps:** Heavy equipment power-ups (e.g., starting an electric arc furnace) that occur with no prior electrical warning account for the majority of large forecast residuals. Integrating digital Manufacturing Execution System (MES) shift schedules would further improve transition accuracy.
