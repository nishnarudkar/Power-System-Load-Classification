# Power System Load Classification

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-blue?style=for-the-badge)](#)

> **Internship Assignment Submission**  
> **Organization:** [USEReady](https://www.useready.com/)  
> **Program:** Internship Selection Program  
> **Topic:** Supervised Machine Learning for Power System Load Classification  

---

## 📌 Executive Summary

Modern smart grid management and industrial energy optimization rely on accurate operational state awareness. This project implements an end-to-end Machine Learning pipeline to classify industrial power system operating conditions into three distinct operational states:

- **`Light_Load`** — Minimal energy consumption (e.g., non-operational hours, night shifts).
- **`Medium_Load`** — Moderate industrial activity (standard daytime operational baseline).
- **`Maximum_Load`** — Peak operating capacity and intensive electrical utilization.

By analyzing multi-dimensional telemetry (active/reactive power consumption, power factors, CO₂ emissions, and temporal markers), the predictive model facilitates load forecasting, automated grid demand balancing, peak demand management, and billing tier optimization.

---

## 🎯 Assignment Objectives & Scope

This repository fulfills the technical assessment assigned by **USEReady** for the Internship Program:

1. **Exploratory Data Analysis & Audit:** Comprehensive assessment of raw telemetry data, including distributions, class imbalance, missing value profiles, and timestamp consistency.
2. **Robust Data Preprocessing:** Leakage-free handling of missing entries, time-series feature engineering, and cyclic encoding of temporal patterns.
3. **Temporal Validation Protocol:** Strict chronological data splitting ensuring the final calendar month (December 2018) is isolated as an untouched out-of-time holdout test set.
4. **Model Exploration & Benchmarking:** Systematic comparison of baseline heuristics, linear estimators, tree-based ensembles, and gradient boosting architectures.
5. **Performance Evaluation:** Holistic assessment leveraging macro-averaged metrics ($F_1$-score, precision, recall) and multi-class confusion matrices to address target imbalance.
6. **Reproducibility & Code Cleanliness:** Modular repository structure with documented dependencies and reproducible Jupyter execution.

---

## 📊 Dataset Profile

The dataset (`data/raw/load_data.csv`) captures 15-minute interval telemetry spanning an entire operational calendar year (35,041 observations across 9 attributes).

### Attribute Schema

| Column Name | Data Type | Description | Unit / Format |
|---|:---:|---|---|
| `Date_Time` | `datetime` | Observation timestamp | `DD-MM-YYYY HH:MM` |
| `Usage_kWh` | `float64` | Real active energy consumption | kWh |
| `Lagging_Current_Reactive.Power_kVarh` | `float64` | Lagging reactive power consumption | kVarh |
| `Leading_Current_Reactive_Power_kVarh` | `float64` | Leading reactive power consumption | kVarh |
| `CO2(tCO2)` | `float64` | Greenhouse gas emission factor equivalent | $t\text{CO}_2$ |
| `Lagging_Current_Power_Factor` | `float64` | Power factor during lagging inductive load | Ratio $[0, 100]$ |
| `Leading_Current_Power_Factor` | `float64` | Power factor during leading capacitive load | Ratio $[0, 100]$ |
| `NSM` | `int64` | Number of Seconds from Midnight | Seconds $[0, 86400)$ |
| `Load_Type` | `category` | **Target Label**: `Light_Load`, `Medium_Load`, `Maximum_Load` | Multi-class label |

### Target Distribution

| Class Label | Frequency (Records) | Relative Share (%) | Nature |
|---|:---:|:---:|---|
| **`Light_Load`** | 18,073 | ~51.58% | Majority baseline class |
| **`Medium_Load`** | 9,696 | ~27.67% | Secondary operational class |
| **`Maximum_Load`** | 7,272 | ~20.75% | Minority peak load class |

> **Note on Evaluation:** Due to substantial class imbalance (~52% vs. ~21%), standard accuracy is insufficient. **Macro-averaged $F_1$-score** is established as the primary decision metric to ensure minority peak-load states are detected with equal fidelity.

### Raw Data Missingness Summary

| Feature Name | Missing Count | % Missing | Handling Strategy |
|---|:---:|:---:|---|
| `Usage_kWh` | 1,559 | 4.45% | Time-aware rolling median / iterative imputation |
| `Lagging_Current_Reactive.Power_kVarh` | 876 | 2.50% | Forward fill with seasonal median fallback |
| `Leading_Current_Reactive_Power_kVarh` | 1,156 | 3.30% | Forward fill with seasonal median fallback |
| `CO2(tCO2)` | 455 | 1.30% | Derived from `Usage_kWh` emission coefficient |
| `Lagging_Current_Power_Factor` | 350 | 1.00% | Bounded numerical imputation |
| `Leading_Current_Power_Factor` | 1,471 | 4.20% | Bounded numerical imputation |
| `NSM` | 455 | 1.30% | Deterministically recomputed from `Date_Time` |
| `Date_Time` / `Load_Type` | 0 | 0.00% | Complete (no missing records) |

---

## 🏗️ Project Architecture

The repository adheres to standardized data science engineering principles:

```text
Power System Load Classification/
├── .gitignore                     # Git ignore rules for Python, virtualenvs & checkpoints
├── Readme.md                      # Comprehensive project documentation
├── requirements.txt               # Locked production & experimentation dependencies
├── data/
│   └── raw/
│       ├── .gitkeep
│       └── load_data.csv          # Telemetry dataset (35,041 records)
├── notebook/
│   └── data_audit.ipynb           # EDA, data hygiene, distributions & sanity audits
├── reports/
│   └── figures/
│       └── .gitkeep               # Generated visualization artifacts
└── src/
    └── .gitkeep                   # Reusable ETL, feature engineering & model modules
```

---

## ⚙️ Environment Setup & Installation

### Prerequisites
- **Python:** Version `3.11` or higher
- **Package Manager:** `pip`

### Step-by-Step Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nishnarudkar/Power-System-Load-Classification.git
   cd Power-System-Load-Classification
   ```

2. **Initialize Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Core Dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Register Jupyter Kernel:**
   ```bash
   python -m ipykernel install --user --name power-load-ml --display-name "Python (Power Load ML)"
   ```

---

## 🔬 End-to-End Methodology

```mermaid
flowchart LR
    A[Raw Telemetry Data] --> B[Data Audit & EDA]
    B --> C[Timestamp Consistency & Imputation]
    C --> D[Feature Engineering & Cyclic Encoding]
    D --> E[Chronological Split: Dev vs Dec 2018 Holdout]
    E --> F[Model Benchmarking & Hyperparameter Tuning]
    F --> G[Final Holdout Evaluation & Reporting]
```

### 1. Data Audit & Exploratory Analysis (`notebook/data_audit.ipynb`)
- Verification of continuous chronological timestamps.
- Consistency cross-validation between `NSM` (Seconds from Midnight) and parsed `Date_Time`.
- Correlation analysis between active/reactive power components and power factors.

### 2. Time-Aware Preprocessing & Feature Engineering
- **Cyclic Temporal Encodings:** Transform hour of day and month into sine/cosine harmonics:
  $$\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
- **Lag & Rolling Dynamics:** Capture sequential momentum via exponential rolling means and previous-interval load shifts.
- **Leakage Prevention:** All scalers, transformers, and imputers are strictly fitted on the development partition and only transformed on the test partition.

### 3. Chronological Holdout Evaluation Strategy
- **Development Set (Training & Cross-Validation):** January 1, 2018 – November 30, 2018 (11 months).
- **Final Holdout Test Set:** December 1, 2018 – December 31, 2018 (1 full calendar month).
- *Strict Rule:* Zero data from December 2018 is utilized for model selection, thresholding, or hyperparameter exploration.

---

## 📈 Benchmark Models & Evaluation Framework

### Planned Candidate Architectures
1. **Dummy Baseline:** Stratified and majority-class heuristic classifier.
2. **Regularized Multinomial Logistic Regression:** Linear baseline with standardized numerical inputs.
3. **Random Forest Classifier:** Non-linear bagged ensemble capturing interaction terms.
4. **Gradient Boosted Decision Trees (HistGradientBoosting / LightGBM):** Optimized tree ensemble capable of robust performance on tabular telemetry.

### Metrics Matrix

$$\text{Macro } F_1 = \frac{1}{K}\sum_{k=1}^K F_{1, k}, \quad K = 3$$

| Model Architecture | Cross-Validation Macro $F_1$ | Holdout Test Accuracy | Holdout Test Macro $F_1$ | Status |
|---|:---:|:---:|:---:|:---:|
| **Dummy Heuristic** | *Baseline* | *Pending* | *Pending* | Planned |
| **Logistic Regression** | *TBD* | *Pending* | *Pending* | Planned |
| **Random Forest** | *TBD* | *Pending* | *Pending* | Planned |
| **Gradient Boosting** | *TBD* | *Pending* | *Pending* | Planned |

---

## 🛡️ Best Practices & Quality Assurance

- **Zero Data Leakage:** Preprocessing transformations do not peek across time splits.
- **Reproducibility:** Seed pinning across all stochastic operations (`random_state=42`).
- **Production Readiness:** Code structured for transition from notebook exploration to modular Python scripts in `src/`.

---

## 📬 Submission & Author Details

- **Candidate:** Nishchay Narudkar
- **Submission For:** USEReady Internship Assessment
- **Deliverable:** Jupyter Notebook & GitHub Project Repository
- **Repository:** [Power-System-Load-Classification](https://github.com/nishnarudkar/Power-System-Load-Classification)