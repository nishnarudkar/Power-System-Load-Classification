# Power System Load Classification

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.4+-F7931E?style=flat-square)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2+-150458?style=flat-square)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)

> **Internship Technical Assessment**  
> **Organization:** USEReady  
> **Topic:** Supervised Machine Learning for Industrial Power System Load Classification  
> **Candidate:** Nishant Narudkar  

---

## Executive Summary

Efficient electrical grid management and industrial energy optimization require accurate, automated classification of operating load profiles. This repository presents an end-to-end machine learning system designed to classify industrial power system operating conditions into three operational states:

- **Light_Load**: Low-consumption operating regime (typically corresponding to night shifts, weekends, and non-operational intervals).
- **Medium_Load**: Standard daytime baseline operational capacity.
- **Maximum_Load**: High-intensity electrical demand during peak operating hours.

By analyzing multivariate telemetry—encompassing real active power consumption, leading and lagging reactive energy, power factor indicators, CO2 emissions, and temporal markers—the trained pipeline enables proactive load forecasting, automated demand balancing, peak-demand management, and dynamic tariff optimization.

---

## Technical Objectives and Scope

1. **Data Hygiene and Audit**: Conduct comprehensive quality assessment on high-frequency (15-minute interval) smart-meter telemetry, resolving timestamp continuity, sensor anomalies, and target class imbalance.
2. **Feature Engineering**: Formulate temporal calendar dynamics (hour of day, day of week, month, 15-minute slot index, weekend flags) and clean raw electrical telemetry.
3. **Leakage-Free Temporal Validation**: Enforce chronological data partitions (January–November 2018 for development and cross-validation; December 2018 as a strictly isolated final holdout set).
4. **Pipeline Encapsulation**: Construct robust scikit-learn transformers and pipelines to bundle custom domain cleaning, numerical imputation, and tree ensemble estimation.
5. **Rigorous Evaluation**: Prioritize macro-averaged metrics ($F_1$-score, precision, recall) over raw accuracy to prevent majority-class bias.
6. **Production-Ready Inference**: Provide a standalone, command-line batch inference script for operational scoring on unseen telemetry streams.

---

## Dataset Profile

The dataset (`data/raw/load_data.csv`) records 15-minute telemetry observations across an industrial power system over an entire calendar year (35,041 total observations across 9 attributes).

### Attribute Schema

| Column Name | Type | Description | Range / Unit |
|---|:---:|---|---|
| `Date_Time` | datetime | Timestamp of observation | DD-MM-YYYY HH:MM |
| `Usage_kWh` | float64 | Active real energy consumption | Continuous (kWh) |
| `Lagging_Current_Reactive.Power_kVarh` | float64 | Inductive reactive power consumption | Continuous (kVarh) |
| `Leading_Current_Reactive_Power_kVarh` | float64 | Capacitive reactive power consumption | Continuous (kVarh) |
| `CO2(tCO2)` | float64 | Equivalent greenhouse gas emissions | Continuous ($t\text{CO}_2$) |
| `Lagging_Current_Power_Factor` | float64 | Inductive power factor ratio | Ratio $[0, 100]$ |
| `Leading_Current_Power_Factor` | float64 | Capacitive power factor ratio | Ratio $[0, 100]$ |
| `NSM` | int64 | Number of seconds elapsed from midnight | $[0, 86400)$ |
| `Load_Type` | category | **Target Variable**: `Light_Load`, `Medium_Load`, `Maximum_Load` | Multi-class categorical |

### Target Class Distribution

| Class Label | Observations | Proportion | Description |
|---|:---:|:---:|---|
| **Light_Load** | 18,073 | 51.58% | Majority baseline state |
| **Medium_Load** | 9,696 | 27.67% | Secondary operational state |
| **Maximum_Load** | 7,272 | 20.75% | Peak operating capacity |

*Evaluation Strategy*: Standard accuracy is biased by the 52% majority baseline. **Macro-averaged $F_1$-score** is established as the principal evaluation benchmark to ensure accurate recognition of peak load conditions.

---

## Repository Structure

```text
Power System Load Classification/
├── .gitignore                         # Version control ignore specifications
├── Readme.md                          # Project documentation and performance report
├── requirements.txt                   # Locked project dependencies
├── data/
│   └── raw/
│       └── load_data.csv              # Industrial telemetry dataset (35,041 records)
├── models/
│   ├── final_random_forest.joblib     # Exported scikit-learn pipeline (model + cleaner)
│   └── model_metadata.json            # Model schema, feature list, and class labels
├── notebook/
│   ├── data_audit.ipynb               # Exploratory analysis, data hygiene, and diagnostics
│   └── feature_engineering_modeling.ipynb # Feature pipeline, CV tuning, and holdout evaluation
├── results/
│   ├── final_metrics.json             # Quantified performance metrics on December holdout
│   └── final_predictions.csv          # Prediction logs on December holdout test set
└── src/
    ├── __init__.py                    # Source package initialization
    ├── predict.py                     # CLI batch inference script
    └── preprocessing.py               # PowerFactorCleaner scikit-learn transformer
```

---

## Environment Setup and Installation

### Prerequisites

- Python 3.11 or higher
- `pip` package manager

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishnarudkar/Power-System-Load-Classification.git
   cd Power-System-Load-Classification
   ```

2. **Configure Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Methodology and Validation Strategy

```mermaid
flowchart LR
    A[Raw Telemetry Data] --> B[Data Audit & Sanitation]
    B --> C[Calendar & Electrical Feature Engineering]
    C --> D[Chronological Split: Jan-Nov Dev vs Dec Holdout]
    D --> E[Pipeline Training: Cleaner + Imputer + Random Forest]
    E --> F[December Holdout Validation & Metric Export]
    F --> G[CLI Batch Prediction Script]
```

### 1. Data Cleaning and Preprocessing

- **Chronological Sorting**: Enforced strict temporal ordering on parsed timestamps.
- **Power Factor Validation**: Addressed out-of-bounds sensor readings ($< 0$ or $> 100$) via a custom `PowerFactorCleaner` transformer, mapping invalid sensor artifacts to `NaN` values for subsequent median imputation.
- **Duplicate Timestamp Resolution**: Evaluated duplicate sensor records on the December holdout using a quality-issue score, preserving the highest-integrity record and verifying complete 2,976-record continuity.

### 2. Feature Engineering

The model incorporates both temporal structure and electrical dynamics:
- **Temporal Calendar Attributes**: `Hour`, `Day_of_Week`, `Month`, `Time_Slot` (15-minute index from 0 to 95), and `Is_Weekend` binary indicator.
- **Telemetry Indicators**: `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor`.

### 3. Chronological Holdout Protocol

To simulate real-world deployment and prevent lookahead data leakage:
- **Development & Cross-Validation**: January 1, 2018 to November 30, 2018 (32,064 observations).
- **Final Holdout Test Partition**: December 1, 2018 to December 31, 2018 (2,976 observations).
- *Strict Rule*: The December holdout partition was isolated and frozen until final model evaluation.

---

## Model Evaluation and Results

### Validation Benchmarks (Development Set)

Models evaluated on the validation partition (October–November 2018):

| Model Architecture | Feature Subset | Validation Accuracy | Validation Macro $F_1$ |
|---|---|:---:|:---:|
| Dummy Classifier (Stratified) | Calendar | 48.36% | 0.2173 |
| Decision Tree Classifier | Calendar | 83.47% | 0.7898 |
| Random Forest Classifier | Calendar Only | 83.47% | 0.7898 |
| **Combined Random Forest** | **Calendar + Telemetry** | **83.73%** | **0.7925** |

### Final Evaluation on December 2018 Holdout

The final frozen Random Forest pipeline was evaluated against the out-of-time December 2018 holdout partition (2,976 records):

| Metric | Score |
|---|:---:|
| **Accuracy** | **89.15%** |
| **Macro Precision** | **86.92%** |
| **Macro Recall** | **87.15%** |
| **Macro $F_1$-Score** | **86.31%** |

### Per-Class Performance Breakdown

| Class | Precision | Recall | $F_1$-Score | Support (Records) |
|---|:---:|:---:|:---:|:---:|
| **Light_Load** | 0.98 | 0.91 | 0.94 | 1,744 |
| **Maximum_Load** | 0.90 | 0.76 | 0.82 | 528 |
| **Medium_Load** | 0.73 | 0.95 | 0.82 | 704 |
| **Macro Average** | **0.87** | **0.87** | **0.86** | **2,976** |
| **Weighted Average** | **0.91** | **0.89** | **0.89** | **2,976** |

---

## Batch Inference Pipeline

The repository provides a standalone batch prediction utility in `src/predict.py`.

### Running Predictions from CLI

Execute predictions on any input CSV containing telemetry or calendar fields:

```bash
python src/predict.py --input data/raw/load_data.csv --output results/new_predictions.csv
```

### Script Execution Parameters

- `--input` (required): Filepath to the raw CSV containing telemetry columns and `Date_Time`.
- `--output` (optional, default: `results/new_predictions.csv`): Destination path for the scored predictions.

---

## Author and Project Information

- **Candidate**: Nishant Narudkar
- **Assessment**: USEReady Internship Selection Program
- **Deliverable**: Data Science & Machine Learning Pipeline
- **Repository**: [Power-System-Load-Classification](https://github.com/nishnarudkar/Power-System-Load-Classification)