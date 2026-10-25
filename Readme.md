Power System Load Classification
A machine learning project to classify industrial power-system operating conditions as Light_Load, Medium_Load, or Maximum_Load from historical electricity-consumption and electrical measurements.
Project status: Dataset audit in progress. Modeling and final evaluation results will be added after the notebooks are executed. No performance metrics are claimed yet.

1. Objective
The internship assignment asks for a supervised classification model that predicts Load_Type using historical power-system data. The workflow covers data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, and evaluation. The last month of observations must be reserved for final testing. The requested submission is a Jupyter notebook, either in a ZIP archive or via a GitHub repository.
2. Dataset
The supplied file, load_data.csv, contains 35,041 rows and 9 columns. The target is Load_Type.
Column	Description
Date_Time	Timestamp of the observation
Usage_kWh	Energy consumption (kWh)
Lagging_Current_Reactive.Power_kVarh	Lagging reactive energy (kVarh)
Leading_Current_Reactive_Power_kVarh	Leading reactive energy (kVarh)
CO2(tCO2)	CO₂ measurement (as labeled in the CSV)
Lagging_Current_Power_Factor	Lagging power factor
Leading_Current_Power_Factor	Leading power factor
NSM	Number of seconds from midnight
Load_Type	Target class


Target distribution
Class	Records
Light_Load	18,073
Medium_Load	9,696
Maximum_Load	7,272


Class imbalance motivates reporting macro-averaged metrics alongside accuracy.
Missing values in the raw file
Feature	Missing
Usage_kWh	1,559
Lagging_Current_Reactive.Power_kVarh	876
Leading_Current_Reactive_Power_kVarh	1,156
CO2(tCO2)	455
Lagging_Current_Power_Factor	350
Leading_Current_Power_Factor	1,471
NSM	455
Date_Time	0
Load_Type	0


These are raw-data counts, not evidence that any missing values have been corrected. The audit also checks duplicate timestamps, timestamp parsing, and whether NSM agrees with the time of day.
3. Repository structure
power-load-classification/
├── data/
│   └── raw/
│       └── load_data.csv
├── notebooks/
│   └── 01_data_audit.ipynb
├── src/
├── reports/
│   └── figures/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
This is the initial project structure. Modeling notebooks and reusable scripts can be added as implementation progresses.
4. Environment setup
Recommended: Python 3.11, VS Code with the Python and Jupyter extensions.
1. Clone or download the repository and open it in VS Code.
2. Place the supplied CSV at data/raw/load_data.csv if it is not included in the repository.
3. Create a virtual environment:
   py -3.11 -m venv .venv
4. Activate it on Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   On macOS/Linux, use source .venv/bin/activate instead.
5. Install dependencies:
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
6. Register the Jupyter kernel:
   python -m ipykernel install --user --name power-load-ml --display-name "Python (Power Load ML)"
7. Open notebooks/01_data_audit.ipynb, select Python (Power Load ML), and run the cells in order. Its data path is relative to the notebooks/ directory (../data/raw/load_data.csv).
The initial requirements.txt contains:
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
ipykernel
openpyxl
If additional modeling libraries are introduced, update requirements.txt accordingly.
5. Methodology
Stage 1 — Data audit and EDA
- Inspect dimensions, feature types, descriptive statistics, and missingness.
- Check duplicate records and duplicate timestamps.
- Parse Date_Time using day-first format (%d-%m-%Y %H:%M).
- Inspect monthly coverage and chronological order.
- Plot target-class balance and numeric feature distributions.
- Validate NSM against seconds calculated from Date_Time.
- Document anomalies before choosing cleaning strategies.
Stage 2 — Preprocessing and feature engineering (planned)
- Derive time-based features such as hour, day of week, and month from parsed timestamps.
- Evaluate whether cyclical time encodings improve validation performance.
- Handle missing numeric values inside fitted preprocessing pipelines.
- Investigate inconsistent NSM values rather than silently trusting them.
- Avoid target-derived features and preprocessing that learns from the final test period.
Stage 3 — Model training and selection (planned)
Compare a simple baseline with conventional classification models, potentially including:
- DummyClassifier (baseline)
- Logistic Regression
- Random Forest
- Histogram Gradient Boosting or XGBoost, if installed
Use chronological validation within the development period to select features, preprocessing, models, and hyperparameters.
Stage 4 — Final evaluation (planned)
Reserve December 2018, the last calendar month in the dataset, for the final holdout test. Earlier observations form the development pool. Confirm the timestamp coverage and final split counts in the audit notebook.

Do not use the final-month labels to select or tune the model. Fit all learned preprocessing operations on the appropriate training folds. Evaluate on the final holdout only after model selection is complete.
6. Evaluation metrics
Report the following for the selected model:
- Accuracy
- Precision, recall, and F1-score (macro and weighted averages)
- Per-class precision, recall, F1-score, and support
- Confusion matrix
Macro F1 is especially informative when classes are imbalanced. Where possible, compare performance with the baseline and discuss failure modes.
Results
Model	Validation macro F1	Test accuracy	Test macro F1
DummyClassifier	Pending	Pending	Pending
Logistic Regression	Pending	Pending	Pending
Random Forest	Pending	Pending	Pending
Gradient boosting	Pending	Pending	Pending


Important: Final-test metrics should be filled in only for models actually evaluated on the held-out December data. Ideally, select one model using validation performance and report its final test metrics, rather than repeatedly comparing candidates on the test set.
7. Reproducing the analysis
1. Install the dependencies as described above.
2. Confirm the CSV is at data/raw/load_data.csv.
3. Run notebooks/01_data_audit.ipynb from top to bottom.
4. Review the printed audit summary and plots.
5. After the modeling notebook is implemented, follow its instructions to reproduce training and final evaluation. Its exact filename and command will be documented here.
For reproducibility, use fixed random seeds for stochastic estimators, record package versions, and document the chronological data split. This repository does not currently claim an end-to-end trained model.
8. Limitations and considerations
- Missing measurements require an explicit handling strategy.
- NSM may be inconsistent with the timestamp and requires validation.
- Class frequencies are unequal; overall accuracy may obscure minority-class errors.
- A single final month measures performance over one specific period, not necessarily all future seasons or operating conditions.
- Any near-perfect performance should be checked for target leakage or features that encode how Load_Type was assigned.
9. Submission checklist
- [ ] Dataset audit notebook runs without errors
- [ ] EDA and data-quality findings are documented
- [ ] Preprocessing and feature engineering are reproducible
- [ ] Models are compared using chronological validation
- [ ] Last-month holdout evaluation is completed
- [ ] Classification metrics and confusion matrix are included
- [ ] README reflects the actual implemented solution and results
- [ ] Final notebook is committed to GitHub or included in a ZIP
10. Assignment deliverable
The requested deliverable is a Jupyter notebook shared through a ZIP file or a GitHub repository. This README is supporting documentation for reproducibility and clarity.