# 🏥 AI-Based Healthcare Data Analysis

> **Stack:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn  
> **Domain:** Predictive Modelling · Healthcare · Responsible AI  
> **Models:** Logistic Regression · Random Forest  
> **Best AUC:** 0.81 (Logistic Regression, 5-fold CV)

---

## Project Overview

This project builds a **diabetes risk classification pipeline** on a synthetic healthcare dataset modelled after the Pima Indians Diabetes dataset (UCI / Kaggle). The focus is not just on achieving high benchmark accuracy, but on understanding **where and why** models fail — the kind of critical thinking required for any real-world clinical deployment.

---

## Project Structure

```
healthcare_project/
├── data/
│   └── healthcare_raw.csv          # Synthetic dataset (2000 patients, 9 features)
├── src/
│   └── data_generator.py           # Reproducible dataset generator
├── visuals/
│   ├── 01_distributions.png        # Feature distributions by outcome
│   ├── 02_correlation.png          # Correlation heatmap
│   ├── 03_missing_map.png          # Missing value visualisation
│   ├── 04_roc_curves.png           # ROC curves for both models
│   ├── 05_confusion_matrices.png   # Confusion matrices
│   ├── 06_feature_importance.png   # Random Forest feature importances
│   └── 07_bias_audit.png           # Performance by age group (fairness)
├── reports/
│   └── reflections.txt             # Critical analysis report
├── analysis_pipeline.py            # Full end-to-end pipeline
└── README.md
```

---

## Pipeline Steps

### 1. Data Generation & Loading
- 2,000 synthetic patient records with 8 clinical features
- Realistic missing value patterns: 34% Insulin, 28% SkinThickness
- Label generated via logistic relationship with key features

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis by diabetes outcome
- Correlation matrix to identify feature collinearity
- Missing value map

### 3. Preprocessing
- Median imputation for missing values (within sklearn.Pipeline)
- StandardScaler for logistic regression
- Stratified train/test split (80/20)

### 4. Modelling

| Model               | CV ROC-AUC     | Test ROC-AUC |
|---------------------|----------------|--------------|
| Logistic Regression | 0.787 ± 0.028  | **0.813**    |
| Random Forest       | 0.778 ± 0.034  | 0.779        |

- Both models use class_weight="balanced" to handle class imbalance
- 5-fold stratified cross-validation prevents data leakage

### 5. Evaluation
- Confusion matrices, classification reports (precision/recall/F1)
- ROC-AUC as primary metric (appropriate for imbalanced data)
- Feature importance analysis

### 6. Fairness / Bias Audit
- Performance evaluated across 4 age groups (<30, 30-45, 45-60, 60+)
- Reveals accuracy disparity and diabetes prevalence differences
- Youngest group underrepresented -> hidden fairness risk

---

## Key Findings & Critical Reflections

### Where the model breaks down
1. Missing data — 35% of insulin values missing; median imputation is a simplifying assumption
2. Class imbalance — without class_weight="balanced", recall on diabetics drops ~10pp
3. Calibration gap — AUC ≠ calibrated probabilities; Platt scaling needed for clinical use
4. Subgroup disparities — younger patients underrepresented, lower performance in <30 group

### Benchmark vs. Real-World Gap
The benchmark assumes i.i.d. train/test splits. Clinical deployment requires prospective validation, EHR integration (HL7/FHIR), regulatory clearance, and ongoing drift monitoring.

### Ethical Considerations
- Models can encode historical inequalities in training data
- Informed patient consent required for algorithmic screening
- Define clinical cost matrix (FP vs FN) before choosing optimisation target

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python analysis_pipeline.py
```

---

*Built as a senior-level demonstration project — emphasising critical thinking over metric chasing.*
