# 🕵️ Credit Card Fraud Detection Project
 ## Project Overview
 
This project focuses on building, evaluating, and explaining machine learning models for credit card fraud detection using transactional data. Due to the highly imbalanced nature of fraud data, the project emphasizes robust evaluation metrics, careful preprocessing, and model explainability using SHAP.

## The project is divided into three main tasks:

1. Exploratory Data Analysis (EDA) & Preprocessing

2. Model Development & Evaluation

3. Model Explainability & Business Insights (SHAP)

  ## Dataset

The dataset contains anonymized credit card transactions with:

284,807 transactions

31 features (Time, Amount, anonymized PCA features V1–V28)

Target variable: Class

0 → Legitimate transaction

1 → Fraudulent transaction

Fraud cases represent ~0.17% of the data, making this a highly imbalanced classification problem.

## Project Structure
 fraud-detection-model/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned / transformed data
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_shap_explainability.ipynb
│
├── models/
│   └── random_forest.pkl    # Saved trained model
│
├── reports/
│   └── figures/             # Plots & visualizations
│
├── requirements.txt
├── README.md
└── .gitignore




## ✅ Task 1: Exploratory Data Analysis & Preprocessing
Objective

### Understand the data, identify patterns, detect anomalies, and prepare the dataset for modeling.

### Key Steps

1. Checked class imbalance and fraud distribution

2. Analyzed transaction amount and time features

3. Created derived features (e.g., Time_hours)

4. Scaled numerical features where necessary

5. Separated features (X) and target (y)

### Key Insights

Fraud transactions tend to occur at unusual times

Fraud amounts show different distribution patterns

Severe class imbalance requires careful evaluation strategy

## ✅ Task 2: Model Development & Evaluation

### Objective

Train and evaluate machine learning models capable of detecting fraud effectively while minimizing false negatives.

Models Trained

  Logistic Regression (baseline)
  Random Forest Classifier (final model)

Evaluation Metrics

  Due to class imbalance, accuracy alone is insufficient. I used:

    Precision

    Recall

    F1-score

    ROC-AUC

Final Model Selection

The Random Forest classifier was selected as the best model because it:

  Achieved higher recall for fraud cases
  Maintained strong ROC-AUC performance
  Captured nonlinear feature interactions effectively

## ✅ Task 3: Model Explainability (SHAP)
Objective

Interpret model predictions using SHAP to understand what drives fraud detection and generate actionable business recommendations.

1. Built-in Feature Importance

Random Forest feature importance highlighted:

1. V14
2. V10
3. V4
4. V12
5. V17

These features are most frequently used by the model during decision-making.

2. SHAP Global Explainability

SHAP values were computed on a sample of 2,000 test transactions for efficiency.

Top SHAP Drivers of Fraud:

1. V4
2. V14
3. V12
4. V7
5. V17

SHAP reveals how strongly each feature pushes predictions toward fraud or legitimacy, making it more interpretable than built-in importance alone.

3. SHAP Local Explainability

SHAP force plots were generated for:

True Positive (TP): Correctly detected fraud

False Positive (FP): Legitimate transaction flagged as fraud

False Negative (FN): Missed fraud case

### Observations

True positives show strong positive contributions from key features (V4, V14)

False positives are driven by isolated high-risk signals

False negatives occur when fraud patterns are weak or mixed

# Business Recommendations

* Enhanced Verification
* Transactions with extreme values in V4 or V14 should trigger additional authentication.
* Adaptive Thresholds
* Introduce flexible decision thresholds to reduce false positives without sacrificing fraud recall.

## Feature Combination Alerts

* Prioritize monitoring transactions where multiple high-impact features appear together.

## Conclusion

This project demonstrates a complete end-to-end fraud detection pipeline:

  * Robust preprocessing
  * Careful model selection for imbalanced data
  * Transparent explainability using SHAP
  * Clear business-oriented insights
Explainable AI techniques bridge the gap between machine learning performance and real-world decision-making in financial systems.

# Environment Setup

 1. Clone Repository
  
 git clone https://github.com/bethywa/fraud-detection-model.
 
 2. Create Virtual Environment
   python -m venv .venv
   .venv\Scripts\activate

3️⃣ Install dependencies

    pip install -r requirements.txt


## 📦 requirements.txt (CLEAN & COMPLETE)

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
joblib
jupyter
ipykernel
shap


  
  Author
 Bethelihem
Machine Learning & Data Science Project

Fraud Detection & Explainable AI
