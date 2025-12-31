# Fraud Detection Using Machine Learning

## 📌 Project Overview
This project builds an end-to-end machine learning pipeline to detect fraudulent transactions using two datasets:
- `fraud_data.csv` (e-commerce fraud)
- `creditcard.csv` (highly imbalanced credit card fraud data)

The project covers:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Geolocation enrichment
- Data transformation
- Handling class imbalance
- Model training and evaluation

---

## 🎯 Objectives
- Identify fraudulent transactions accurately
- Handle extreme class imbalance
- Compare interpretable and ensemble models
- Select the best-performing model based on appropriate metrics

---

## 📊 Datasets
| Dataset | Target Column | Fraud Rate |
|------|-------------|-----------|
| fraud_data.csv | `class` | ~9.5% |
| creditcard.csv | `Class` | ~0.17% |

---

## 🧠 Modeling Approach

### Task 1 – Fraud_Data
- Logistic Regression (baseline)
- Random Forest
- Evaluation: AUC-PR, F1-score, Confusion Matrix

### Task 2 – Creditcard
- Logistic Regression (with & without SMOTE)
- Random Forest + SMOTE
- Extreme imbalance handled using SMOTE
- Evaluation focused on AUC-PR and Recall

---

## 🏆 Final Model Selection

| Dataset | Selected Model | Reason |
|------|---------------|--------|
| fraud_data | Random Forest | Best balance of performance & interpretability |
| creditcard | Random Forest + SMOTE | Highest AUC-PR and F1-score |

---

## 📁 Project Structure
See folder layout above. All notebooks are numbered to reflect execution order.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/bethywa/fraud-detection-model.git
cd fraud-detection-model


2️⃣ Create virtual environment
   python -m venv .venv
   .venv\Scripts\activate

3️⃣ Install dependencies
   pip install -r requirements.txt

4️⃣ Run notebooks

Open Jupyter and run notebooks in numerical order.

📌 Key Metrics Used

- AUC-PR (primary)

- F1-score

- Recall (Fraud Detection Rate)

- Confusion Matrix


---

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

📌 This covers Task 1 + Task 2 fully.