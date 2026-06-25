# 📉 Customer Churn Analytics

> **Predicting & understanding why customers leave — using Machine Learning on the IBM Telco dataset.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)

---

## 📌 Project Overview

Customer churn is one of the most critical problems in subscription-based businesses. Losing a customer is far more expensive than retaining one. This project builds an end-to-end **churn prediction pipeline** — from raw data cleaning to a trained classification model — using the IBM Telco Customer Churn dataset.

### 🎯 Goals
- Identify key drivers of customer churn
- Build a predictive model to flag at-risk customers
- Provide actionable business insights

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Features** | 21 columns |
| **Target** | `Churn` (Yes / No) |
| **Churn Rate** | ~26.5% |

### Key Features
- **Demographics:** `gender`, `senior_citizen`, `partner`, `dependents`
- **Account Info:** `tenure`, `contract`, `paperless_billing`, `payment_method`
- **Services:** `phone_service`, `internet_service`, `online_security`, `streaming_tv`, etc.
- **Charges:** `monthly_charges`, `total_charges`

---

## 🗂️ Repository Structure

```
Customer-Churn-Analytics/
│
├── 📂 data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Raw dataset
│   └── telco_churn_cleaned.csv                 ← Cleaned dataset (output)
│
├── 📂 notebooks/
│   ├── 01_data_cleaning.ipynb                  ← Data cleaning pipeline
│   ├── 02_eda.ipynb                            ← Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb            ← Feature creation & encoding
│   └── 04_model_building.ipynb                 ← ML model training & evaluation
│
├── 📂 src/                                     ← Reusable Python scripts
├── 📂 dashboard/                               ← Visualisation dashboard
├── 📂 reports/                                 ← Output reports & figures
│
├── requirements.txt                            ← Python dependencies
└── README.md
```

---

## 🔄 Pipeline Walkthrough

```
Raw CSV  →  [01] Data Cleaning  →  [02] EDA  →  [03] Feature Engineering  →  [04] Model Building
```

### 🧹 01 — Data Cleaning [`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb)
- Fixed `TotalCharges` column (stored as `object` with 11 whitespace entries)
- Filled missing `total_charges` with `0.0` for new customers (`tenure = 0`)
- Converted `SeniorCitizen` from `0/1` → `No/Yes`
- Simplified `'No phone service'` and `'No internet service'` → `'No'`
- Encoded `Churn` target: `No → 0`, `Yes → 1`
- Standardised all column names to `snake_case`
- **Output:** `data/telco_churn_cleaned.csv` (7,043 rows, 0 missing values)

### 🔍 02 — EDA [`02_eda.ipynb`](notebooks/02_eda.ipynb)
- Univariate & bivariate analysis of all features
- Churn rate breakdown by contract type, payment method, tenure, etc.
- Correlation heatmap

### ⚙️ 03 — Feature Engineering [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)
- Label encoding & one-hot encoding of categorical features
- Feature scaling (StandardScaler)
- Feature importance analysis

### 🤖 04 — Model Building [`04_model_building.ipynb`](notebooks/04_model_building.ipynb)
- Models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Hyperparameter tuning

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.9+** | Core language |
| **Pandas & NumPy** | Data manipulation |
| **Matplotlib & Seaborn** | Visualisation |
| **Scikit-learn** | ML models & preprocessing |
| **Jupyter Notebook** | Interactive analysis |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Hetvi16-05/Customer-Churn-Analytics.git
cd Customer-Churn-Analytics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order
```bash
jupyter notebook notebooks/01_data_cleaning.ipynb
```

---

## 📈 Key Findings

> 🔄 *Results will be updated after model training is complete.*

- Customers on **Month-to-Month contracts** churn significantly more
- **Fiber optic** internet users show higher churn rates
- Customers with **shorter tenure** are more likely to churn
- **Electronic check** payment users have the highest churn rate

---

## 👩‍💻 Author

**Hetvi Sheth**  
Data Science & ML Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hetvi-sheth-4116a3346/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Hetvi16-05)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
