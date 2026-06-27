# 🚀 Customer Churn Analytics: Beyond the Baseline

> **Transforming raw telco data into actionable retention strategies using advanced Machine Learning and creative feature engineering.**

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Status-Innovating%20%E2%9C%A8-success?style=for-the-badge" alt="Status" />
</div>

---

## 🌟 The Vision

Customer churn is the silent killer of subscription-based businesses. While standard models simply predict *who* will leave, this project takes a more **innovative approach**: we focus on the *why* and the *what's next*. By building an end-to-end analytical pipeline, we translate complex data into direct business value.

### 🎯 Strategic Goals
- **Decode the DNA of Churn:** Uncover hidden patterns and non-obvious drivers of customer attrition.
- **Predictive Power:** Deploy robust machine learning models to flag at-risk customers before they hit 'cancel'.
- **Actionable Intelligence:** Bridge the gap between data science and business strategy with intuitive visualizations and actionable recommendations.

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

## ✨ Creative & Innovative Approaches

What sets this project apart from standard churn analyses?

- **Behavioral Feature Engineering:** Moving beyond standard metrics to create derived features that capture customer behavioral shifts over time.
- **Explainable AI (XAI):** We don't just want a black-box prediction. We aim to use SHAP/LIME to explain *exactly* why a specific customer is flagged, empowering customer success teams with context.
- **Interactive Storytelling:** The `dashboard/` directory isn't just for static charts; it's designed to tell a compelling story to non-technical stakeholders, making data accessible and actionable.
- **Holistic Pipeline:** A clean, modular architecture separating raw data, transformations, and final insights, ensuring reproducibility and scalability.

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

### 🧠 03 — Creative Feature Engineering [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)
- **Transformative Encoding:** Label & one-hot encoding optimized for diverse model architectures.
- **Advanced Scaling:** Robust feature scaling ensuring models don't just learn, but learn *well*.
- **Signal Extraction:** Deep dive into feature importance to separate the signal from the noise, creating custom behavioral indicators.

### 🤖 04 — Advanced Model Building [`04_model_building.ipynb`](notebooks/04_model_building.ipynb)
- **Algorithm Arsenal:** Deploying a spectrum of models from interpretable Logistic Regression to powerful ensemble methods (Random Forest, XGBoost).
- **Rigorous Evaluation:** Looking beyond accuracy—focusing on Precision, Recall, F1-Score, and ROC-AUC to balance business trade-offs.
- **Hyperparameter Optimization:** Fine-tuning models to squeeze out every drop of predictive performance.

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
