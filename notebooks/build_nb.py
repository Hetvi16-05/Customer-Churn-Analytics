import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """# 🧠 03 - Creative Feature Engineering

In this notebook, we move beyond basic preprocessing to construct **innovative behavioral features**, apply **transformative encoding** and **robust scaling**, and use a baseline model to **extract feature importance**.

Let's transform our clean data into a high-signal dataset ready for advanced modeling!"""

code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
import warnings
warnings.filterwarnings('ignore')"""

text_2 = """## 1. Load Cleaned Dataset"""

code_2 = """df = pd.read_csv('../data/telco_churn_cleaned.csv')
print(f"Dataset shape: {df.shape}")
df.head()"""

text_3 = """## 2. Behavioral Feature Engineering

Instead of just using `tenure` or `monthly_charges`, let's create derived features that capture the **behavior** and **value** of a customer over time."""

code_3 = """# 1. Tenure Groups (Binning)
def bin_tenure(tenure):
    if tenure <= 12: return '0-1 Year'
    elif tenure <= 24: return '1-2 Years'
    elif tenure <= 36: return '2-3 Years'
    elif tenure <= 48: return '3-4 Years'
    elif tenure <= 60: return '4-5 Years'
    else: return '5+ Years'

df['tenure_group'] = df['tenure'].apply(bin_tenure)

# 2. Number of Additional Services
# Customers with more services might be more 'locked in'
services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
df['num_additional_services'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)

# 3. Average Charge Per Tenure Month
# This compares their actual total to what we expect, helping find anomalies or discount users.
# Adding 1 to tenure to avoid division by zero for new customers
df['avg_charge_per_month'] = df['total_charges'] / (df['tenure'] + 1)

# 4. High Value Customer Flag
# Top 25% of monthly charges are considered high value
high_val_threshold = df['monthly_charges'].quantile(0.75)
df['is_high_value'] = (df['monthly_charges'] > high_val_threshold).astype(int)

df[['tenure', 'tenure_group', 'num_additional_services', 'monthly_charges', 'avg_charge_per_month', 'is_high_value']].head()"""

text_4 = """## 3. Transformative Encoding

We have a mix of binary and multi-class categorical variables. We will:
- **Drop** `customer_id` (not useful for prediction).
- **One-Hot Encode** multi-class categorical variables."""

code_4 = """# Drop ID
if 'customer_id' in df.columns:
    df.drop('customer_id', axis=1, inplace=True)

# Identify columns
target = 'churn'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove(target) # target is already 0/1 numeric
if 'is_high_value' in numerical_cols:
    numerical_cols.remove('is_high_value') # keep as binary

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print(f"Shape after encoding: {df_encoded.shape}")
df_encoded.head()"""

text_5 = """## 4. Advanced Scaling

We use **RobustScaler** on our continuous numerical features (`tenure`, `monthly_charges`, `total_charges`, `avg_charge_per_month`, `num_additional_services`). RobustScaler uses statistics that are robust to outliers (median and IQR)."""

code_5 = """scaler = RobustScaler()

cols_to_scale = ['tenure', 'monthly_charges', 'total_charges', 'avg_charge_per_month', 'num_additional_services']
df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])

df_encoded[cols_to_scale].head()"""

text_6 = """## 5. Signal Extraction (Feature Importance)

Before we move to full model building, let's train a quick Random Forest to see which features provide the most **signal**."""

code_6 = """X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# Quick RF model
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
rf.fit(X, y)

# Get Feature Importances
importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title('Top 15 Most Important Features (Random Forest Signal Extraction)')
plt.tight_layout()
plt.show()"""

text_7 = """## 6. Save Engineered Dataset

We now have a fully engineered, encoded, and scaled dataset ready for rigorous model building!"""

code_7 = """output_path = '../data/telco_churn_engineered.csv'
df_encoded.to_csv(output_path, index=False)
print(f"Engineered dataset saved to: {output_path}")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(text_7),
    nbf.v4.new_code_cell(code_7)
]

with open('/Users/HetviSheth/Library/CloudStorage/OneDrive-NavrachanaUniversity/Customer-Churn-Analytics/notebooks/03_feature_engineering.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 03_feature_engineering.ipynb successfully generated!")
