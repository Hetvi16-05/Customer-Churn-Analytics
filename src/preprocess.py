import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler

# Define paths for model artifacts
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODING_COLS_PATH = os.path.join(MODELS_DIR, 'encoding_cols.pkl')

def clean_data(df):
    """
    Applies basic data cleaning steps, mirroring the 01_data_cleaning notebook.
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Fix TotalCharges
    if 'totalcharges' in df.columns or 'total_charges' in df.columns:
        tc_col = 'total_charges' if 'total_charges' in df.columns else 'totalcharges'
        df[tc_col] = pd.to_numeric(df[tc_col], errors='coerce').fillna(0.0)
        if tc_col == 'totalcharges':
            df.rename(columns={'totalcharges': 'total_charges'}, inplace=True)
            
    # Convert SeniorCitizen
    if 'seniorcitizen' in df.columns or 'senior_citizen' in df.columns:
        sc_col = 'senior_citizen' if 'senior_citizen' in df.columns else 'seniorcitizen'
        df[sc_col] = df[sc_col].map({0: 'No', 1: 'Yes', '0': 'No', '1': 'Yes'}).fillna(df[sc_col])
        if sc_col == 'seniorcitizen':
            df.rename(columns={'seniorcitizen': 'senior_citizen'}, inplace=True)
            
    # Simplify categories
    cols_to_simplify = ['multiple_lines', 'online_security', 'online_backup', 
                        'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    for col in cols_to_simplify:
        if col in df.columns:
            df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})
            
    # Encode Churn if exists (training mode)
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'No': 0, 'Yes': 1, 0: 0, 1: 1})
        
    return df

def bin_tenure(tenure):
    """Helper function to bin tenure values."""
    if tenure <= 12: return '0-1 Year'
    elif tenure <= 24: return '1-2 Years'
    elif tenure <= 36: return '2-3 Years'
    elif tenure <= 48: return '3-4 Years'
    elif tenure <= 60: return '4-5 Years'
    else: return '5+ Years'

def feature_engineer(df, is_training=False):
    """
    Applies feature engineering, encoding, and scaling.
    """
    df = df.copy()
    
    # 1. Behavioral Features
    if 'tenure' in df.columns:
        df['tenure_group'] = df['tenure'].apply(bin_tenure)
        
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    available_services = [s for s in services if s in df.columns]
    if available_services:
        df['num_additional_services'] = df[available_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
        
    if 'total_charges' in df.columns and 'tenure' in df.columns:
        df['avg_charge_per_month'] = df['total_charges'] / (df['tenure'] + 1)
        
    if 'monthly_charges' in df.columns:
        # Approximate 75th percentile from IBM dataset
        df['is_high_value'] = (df['monthly_charges'] > 89.85).astype(int)
        
    # Drop ID
    if 'customer_id' in df.columns:
        df.drop('customer_id', axis=1, inplace=True)
        
    # Categorical Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'churn' in categorical_cols:
        categorical_cols.remove('churn')
        
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Align columns with training data if artifacts exist
    if not is_training and os.path.exists(ENCODING_COLS_PATH):
        expected_cols = joblib.load(ENCODING_COLS_PATH)
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_cols]
        
    # Scaling
    cols_to_scale = ['tenure', 'monthly_charges', 'total_charges', 'avg_charge_per_month', 'num_additional_services']
    available_scale_cols = [c for c in cols_to_scale if c in df_encoded.columns]
    
    if is_training:
        scaler = RobustScaler()
        df_encoded[available_scale_cols] = scaler.fit_transform(df_encoded[available_scale_cols])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(df_encoded.columns.tolist(), ENCODING_COLS_PATH)
    else:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            df_encoded[available_scale_cols] = scaler.transform(df_encoded[available_scale_cols])
            
    return df_encoded

def preprocess_pipeline(df, is_training=False):
    """
    Runs the full preprocessing pipeline: basic data cleaning followed by feature engineering.
    """
    df_clean = clean_data(df)
    df_engineered = feature_engineer(df_clean, is_training=is_training)
    return df_engineered
