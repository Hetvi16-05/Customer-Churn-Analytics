import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.preprocessing import RobustScaler

# Define paths for model artifacts (assuming they are saved in a models/ directory)
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODING_COLS_PATH = os.path.join(MODELS_DIR, 'encoding_cols.pkl')

def bin_tenure(tenure):
    """Helper function to bin tenure values exactly as done in training."""
    if tenure <= 12: return '0-1 Year'
    elif tenure <= 24: return '1-2 Years'
    elif tenure <= 36: return '2-3 Years'
    elif tenure <= 48: return '3-4 Years'
    elif tenure <= 60: return '4-5 Years'
    else: return '5+ Years'

def preprocess_new_data(df, is_training=False):
    """
    Applies the same feature engineering steps as the 03_feature_engineering notebook.
    """
    df = df.copy()
    
    # 1. Behavioral Features
    if 'tenure' in df.columns:
        df['tenure_group'] = df['tenure'].apply(bin_tenure)
        
    services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
    # Check which services actually exist in df
    available_services = [s for s in services if s in df.columns]
    if available_services:
        df['num_additional_services'] = df[available_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
        
    if 'total_charges' in df.columns and 'tenure' in df.columns:
        # Convert total_charges to numeric just in case
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0.0)
        df['avg_charge_per_month'] = df['total_charges'] / (df['tenure'] + 1)
        
    if 'monthly_charges' in df.columns:
        # Approximate 75th percentile from IBM dataset as high value threshold
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
        # Add missing columns with 0
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        # Reorder and keep only expected columns
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

def predict_churn(customer_data):
    """
    Predicts churn for new customer data.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save the model (along with scaler and encoding columns) first.")
        
    print("Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    
    print("Preprocessing data...")
    processed_data = preprocess_new_data(customer_data, is_training=False)
    
    print("Generating predictions...")
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    
    return predictions, probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Customer Churn using trained ML model")
    parser.add_argument("--input", type=str, help="Path to input CSV file containing raw customer data")
    parser.add_argument("--output", type=str, help="Path to save predictions CSV", default="predictions.csv")
    
    args = parser.parse_args()
    
    if args.input:
        print(f"Loading data from {args.input}...")
        try:
            df = pd.read_csv(args.input)
            
            preds, probs = predict_churn(df)
            df['churn_prediction'] = preds
            df['churn_probability'] = np.round(probs, 4)
            
            df.to_csv(args.output, index=False)
            print(f"✅ Success! Predictions saved to {args.output}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Note: You must export your trained model (e.g., using joblib) from the training notebooks into a `models/` directory first!")
        except Exception as e:
            print(f"❌ An unexpected error occurred: {e}")
    else:
        print("Customer Churn Prediction Module")
        print("--------------------------------")
        print("Usage:")
        print("  Command line: python src/predict.py --input data/new_customers.csv --output data/predictions.csv")
        print("  In Python:    from src.predict import predict_churn; preds, probs = predict_churn(df)")
