import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from preprocess import preprocess_pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    from xgboost import XGBClassifier

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')

def train_model(data_path=RAW_DATA_PATH):
    """
    Trains the XGBoost churn prediction model from raw data and saves all artifacts.
    """
    print(f"Loading raw data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    print("Splitting data into train and test sets...")
    # Stratify split requires the target variable. Since it's 'Churn' in raw data:
    target_col = 'Churn' if 'Churn' in df.columns else 'churn'
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    print(f"Training set: {train_df.shape[0]} rows | Test set: {test_df.shape[0]} rows")
    
    print("Preprocessing training data (fitting scalers & saving encoding artifacts)...")
    train_processed = preprocess_pipeline(train_df, is_training=True)
    
    print("Preprocessing test data...")
    test_processed = preprocess_pipeline(test_df, is_training=False)
    
    # Separate features and target
    X_train = train_processed.drop('churn', axis=1)
    y_train = train_processed['churn']
    
    X_test = test_processed.drop('churn', axis=1)
    y_test = test_processed['churn']
    
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42, 
        learning_rate=0.05, 
        max_depth=5,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    
    print("\n--- Model Evaluation (Test Set) ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Saving trained model to {MODEL_PATH}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    print("✅ Training complete! Model and preprocessing artifacts successfully saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Customer Churn ML Model")
    parser.add_argument("--data", type=str, help="Path to raw CSV dataset", default=RAW_DATA_PATH)
    
    args = parser.parse_args()
    
    try:
        train_model(args.data)
    except Exception as e:
        print(f"❌ Error during training: {e}")
