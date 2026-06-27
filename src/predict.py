import pandas as pd
import numpy as np
import joblib
import os
import argparse
from preprocess import preprocess_pipeline

# Define paths for model artifacts
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models')
MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')

def predict_churn(customer_data):
    """
    Predicts churn for new customer data using the trained model and preprocessing pipeline.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train and save the model (along with scaler and encoding columns) first.")
        
    print("Loading model artifacts...")
    model = joblib.load(MODEL_PATH)
    
    print("Preprocessing data...")
    # Run the full pipeline (cleaning + feature engineering)
    processed_data = preprocess_pipeline(customer_data, is_training=False)
    
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
