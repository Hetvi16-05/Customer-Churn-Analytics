import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Add src directory to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from predict import predict_churn
except ImportError:
    st.error("Could not import prediction module. Make sure src/predict.py exists.")

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="🚀",
    layout="wide"
)

# Constants
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Basic cleanup for EDA display
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
        return df
    return None

def main():
    st.title("🚀 Customer Churn Analytics Dashboard")
    st.markdown("Transforming raw telco data into actionable retention strategies.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview & EDA", "👤 Single Customer Prediction", "📂 Batch Prediction"])
    
    with tab1:
        st.header("Dataset Overview")
        df = load_data()
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", f"{len(df):,}")
            churn_rate = (df['Churn'] == 'Yes').mean() * 100
            col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
            col3.metric("Avg Monthly Charge", f"${df['MonthlyCharges'].mean():.2f}")
            col4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig1 = px.histogram(df, x="Contract", color="Churn", barmode="group",
                                    title="Churn by Contract Type",
                                    color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
                st.plotly_chart(fig1, use_container_width=True)
                
            with col_b:
                fig2 = px.histogram(df, x="InternetService", color="Churn", barmode="group",
                                    title="Churn by Internet Service",
                                    color_discrete_map={"Yes": "#EF553B", "No": "#00CC96"})
                st.plotly_chart(fig2, use_container_width=True)
                
            st.markdown("### Raw Data Sample")
            st.dataframe(df.head(10))
        else:
            st.warning("Raw dataset not found at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`")
            
    with tab2:
        st.header("Single Customer Prediction")
        st.markdown("Enter customer details below to predict their likelihood of churning.")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.selectbox("Senior Citizen", ["No", "Yes"]) # will convert to 0/1 in dict
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                
            with col2:
                st.subheader("Services")
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_bak = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_pro = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                stream_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                
            with col3:
                st.subheader("Account Info")
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, format="%.2f")
                total = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, format="%.2f")
                
            submit = st.form_submit_button("Predict Churn")
            
        if submit:
            customer_dict = {
                'customerID': '0000-NEWCUST',
                'gender': gender,
                'SeniorCitizen': 1 if senior == "Yes" else 0,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone,
                'MultipleLines': multiple,
                'InternetService': internet,
                'OnlineSecurity': online_sec,
                'OnlineBackup': online_bak,
                'DeviceProtection': device_pro,
                'TechSupport': tech_sup,
                'StreamingTV': stream_tv,
                'StreamingMovies': stream_mov,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment,
                'MonthlyCharges': monthly,
                'TotalCharges': total
            }
            input_df = pd.DataFrame([customer_dict])
            
            try:
                preds, probs = predict_churn(input_df)
                churn_prob = probs[0]
                prediction = "Likely to Churn" if preds[0] == 1 else "Likely to Stay"
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if preds[0] == 1:
                        st.error(f"Prediction: {prediction}")
                    else:
                        st.success(f"Prediction: {prediction}")
                with res_col2:
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Have you trained the model yet? Ensure models/best_model.pkl exists.")
                
    with tab3:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file of new customers to predict churn for multiple records at once.")
        
        uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing data and generating predictions..."):
                    try:
                        preds, probs = predict_churn(batch_df)
                        result_df = batch_df.copy()
                        result_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in preds]
                        result_df['Churn_Probability'] = [f"{prob:.1%}" for prob in probs]
                        
                        st.success("Predictions generated successfully!")
                        st.dataframe(result_df)
                        
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Error during batch prediction: {e}")

if __name__ == "__main__":
    main()
