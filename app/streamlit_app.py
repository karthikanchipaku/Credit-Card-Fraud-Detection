import streamlit as st
import requests
import json
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard", 
    page_icon="🛡️", 
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("Enter transaction details below or view recent prediction logs.")

# 'backend' matches the service name in docker-compose.yml
# Change these from 'backend' to 'localhost'
PREDICT_URL = "http://localhost:8000/api/v1/predict"
HISTORY_URL = "http://localhost:8000/api/v1/history"
BATCH_URL = "http://localhost:8000/api/v1/predict_batch"
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "🗄️ Prediction History", "📁 Batch Prediction"])

with tab1:
    st.subheader("Transaction Data")
    
    sample_data = {
        "Time": 0.0, "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
        "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986, "V9": 0.3637,
        "V10": 0.0907, "V11": -0.5515, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111,
        "V15": 1.4681, "V16": -0.4704, "V17": 0.2079, "V18": 0.0257, "V19": 0.4039,
        "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210, "Amount": 149.62
    }

    input_json = st.text_area(
        "Input JSON (30 features):", 
        value=json.dumps(sample_data, indent=4), 
        height=300
    )

    if st.button("Predict Fraud"):
        try:
            data = json.loads(input_json)
            with st.spinner('Querying API...'):
                response = requests.post(PREDICT_URL, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("prediction")
                    
                    st.markdown("---")
                    if result.get("is_fraud"):
                        st.error(f"🚨 **ALERT**: This transaction is classified as **{status}**!")
                    else:
                        st.success(f"✅ **SAFE**: This transaction is classified as **{status}**.")
                    
                    # --- NEW: SHAP Explainability ---
                    st.markdown("### 🧠 Model Explainability (SHAP)")
                    st.markdown("This chart explains *why* the model made its decision. Red features pushed the score higher (toward fraud), and blue features pushed it lower (toward legitimate).")
                    
                    try:
                        # Load artifacts
                        model_path = os.path.join("artifacts", "models", "model.pkl")
                        preprocessor_path = os.path.join("artifacts", "models", "preprocessor.pkl")
                        
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                        with open(preprocessor_path, "rb") as f:
                            preprocessor = pickle.load(f)
                            
                        # Transform input and calculate SHAP
                        df_input = pd.DataFrame([data])
                        df_scaled = preprocessor.transform(df_input)
                        
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer(df_scaled)
                        
                        # Generate and display plot
                        fig, ax = plt.subplots(figsize=(10, 4))
                        # We use shap_values[0] because we are only explaining this single prediction
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {e}")
                        
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your syntax.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is your FastAPI server running on port 8000?")

with tab2:
    st.subheader("Recent Database Logs")
    st.markdown("Fetch the latest transaction records directly from the SQLite database.")
    
    if st.button("🔄 Refresh History"):
        try:
            with st.spinner('Fetching records...'):
                response = requests.get(HISTORY_URL)
                
                if response.status_code == 200:
                    history_data = response.json().get("history", [])
                    if history_data:
                        df = pd.DataFrame(history_data)
                        df = df[['id', 'timestamp', 'amount', 'prediction_result', 'is_fraud']]
                        st.dataframe(df, width='stretch')
                    else:
                        st.info("No predictions found in the database yet.")
                else:
                    st.error(f"API Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API.")

with tab3:
    st.subheader("Batch Prediction via CSV")
    st.markdown("Upload a CSV file containing multiple transactions. The system will predict all of them at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Process Batch"):
            try:
                with st.spinner('Uploading and processing file... This may take a moment for large files.'):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(BATCH_URL, files=files)
                    
                    if response.status_code == 200:
                        batch_result = response.json()
                        st.success(f"✅ Successfully processed {batch_result['total_processed']} transactions!")
                        
                        result_df = pd.DataFrame(batch_result['results'])
                        cols = ['Prediction'] + [col for col in result_df.columns if col != 'Prediction']
                        result_df = result_df[cols]
                        
                        st.dataframe(result_df, width='stretch')
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API.")