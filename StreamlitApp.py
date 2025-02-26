import streamlit as st 
import numpy as np 
import pandas as pd
import boto3
import os
import joblib

def download_from_s3(bucket_name, object_name, local_file):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    
    if not os.path.exists(local_file):  # Check if the file already exists locally
        with st.spinner("Downloading file... This may take a while."):
            s3.download_file(bucket_name, object_name, local_file)
            st.success(f"{local_file} downloaded successfully!")
    else:
        st.success(f"{local_file} already exists locally, loading it...")

# S3 bucket details for model and preprocessor
bucket_name = "patricklaptopprice"
model_object_name = "final_model/02_17_2025_13_16_38/laptop_model.pkl"
preprocessor_object_name = "final_model/02_17_2025_13_16_38/data_preprocessor.pkl"
model_local_file = "laptop_model.pkl"
preprocessor_local_file = "data_preprocessor.pkl"

# Download model and preprocessor from S3 if not already downloaded
download_from_s3(bucket_name, model_object_name, model_local_file)
download_from_s3(bucket_name, preprocessor_object_name, preprocessor_local_file)

# Load the trained CatBoost model and preprocessor
model = joblib.load(model_local_file)
preprocessor = joblib.load(preprocessor_local_file)
# Load the trained model
#model = joblib.load("/Users/sot/StreamlitTutorial/final_laptop_model.pk1")

st.markdown("## Welcome to Laptop Prediction App")
st.image("Screenshot 2025-02-26 at 16.49.04.png", caption="Laptop Price Prediction")
st.sidebar.markdown("## About App")
st.sidebar.markdown("""This web application leverages machine learning to predict the price of a laptop based on several user-defined features. It uses an XGBoost classifier to generate the predicted price based on the input data.

Features in the Application:

Status:
The user can select the laptop's status (New or Refurbished). 
                    
RAM (GB):
The user can choose the amount of RAM.
                    
Storage (GB):
 Users can specify the storage size in gigabytes,

Storage Type:
The user selects the storage type, either SSD or eMMC. 
                    
Screen Size (inches):
The size of the laptop screen is selected by the user.
                    
Touchscreen:
Users can also specify whether the laptop has a touchscreen feature""")

# User Inputs
status = st.selectbox("Status", ["New", "Refurbished"])
ram = st.slider("RAM (GB)", min_value=0, max_value=500, step=5, value=8)
storage = st.number_input("Storage (GB)", min_value=0, max_value=10000, step=5, value=256)
storage_type = st.selectbox("Storage Type", ["SSD", "eMMC"])
screen = st.slider("Screen Size (inches)", min_value=0.0, max_value=500.0, step=0.5, value=15.6)
touch = st.checkbox("Touchscreen")  

input_df = pd.DataFrame({
    "status": [status],
    "ram": [ram],
    "storage": [storage],
    "storage_type": [storage_type],
    "screen": [screen],
    "touch": [touch]
})

if st.button("Predict"):
        try:
            # Preprocess the input data using the preprocessor
            processed_input = preprocessor.transform(input_df)

            # Make prediction using the model
            prediction = model.predict(processed_input)

            # Display the result
            st.success(f"Predicted Laptop Price: ${prediction.item():,.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.error("Model or preprocessor failed to load.")
