import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
best_model = joblib.load('C:/Users/HP/Downloads/AI Engineer Assignment/AI Engineer Assignment/best_model.pkl')
scaler = joblib.load('C:/Users/HP/Downloads/AI Engineer Assignment/AI Engineer Assignment/scaler.pkl')

# Sensors used
final_sensors = ['sensor_00', 'sensor_04', 'sensor_06', 'sensor_07',
                 'sensor_08', 'sensor_09', 'sensor_10', 'sensor_11',
                 'sensor_12']

def get_new_features(sensor_list, df):
    new_features = {}
    for sensor in sensor_list:
        val = df[sensor] - np.mean(df[sensor])
        new_features[sensor] = val
    return pd.DataFrame(new_features)

def predict_machine_status(user_df):
    missing_cols = set(final_sensors) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0
    user_df = user_df[final_sensors]
    user_df = get_new_features(final_sensors, user_df)
    user_df_scaled = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)
    predictions = best_model.predict(user_df_scaled)
    return ['NORMAL' if pred == 0 else 'BROKEN' for pred in predictions]

# --- Streamlit UI ---
st.title("🛠️ Machine Status Prediction App")
option = st.radio("Choose Input Method", ["Single Data Point", "Upload Multiple Data Points (CSV)"])

if option == "Single Data Point":
    st.subheader("📌 Enter Sensor Values")
    user_input = {}
    for sensor in final_sensors:
        user_input[sensor] = st.number_input(f"{sensor}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        result = predict_machine_status(input_df)[0]
        st.markdown(f"### 🔍 Predicted Machine Status: **{result}**")

elif option == "Upload Multiple Data Points (CSV)":
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with sensor columns", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📊 Uploaded Data Preview:")
            st.dataframe(df.head())

            if st.button("Predict for Uploaded Data"):
                results = predict_machine_status(df)
                df['Prediction'] = results
                st.success("✅ Prediction Completed!")
                st.dataframe(df)

                # Download option
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
