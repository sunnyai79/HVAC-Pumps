# 🛠️ HVAC Pumps Prediction App

This project is a **Streamlit-based web application** that predicts machine status (NORMAL or BROKEN) based on sensor readings using a pre-trained machine learning model. It allows users to input single data points or upload batch CSV files for bulk prediction.

---

## 🔗 Dataset Source

The model was trained using sensor data available on Kaggle:

**[📂 View the pump sensor data]([https://www.kaggle.com/your-dataset-link](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data))**  

---

## 🚀 Deployment & Running the App

### 🔧 Requirements

Install the necessary Python libraries with:

```bash
pip install -r requirements.txt
```

### ▶️ Running the App

Once dependencies are installed, launch the app locally using:

```bash
streamlit run app.py
```

Ensure the following files are in the same directory:
- `app.py`
- `best_model.pkl`
- `scaler.pkl`

---

## 🌐 Accessing the API Endpoint

This app is built with Streamlit and does not expose a separate REST API by default. However, users can interact with it through the browser UI:

### Features:
- **Single Data Point Prediction:** Enter individual sensor values.
- **Batch Upload:** Upload CSV files with the required sensors and download results.

To access:
1. Run the app locally.
2. Navigate to `http://localhost:8501` in your browser.
3. Choose your input method and proceed with predictions.

---

## 📊 Model Summary & Insights

- **Model Type**: Pre-trained scikit-learn model (loaded via `joblib`)
- **Features Used**:  
  `sensor_00`, `sensor_04`, `sensor_06`, `sensor_07`, `sensor_08`, `sensor_09`, `sensor_10`, `sensor_11`, `sensor_12`
- **Feature Engineering**: For each sensor, values are mean-adjusted to capture deviations.

### ✅ Summary of Findings
- The trained model performs binary classification of machine states with reliable accuracy on validation data.
- Feature normalization and careful sensor selection improved prediction robustness.

### 🔧 Actionable Recommendations
- **Monitor Key Sensors**: Focus on the 9 listed sensors for preventive maintenance.
- **Periodic Data Recalibration**: Regularly retrain the model with new operational data to maintain accuracy.
- **Integrate into Monitoring Pipelines**: Consider converting the app into an API or dashboard for real-time factory use.

---

## 📁 File Structure

```
.
├── app.py                 # Streamlit application code
├── best_model.pkl         # Trained ML model
├── scaler.pkl             # Pre-fitted scaler for input normalization
├── requirements.txt       # Python dependencies
└── create_app.ipynb              # (Optional) Jupyter notebook with code exploration (if needed)
```

---

## 📬 Contact

For questions or collaboration opportunities, please reach out via GitHub issues or pull requests.
