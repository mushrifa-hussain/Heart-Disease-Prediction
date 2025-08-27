import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and training feature columns
model = joblib.load("heart_rf_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
X_columns = joblib.load("X_columns.pkl")   # list of feature names from training

st.title("❤️ Heart Disease Prediction App")

# --- User Inputs ---
age = st.number_input("Age", value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox(
    "Chest Pain Type",
    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Cholesterol (mg/dl)", value=200)
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

# --- Convert inputs to dataframe ---
input_dict = {
    "age": age,
    "sex": sex,               # keep as string for encoding
    "cp": cp,                 # keep as string for encoding
    "trestbps": trestbps,
    "chol": chol,
    "thalach": thalach,
    "exang": exang            # keep as string for encoding
}
input_df = pd.DataFrame([input_dict])

# --- One-hot encode categorical columns (same as training) ---
cat_cols = ["sex", "cp", "exang"]
input_encoded = pd.get_dummies(input_df, columns=cat_cols)

# --- Align with training columns ---
input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

# --- Scale numeric features ---
input_scaled = scaler.transform(input_encoded)

# --- Prediction ---
if st.button("Predict"):

    # ⚠️ Safety checks (realistic medical ranges)
    if age > 90 or trestbps > 200 or chol > 400 or thalach > 210:
        st.warning("⚠️ Warning: Entered values are outside typical medical ranges. "
                   "Prediction may not be reliable.")

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100  # probability of class 1

    if prediction == 1:
        st.error(f"⚠️ High chance of Heart Disease\n\n🩺 Probability: **{probability:.2f}%**")
    else:
        st.success(f"✅ No Heart Disease Risk\n\n🩺 Probability: **{probability:.2f}%**")

st.caption("ℹ️ Disclaimer: This app is for educational purposes only and not a substitute for professional medical advice.")
