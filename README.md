❤️ Heart Disease Prediction App

A Machine Learning powered Streamlit app that predicts the risk of Heart Disease based on patient health parameters.

📌 Features

Predicts the probability of having heart disease
Takes inputs like age, sex, chest pain type, blood pressure, cholesterol, etc.
Uses a Random Forest Classifier trained on the UCI Heart Disease Dataset
Provides a clear risk prediction with probability
User-friendly web interface built with Streamlit

🚀 How to Run Locally
git clone https://github.com/mushrifa-hussain/Heart-Disease-Prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app.py


The app will open in your browser at http://localhost:8501🎉

🧑‍⚕️ Input Parameters
Age: Patient's age (years)
Sex: Male / Female
Chest Pain Type: typical angina, atypical angina, non-anginal, asymptomatic
Resting Blood Pressure (mm Hg)
Cholesterol (mg/dl)
Max Heart Rate Achieved
Exercise Induced Angina: Yes / No

📂 Project Structure
heart-disease-prediction/
│── app.py                # Streamlit app
│── heart_rf_model.pkl    # Trained Random Forest model
│── heart_scaler.pkl      # StandardScaler used for preprocessing
│── X_columns.pkl         # Feature columns from training
│── requirements.txt      # Dependencies
│── README.md             # Project description

📊 Model Information
Algorithm: Random Forest Classifier
Libraries: scikit-learn, pandas, matplotlib, seaborn 
Accuracy: ~88% on test data

🌐 Deployment

This app can be easily deployed for free on Streamlit Community Cloud or platforms like Heroku / AWS / GCP.

⚠️ Disclaimer

This project is created for educational purposes only and should not be used as a substitute for professional medical advice.
