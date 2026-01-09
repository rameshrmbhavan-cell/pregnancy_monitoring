import streamlit as st
import pandas as pd
import  pickle
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fetal Health Dashboard", layout="wide")
st.title("🩺 Fetal Health Classification AI")
st.write("Adjust the values on the left to predict the health status of the fetus.")

# --- LOAD & TRAIN MODEL (Simplified for the app) ---
@st.cache_data
def load_and_train():
    df = pd.read_csv('pregnancy.csv', comment='#').dropna(subset=['fetal_health'])
    # Clean non-numeric columns
    for col in ['light_decelerations', 'prolongued_decelerations']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = load_and_train()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Patient Data")
user_inputs = {}

# We will create sliders for the top 5 most important features
user_inputs['abnormal_short_term_variability'] = st.sidebar.slider("Abnormal Short Term Variability", 0.0, 100.0, 50.0)
user_inputs['mean_value_of_short_term_variability'] = st.sidebar.slider("Mean Short Term Var", 0.0, 10.0, 1.0)
user_inputs['percentage_of_time_with_abnormal_long_term_variability'] = st.sidebar.slider("Abnormal Long Term Var %", 0.0, 100.0, 10.0)
user_inputs['baseline value'] = st.sidebar.slider("Baseline Heart Rate", 100.0, 180.0, 120.0)
user_inputs['accelerations'] = st.sidebar.slider("Accelerations", 0.0, 0.02, 0.005, format="%.3f")

# Fill other columns with median values to keep the model happy
for col in feature_names:
    if col not in user_inputs:
        user_inputs[col] = 0.0

# --- PREDICTION LOGIC ---
input_df = pd.DataFrame([user_inputs])[feature_names]
prediction = model.predict(input_df)[0]
probs = model.predict_proba(input_df)[0]

# --- DISPLAY RESULTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    if prediction == 1.0:
        st.success("✅ NORMAL")
    elif prediction == 2.0:
        st.warning("⚠️ SUSPECT")
    else:
        st.error("🚨 PATHOLOGICAL")

with col2:
    st.subheader("Confidence Levels")
    st.write(f"Normal: {probs[0]:.1%}")
    st.write(f"Suspect: {probs[1]:.1%}")
    st.write(f"Pathological: {probs[2]:.1%}")