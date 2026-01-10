import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Fetal Health AI Dashboard", layout="wide")
st.title("🩺 Fetal Health Monitoring System")

# --- 2. LOAD & TRAIN MODEL ---
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv('pregnancy.csv', comment='#').dropna(subset=['fetal_health'])
    
    # Data Cleaning
    cols_to_fix = ['light_decelerations', 'prolongued_decelerations']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    
    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns, df

model, feature_names, raw_data = load_and_train()

# --- 3. SIDEBAR: PATIENT INFO & VITALS ---
st.sidebar.header("📝 Patient Information")
patient_id = st.sidebar.text_input("Patient ID", "PAT-001")
patient_name = st.sidebar.text_input("Patient Name", "John Doe")

st.sidebar.markdown("---")
st.sidebar.header("📊 Clinical Vitals")
user_inputs = {}

# Primary Sliders
user_inputs['abnormal_short_term_variability'] = st.sidebar.slider("Abnormal Short Term Var (%)", 0, 100, 45)
user_inputs['mean_value_of_short_term_variability'] = st.sidebar.slider("Mean Short Term Var", 0.0, 10.0, 1.3)
user_inputs['baseline value'] = st.sidebar.slider("Heart Rate (BPM)", 100, 180, 130)
user_inputs['accelerations'] = st.sidebar.slider("Accelerations", 0.0, 0.02, 0.003, format="%.3f")
user_inputs['prolongued_decelerations'] = st.sidebar.slider("Prolonged Decelerations", 0.0, 0.01, 0.0, format="%.3f")

# Fill missing columns for the AI model
for col in feature_names:
    if col not in user_inputs:
        user_inputs[col] = 0.0

# --- 4. PREDICTION LOGIC ---
input_df = pd.DataFrame([user_inputs])[feature_names]
prediction = model.predict(input_df)[0]
confidence = model.predict_proba(input_df).max()
probs = model.predict_proba(input_df)[0]

# Mapping condition to your requested terms
if prediction == 1.0:
    p_status = "NORMAL"
    advice = f"Patient {patient_name} ({patient_id}) is Healthy. No risk detected."
    status_msg = "success"
elif prediction == 2.0:
    p_status = "MILD (SUSPECT)"
    advice = f"Patient {patient_name} ({patient_id}) shows Mild variations. Suggest close monitoring."
    status_msg = "warning"
else:
    p_status = "RISK (PATHOLOGICAL)"
    advice = f"Patient {patient_name} ({patient_id}) is at HIGH RISK! Urgent medical intervention required."
    status_msg = "error"

# --- 5. TOP DISPLAY: PATIENT REPORT ---
st.subheader(f"📋 Clinical Report: {patient_name}")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Patient ID", patient_id)
with col_b:
    if status_msg == "success": st.success(p_status)
    elif status_msg == "warning": st.warning(p_status)
    else: st.error(p_status)
with col_c:
    st.metric("AI Confidence", f"{confidence:.1%}")

# Chat-style advice
st.chat_message("assistant", avatar="🩺").write(f"**Clinical Insight:** {advice}")

st.markdown("---")

# --- 6. GRAPHS SECTION ---
st.subheader("🔍 Analytical Visuals")
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.write("### Top Importance Factors")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
    fig, ax = plt.subplots()
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    st.pyplot(fig)

with g_col2:
    st.write("### Patient Risk Probability")
    fig2, ax2 = plt.subplots()
    # Pie chart showing the specific probability for this patient
    ax2.pie(probs, labels=["Normal", "Mild", "Risk"], autopct='%1.1f%%', colors=['#2ecc71', '#f39c12', '#e74c3c'], startangle=140)
    st.pyplot(fig2)

st.info(f"Summary: The patient condition is currently {p_status} based on {confidence:.1%} AI certainty.")