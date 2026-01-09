import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Fetal Health AI Dashboard", layout="wide")
st.title("🩺 Fetal Health AI & Medical Chat")

# --- 2. LOAD & TRAIN MODEL ---
@st.cache_resource
def load_and_train():
    # Loading the data
    df = pd.read_csv('pregnancy.csv', comment='#').dropna(subset=['fetal_health'])
    
    # Cleaning data
    for col in ['light_decelerations', 'prolongued_decelerations']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns, df

model, feature_names, raw_data = load_and_train()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Patient Vitals")
user_inputs = {}
user_inputs['abnormal_short_term_variability'] = st.sidebar.slider("Abnormal Short Term Var (%)", 0, 100, 45)
user_inputs['mean_value_of_short_term_variability'] = st.sidebar.slider("Mean Short Term Var", 0.0, 10.0, 1.3)
user_inputs['baseline value'] = st.sidebar.slider("Heart Rate (BPM)", 100, 180, 130)
user_inputs['accelerations'] = st.sidebar.slider("Accelerations", 0.0, 0.02, 0.003, format="%.3f")
user_inputs['prolongued_decelerations'] = st.sidebar.slider("Prolonged Decelerations", 0.0, 0.01, 0.0, format="%.3f")

# Match all columns for the model
for col in feature_names:
    if col not in user_inputs:
        user_inputs[col] = 0.0

# --- 4. PREDICTION & CHAT LOGIC ---
input_df = pd.DataFrame([user_inputs])[feature_names]
prediction = model.predict(input_df)[0]
confidence = model.predict_proba(input_df).max()

# Decide the Chat message based on prediction
if prediction == 1.0:
    chat_response = "✅ **Normal Status:** The fetal heart rate patterns are healthy. No immediate action is required. Continue routine monitoring."
    status_type = "success"
elif prediction == 2.0:
    chat_response = "⚠️ **Suspect Status:** There are minor irregularities. I recommend increasing the frequency of monitoring and consulting a senior obstetrician."
    status_type = "warning"
else:
    chat_response = "🚨 **Pathological Status:** Urgent! The AI has detected high-risk distress patterns. Immediate medical intervention or a C-section may be necessary."
    status_type = "error"

# --- 5. MAIN DASHBOARD UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🤖 AI Result")
    if status_type == "success": st.success(f"Result: NORMAL ({confidence:.1%})")
    elif status_type == "warning": st.warning(f"Result: SUSPECT ({confidence:.1%})")
    else: st.error(f"Result: PATHOLOGICAL ({confidence:.1%})")

with col2:
    st.subheader("💬 Medical Assistant")
    # --- THIS IS THE CHAT COMPONENT ---
    with st.chat_message("assistant", avatar="🩺"):
        st.write(chat_response)

st.markdown("---")

# --- 6. GRAPHS SECTION ---
st.subheader("📊 Data Visuals")
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.write("**Top Importance Factors**")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
    fig, ax = plt.subplots()
    sns.barplot(x=importances.values, y=importances.index, palette="Blues_d")
    st.pyplot(fig)

with g_col2:
    st.write("**Overall Health Distribution**")
    fig2, ax2 = plt.subplots()
    raw_data['fetal_health'].value_counts().plot.pie(autopct='%1.1f%%', labels=["Normal", "Suspect", "Pathological"], ax=ax2, colors=['#90ee90','#ffcc00','#ff4b4b'])
    st.pyplot(fig2)