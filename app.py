import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------- PAGE CONFIG (must be first Streamlit command) ----------
st.set_page_config(page_title="Sleep Health Analyzer", page_icon="🛌", layout="wide")

# ---------- LOAD MODELS ----------
clf = joblib.load('disorder_model.pkl')
reg = joblib.load('quality_model.pkl')
kmeans = joblib.load('cluster_model.pkl')
scaler = joblib.load('cluster_scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')
target_le = joblib.load('target_encoder.pkl')

cluster_names = {
    0: "Active, Low-Stress",
    1: "Sedentary, High-Stress",
    2: "Balanced, Moderate",
    3: "Active, Short-Sleep"
}

# ---------- HERO BANNER ----------
st.image(
    "https://images.unsplash.com/photo-1541781774459-bb2af2f05b55?auto=format&fit=crop&w=1600&q=80",
    use_container_width=True
)

st.markdown(
    """
    <div style="text-align:center; padding-top: 10px;">
        <h1 style="font-size: 42px; margin-bottom:0;">🛌 Sleep Health & Lifestyle Risk Analyzer</h1>
        <p style="color:gray; font-size:16px;">Adjust your lifestyle inputs and watch your sleep risk profile update live.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------- SIDEBAR INPUTS ----------
with st.sidebar:
    st.header("⚙️ Your Lifestyle Inputs")

    gender = st.selectbox("Gender", le_dict['Gender'].classes_)
    age = st.slider("Age", 18, 80, 35)
    occupation = st.selectbox("Occupation", le_dict['Occupation'].classes_)
    bmi = st.selectbox("BMI Category", le_dict['BMI Category'].classes_)

    st.subheader("Sleep")
    sleep_duration = st.slider("Sleep Duration (hrs)", 3.0, 10.0, 7.0, 0.1)

    st.subheader("Activity & Stress")
    activity = st.slider("Physical Activity Level (0-100)", 0, 100, 50)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    steps = st.slider("Daily Steps", 1000, 20000, 6000)

    st.subheader("Vitals")
    heart_rate = st.slider("Heart Rate", 50, 120, 70)
    bp_sys = st.slider("BP Systolic", 90, 180, 120)
    bp_dia = st.slider("BP Diastolic", 60, 120, 80)

# ---------- ENCODE INPUTS ----------
gender_enc = le_dict['Gender'].transform([gender])[0]
occ_enc = le_dict['Occupation'].transform([occupation])[0]
bmi_enc = le_dict['BMI Category'].transform([bmi])[0]

# Sleep quality prediction
reg_input = np.array([[gender_enc, age, occ_enc, sleep_duration, activity,
                        stress, bmi_enc, heart_rate, steps, bp_sys, bp_dia]])
predicted_quality = reg.predict(reg_input)[0]

# Disorder prediction
clf_input = np.array([[gender_enc, age, occ_enc, sleep_duration, predicted_quality,
                        activity, stress, bmi_enc, heart_rate, steps, bp_sys, bp_dia]])
disorder_pred = clf.predict(clf_input)[0]
disorder_proba = clf.predict_proba(clf_input)[0]
disorder_label = target_le.inverse_transform([disorder_pred])[0]

# Cluster assignment
cluster_input = scaler.transform([[age, sleep_duration, activity, stress, heart_rate, steps]])
cluster_id = kmeans.predict(cluster_input)[0]

# ---------- RESULT COLOR CODING ----------
risk_icons = {"None": "🟢", "Insomnia": "🟠", "Sleep Apnea": "🔴"}
risk_icon = risk_icons.get(disorder_label, "⚪")

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📊 Results", "🔍 Why This Prediction"])

with tab1:
    st.subheader("Your Sleep Profile")

    c1, c2, c3 = st.columns(3)
    c1.metric("😴 Predicted Sleep Quality", f"{predicted_quality:.1f} / 10")
    c2.metric(f"{risk_icon} Sleep Disorder Risk", disorder_label)
    c3.metric("🧬 Lifestyle Type", cluster_names.get(cluster_id, f"Cluster {cluster_id}"))

    st.markdown("### Risk Probability Breakdown")

    fig = go.Figure(go.Bar(
        x=disorder_proba * 100,
        y=target_le.classes_,
        orientation='h',
        marker_color=['#F94144' if p == max(disorder_proba) else '#577590' for p in disorder_proba],
        text=[f"{p*100:.1f}%" for p in disorder_proba],
        textposition='outside'
    ))
    fig.update_layout(
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Personalized takeaway
    if disorder_label == "None" and stress >= 7:
        st.info("💡 Your risk is currently low, but your stress level is high — this is worth watching over time.")
    elif disorder_label != "None":
        st.warning(f"⚠️ Your inputs show patterns associated with **{disorder_label}**. Consider speaking with a sleep specialist.")
    else:
        st.success("✅ Your current lifestyle inputs are associated with a healthy sleep pattern.")

with tab2:
    st.subheader("What's Driving This Prediction")

    importances = clf.feature_importances_
    feature_names = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                      'Activity', 'Stress', 'BMI', 'Heart Rate', 'Steps', 'BP Sys', 'BP Dia']
    top_idx = np.argsort(importances)[-6:][::-1]

    imp_fig = go.Figure(go.Bar(
        x=[importances[i] for i in top_idx],
        y=[feature_names[i] for i in top_idx],
        orientation='h',
        marker_color='#43AA8B'
    ))
    imp_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Importance"
    )
    st.plotly_chart(imp_fig, use_container_width=True)

    st.caption("These are the lifestyle factors the model relies on most heavily across all predictions — not just for your specific inputs.")

st.divider()
st.caption("Built with Random Forest classification & regression, and K-Means clustering, on the Sleep Health & Lifestyle Dataset (Kaggle).")