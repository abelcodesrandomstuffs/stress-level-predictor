import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Load model, scaler, features
# -----------------------------
try:
    model = joblib.load("model/stress_level_best_model.pkl")
    scaler = joblib.load("model/stress_level_scaler.pkl")
    top_features = joblib.load("model/top_features.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are in the directory.")
    st.stop()

# -----------------------------
# UI Configuration
# -----------------------------
st.set_page_config(page_title="Stress Predictor", layout="wide")

st.title("Stress Level Predictor 🧠")
st.markdown("This app predicts stress level based on lifestyle and behavioral features using a trained ML model.")
st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("User Input Features")

input_data = {}
for feature in top_features:
    input_data[feature] = st.sidebar.slider(f"{feature}", 0, 30, 15)

input_df = pd.DataFrame([input_data])[top_features]

# -----------------------------
# Prediction
# -----------------------------
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

# Label Mapping (ensure dataset matches this)
stress_labels = {0: "Low", 1: "Medium", 2: "High"}
display_pred = stress_labels.get(prediction, prediction)

# Color Logic
if prediction == 0:
    color = "green"
elif prediction == 1:
    color = "orange"
else:
    color = "red"

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Result")

    # Colored Output
    st.markdown(
        f"<h2 style='color:{color};'>Predicted Stress Level: {display_pred}</h2>",
        unsafe_allow_html=True
    )

    # Progress bar
    progress_val = (prediction / 2) if isinstance(prediction, (int, float)) else 0.5
    st.progress(min(max(float(progress_val), 0.0), 1.0))

    # Recommendations
    if prediction == 0:
        st.success("Low Stress: You are maintaining a healthy lifestyle. Keep it up.")
    elif prediction == 1:
        st.warning("Moderate Stress: Try improving sleep, managing workload, and practicing relaxation techniques.")
    else:
        st.error("High Stress: Immediate attention recommended. Focus on sleep, reduce pressure, exercise regularly, and seek support if needed.")

with col2:
    st.subheader("Feature Signature")

    categories = list(top_features)
    values = list(input_df.iloc[0].values)

    # Radar Chart
    plot_values = values + [values[0]]
    plot_categories = categories + [categories[0]]

    fig_radar = go.Figure(
        data=[
            go.Scatterpolar(
                r=plot_values,
                theta=plot_categories,
                fill='toself',
                name='Input Profile',
                line_color='#00f2ff'
            )
        ]
    )

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
        showlegend=False,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_radar, use_container_width=True)

# -----------------------------
# Feature Intensity (Bubble Chart)
# -----------------------------
st.markdown("---")
st.subheader("Feature Intensity Analysis")

bubble_df = pd.DataFrame({
    'Feature': top_features,
    'Value': values,
    'Size': [v * 2 for v in values]
})

fig_bubble = px.scatter(
    bubble_df,
    x='Feature',
    y='Value',
    size='Size',
    color='Value',
    color_continuous_scale='Viridis',
    template='plotly_dark'
)

st.plotly_chart(fig_bubble, use_container_width=True)