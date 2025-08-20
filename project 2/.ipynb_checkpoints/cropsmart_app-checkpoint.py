# cropsmart_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="🌱 CropSmart", page_icon="🌾", layout="wide")

# ==============================
# 1. Load & Train Model
# ==============================
@st.cache_resource
def load_and_train():
    data = pd.read_csv("Crop_recommendation.csv")

    X = data.drop("label", axis=1)
    y = data["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, le, data, acc

model, le, data, acc = load_and_train()

# ==============================
# 2. Sidebar
# ==============================
st.sidebar.title("🌾 CropSmart Dashboard")
st.sidebar.markdown("Navigate between **Prediction** and **EDA**")

tab1, tab2 = st.tabs(["🌱 Crop Prediction", "📊 Data Insights"])

# ==============================
# 3. Prediction Tab
# ==============================
with tab1:
    st.header("🌱 Smart Crop Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 50)
        P = st.number_input("Phosphorus (P)", 0, 200, 50)
        K = st.number_input("Potassium (K)", 0, 200, 50)
    with col2:
        temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0)
    with col3:
        ph = st.number_input("⚖️ Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 100.0)

    if st.button("🔍 Predict Best Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        crop_name = le.inverse_transform([prediction])[0]

        st.success(f"🌾 Recommended Crop: **{crop_name.capitalize()}**")
        st.metric(label="📈 Model Accuracy", value=f"{acc*100:.2f}%")

# ==============================
# 4. EDA Tab
# ==============================
with tab2:
    st.header("📊 Crop Data Insights")

    st.subheader("1. Crop Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.countplot(y=data["label"], order=data["label"].value_counts().index, ax=ax1, palette="viridis")
    ax1.set_title("Crop Frequency in Dataset")
    st.pyplot(fig1)

    st.subheader("2. Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(data.drop("label", axis=1).corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Average Feature Values per Crop")
    feature = st.selectbox("Select Feature", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    avg_values = data.groupby("label")[feature].mean().sort_values(ascending=False)
    avg_values.plot(kind="bar", ax=ax3, color="skyblue")
    ax3.set_ylabel(f"Average {feature}")
    st.pyplot(fig3)
