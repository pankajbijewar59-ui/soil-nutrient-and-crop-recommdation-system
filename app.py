"""
app.py
------
Streamlit web app for AI-based Soil Nutrient Analysis & Crop
Recommendation.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Soil Nutrient & Crop Recommendation", page_icon="🌾", layout="wide")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Ideal general reference ranges used ONLY for the deficiency/advisory panel
# (broad agronomic averages, not crop-specific)
IDEAL_RANGES = {
    "N": (40, 100, "kg/ha", "Nitrogen"),
    "P": (30, 80, "kg/ha", "Phosphorus"),
    "K": (20, 60, "kg/ha", "Potassium"),
    "ph": (6.0, 7.5, "", "Soil pH"),
}

ADVICE = {
    "N": {
        "low": "Nitrogen kam hai — Urea ya Ammonium Sulphate jaisi nitrogen-rich khaad daalein, aur green manure crops (jaise dhaincha) use karein.",
        "high": "Nitrogen zyada hai — nitrogen fertilizer application rok dein, isse excessive vegetative growth aur pest attack ho sakta hai.",
    },
    "P": {
        "low": "Phosphorus kam hai — Single Super Phosphate (SSP) ya DAP add karein, root development ke liye zaroori hai.",
        "high": "Phosphorus zyada hai — phosphate fertilizers avoid karein, ye micronutrient (zinc/iron) uptake ko block kar sakta hai.",
    },
    "K": {
        "low": "Potassium kam hai — Muriate of Potash (MOP) ya wood ash use karein, disease resistance aur fruit quality ke liye important hai.",
        "high": "Potassium zyada hai — potash fertilizer application kam karein.",
    },
    "ph": {
        "low": "Mitti acidic hai — Agricultural Lime (chuna) daalein pH balance karne ke liye.",
        "high": "Mitti alkaline hai — Gypsum ya organic compost (jaise farmyard manure) add karein pH kam karne ke liye.",
    },
}


@st.cache_resource
def load_model():
    model = joblib.load("crop_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def get_nutrient_analysis(values):
    """values: dict with N,P,K,ph -> returns list of (nutrient_name, status, advice)"""
    analysis = []
    for key, (low, high, unit, label) in IDEAL_RANGES.items():
        val = values[key]
        if val < low:
            analysis.append((label, val, unit, "Low ⚠️", ADVICE[key]["low"]))
        elif val > high:
            analysis.append((label, val, unit, "High ⚠️", ADVICE[key]["high"]))
        else:
            analysis.append((label, val, unit, "Optimal ✅", "Is nutrient ka level theek hai, koi correction ki zaroorat nahi."))
    return analysis


def main():
    st.title("🌾 AI Soil Nutrient Analysis & Crop Recommendation System")
    st.markdown(
        "Apni mitti (soil) ki values daaliye — N, P, K, temperature, humidity, pH aur rainfall — "
        "aur AI model aapko sabse suitable **crop recommend** karega, saath hi **nutrient deficiency analysis** bhi dega."
    )

    if not (os.path.exists("crop_model.pkl") and os.path.exists("scaler.pkl")):
        st.error("Model files nahi mile. Pehle `python data_generator.py` aur `python train_model.py` run karein.")
        return

    model, scaler = load_model()

    st.sidebar.header("🧪 Soil & Climate Inputs")
    N = st.sidebar.slider("Nitrogen - N (kg/ha)", 0, 150, 50)
    P = st.sidebar.slider("Phosphorus - P (kg/ha)", 0, 150, 50)
    K = st.sidebar.slider("Potassium - K (kg/ha)", 0, 210, 50)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 45.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
    ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

    predict_btn = st.sidebar.button("🔍 Analyze & Recommend", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(model.classes_[i], probs[i] * 100) for i in top5_idx]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("✅ Recommended Crop")
            st.success(f"**{prediction.upper()}**  (confidence: {top5[0][1]:.1f}%)")

            st.subheader("📊 Top 5 Crop Matches")
            top5_df = pd.DataFrame(top5, columns=["Crop", "Confidence (%)"])
            st.bar_chart(top5_df.set_index("Crop"))

        with col2:
            st.subheader("🧬 Soil Nutrient Analysis")
            values = {"N": N, "P": P, "K": K, "ph": ph}
            analysis = get_nutrient_analysis(values)
            for label, val, unit, status, advice in analysis:
                with st.expander(f"{label}: {val} {unit} — {status}"):
                    st.write(advice)

        st.divider()
        st.subheader("📈 Your Input Summary")
        st.dataframe(input_df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("Sidebar mein soil values set karke **'Analyze & Recommend'** button dabayein.")


if __name__ == "__main__":
    main()
