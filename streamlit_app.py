import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="AIRide Analytics", page_icon="🎢", layout="wide")

# --- HEADER ---
st.title("🎢 AIRide: Predictive Analytics Dashboard")
st.markdown("**Proof of Concept** | Modellierung exogener Faktoren (HCI) & Dreiländereck-Dynamik")
st.markdown("---")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('airide_model_scientific.pkl')
    except:
        return None

model = load_model()

if not model:
    st.error("⚠️ Modell fehlt! Bitte führe zuerst '1_generate_and_train.py' aus.")
    st.stop()

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.header("⚙️ Simulations-Parameter")
    
    st.subheader("1. Zeit & Kalender")
    hour = st.slider("Uhrzeit", 9, 20, 14, format="%d:00 Uhr")
    is_weekend = st.checkbox("Wochenende?", value=True)
    
    st.markdown("**Dreiländereck (Ferien-Status):**")
    c1, c2, c3 = st.columns(3)
    is_bw = c1.checkbox("DE (BW)", value=False)
    is_fr = c2.checkbox("FR (Zone B)", value=False)
    is_ch = c3.checkbox("CH (BS)", value=False)
    
    st.subheader("2. Meteorologie (HCI-Faktoren)")
    temp = st.slider("Temperatur (°C)", 0, 40, 24)
    clouds = st.slider("Bewölkung (%)", 0, 100, 20)
    rain_mm = st.number_input("Niederschlag (mm)", 0.0, 50.0, 0.0, step=0.1)
    wind_kmh = st.slider("Wind (km/h)", 0, 80, 10, help="Ab 40 km/h Gefahr von Ride-Closure")
    
    st.subheader("3. System-Historie")
    lag_1h = st.slider("Wartezeit vor 1h (Lag)", 0, 120, 30, help="Trägheit der Schlange")

# --- LIVE BERECHNUNG ---
# 1. HCI Berechnung (Identisch zum Training!)
tc_score = (np.clip(temp, 0, 35) / 35) * 10
a_score = ((100 - clouds) / 100) * 10
p_score = 0 if rain_mm > 0.5 else 10
w_score = ((60 - wind_kmh) / 60) if wind_kmh < 60 else 0
w_score *= 10

hci_val = (4 * tc_score + 2 * a_score + 3 * p_score + 1 * w_score) / 10 * 10

# 2. DataFrame erstellen
input_df = pd.DataFrame({
    'hour': [hour],
    'is_weekend': [int(is_weekend)],
    'is_holiday_BW': [int(is_bw)],
    'is_school_holiday_FR': [int(is_fr)],
    'is_holiday_CH': [int(is_ch)],
    'temp': [temp],
    'clouds': [clouds],
    'rain_mm': [rain_mm],
    'wind_kmh': [wind_kmh],
    'HCI_Urban': [hci_val],
    'wait_time_lag_1h': [lag_1h]
})

# 3. Vorhersage
prediction = int(max(0, model.predict(input_df)[0]))

# --- AUSGABE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⏱️ Prognose")
    
    # Status Farbe
    color = "#28a745" # Grün
    status = "Niedrig"
    if prediction > 25: 
        color = "#ffc107"; status = "Mittel" # Gelb
    if prediction > 50: 
        color = "#dc3545"; status = "Hoch" # Rot
        
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 10px solid {color};">
        <h1 style="color: #333; margin:0;">{prediction} Min.</h1>
        <p style="margin:0; font-weight:bold; color: {color}">{status}e Auslastung</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"🌧️ **HCI-Index:** {hci_val:.1f} / 100")
    
    if wind_kmh > 45:
        st.error("⚠️ **Wind-Warnung:** Mögliche Schließung (Tipping Point)!")
    elif temp > 32:
        st.warning("🔥 **Hitze-Warnung:** Besucher wechseln zu Wasserbahnen.")

with col2:
    st.subheader("📊 XAI: Feature Importance")
    
    # Plotting
    importances = model.feature_importances_
    features = input_df.columns
    idx = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(range(len(idx)), importances[idx], color="#007bff")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(features[idx])
    ax.set_xlabel("Einflussstärke")
    st.pyplot(fig)

# --- DEBUG ---
with st.expander("🔍 Input-Daten (Validierung)"):
    st.dataframe(input_df)