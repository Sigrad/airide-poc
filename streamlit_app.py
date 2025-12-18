import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="AIRide Real Data", page_icon="🎢", layout="wide")

st.title("🎢 AIRide: Live Dashboard")
st.markdown("**Status:** Production Mode (Trained on Real Harvested Data)")
st.markdown("---")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('airide_model_scientific.pkl')
    except:
        return None

model = load_model()

if not model:
    st.error("❌ Kein Modell gefunden! Hast du '1_train_on_real_csv.py' ausgeführt (nachdem der Harvester lief)?")
    st.stop()

# --- INPUTS ---
with st.sidebar:
    st.header("Parameter")
    
    hour = st.slider("Uhrzeit", 9, 20, 14, format="%d:00 Uhr")
    is_weekend = st.checkbox("Wochenende?", value=True)
    
    st.subheader("Dreiländereck")
    c1, c2, c3 = st.columns(3)
    is_bw = c1.checkbox("DE", value=False)
    is_fr = c2.checkbox("FR", value=False)
    is_ch = c3.checkbox("CH", value=False)
    
    st.subheader("Wetter (HCI)")
    temp = st.slider("Temp (°C)", 0, 40, 24)
    clouds = st.slider("Wolken (%)", 0, 100, 20)
    rain = st.number_input("Regen (mm)", 0.0, 50.0, 0.0)
    wind = st.slider("Wind (km/h)", 0, 80, 10)
    
    st.subheader("Live-Situation")
    lag = st.slider("Wartezeit vor 1h", 0, 120, 30)

# --- CALCULATION ---
# HCI
tc = (np.clip(temp, 0, 35) / 35) * 10
a = ((100 - clouds) / 100) * 10
p = 0 if rain > 0.5 else 10
w = ((60 - wind) / 60) if wind < 60 else 0
w *= 10
hci = (4*tc + 2*a + 3*p + 1*w) / 10 * 10

# DataFrame
input_df = pd.DataFrame({
    'hour': [hour], 'is_weekend': [int(is_weekend)],
    'is_holiday_BW': [int(is_bw)], 'is_school_holiday_FR': [int(is_fr)], 'is_holiday_CH': [int(is_ch)],
    'temp': [temp], 'clouds': [clouds], 'rain_mm': [rain], 'wind_kmh': [wind],
    'HCI_Urban': [hci], 'wait_time_lag_1h': [lag]
})

# Predict
pred = int(max(0, model.predict(input_df)[0]))

# --- VIEW ---
c1, c2 = st.columns([1, 1])
with c1:
    st.metric("Prognose Wartezeit", f"{pred} Min", delta_color="inverse")
    st.info(f"HCI Score: {hci:.1f}")

with c2:
    st.caption("Einflussfaktoren (Basierend auf deinen gesammelten Daten)")
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(input_df.columns[idx])
    st.pyplot(fig)