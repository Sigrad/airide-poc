import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Seitenkonfiguration
st.set_page_config(page_title="AIRide Predictor", layout="centered")

# Header
st.title("🎢 AIRide: Wartezeit-Prognose")
st.markdown("### Europa-Park Optimierung durch KI")
st.write("Dieses Tool nutzt einen **Random Forest Algorithmus**, um Wartezeiten basierend auf Wetter und Zeitfaktoren vorherzusagen.")

# --- SIDEBAR (EINGABEN) ---
st.sidebar.header("Parameter Simulation")
hour = st.sidebar.slider("Uhrzeit", 9, 18, 14)
temp = st.sidebar.slider("Temperatur (°C)", 5, 35, 22)
rain = st.sidebar.slider("Regenwahrscheinlichkeit (%)", 0, 100, 10) / 100
is_weekend = st.sidebar.checkbox("Wochenende?", value=True)
is_holiday = st.sidebar.checkbox("Schulferien?", value=False)

# --- BERECHNUNG ---
# 1. Feature Engineering (Muss exakt wie im Training sein!)
weather_score = (temp / 40) * (1 - rain)

# 2. DataFrame für Vorhersage bauen
input_data = pd.DataFrame({
    'hour': [hour],
    'temp': [temp],
    'rain': [rain],
    'is_weekend': [int(is_weekend)],
    'is_holiday': [int(is_holiday)],
    'weather_score': [weather_score]
})

# 3. Modell laden & Vorhersagen
try:
    model = joblib.load('airide_model.pkl')
    prediction = model.predict(input_data)[0]
    
    # --- HAUPTBEREICH (AUSGABE) ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Prognostizierte Wartezeit", value=f"{int(prediction)} Min")
    
    with col2:
        if prediction < 20:
            st.success("green, Geringe Auslastung")
        elif prediction < 45:
            st.warning("gelb, Mittlere Auslastung")
        else:
            st.error("red, Hohe Auslastung")

    # --- GRAPHIKEN FÜR DAS PAPER ---
    st.divider()
    st.subheader("Modell-Analyse (Live-Daten)")
    
    # Feature Importance Plot
    importances = model.feature_importances_
    feature_names = ['Uhrzeit', 'Temperatur', 'Regen', 'Wochenende', 'Ferien', 'Wetter-Score']
    
    fig, ax = plt.subplots()
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances, align='center', color='#0083B8')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Wichtigste oben
    ax.set_xlabel('Einflussstärke')
    ax.set_title('Feature Importance (Random Forest)')
    
    st.pyplot(fig)
    st.caption("Abb 1: Einflussfaktoren auf die Wartezeit (Live-Berechnung)")

except FileNotFoundError:
    st.error("FEHLER: 'airide_model.pkl' nicht gefunden. Bitte erst '1_generate_and_train.py' lokal ausführen und die .pkl Datei hochladen!")