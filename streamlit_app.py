import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

# --- KONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")

# --- HEADER BEREICH ---
st.title("AIRide: Analyse von Besucherströmen & Wartezeiten")
st.markdown("Echtzeit-Datenüberwachung und prädiktive Modellierung für den Europa-Park.")

# --- SIDEBAR STEUERUNG ---
st.sidebar.header("Systemsteuerung")

if st.sidebar.button("🔄 Daten aktualisieren / Cache leeren"):
    st.cache_data.clear()
    st.rerun()

if os.path.exists("real_waiting_times.csv"):
    st.sidebar.success("Status: ONLINE (Echtzeit-Daten)")
else:
    st.sidebar.warning("Status: SIMULATION (Synthetische Daten)")

st.sidebar.markdown("---")
st.sidebar.subheader("Modell-Verwaltung")
days = st.sidebar.slider("Trainingszeitraum (Tage)", 30, 90, 60, help="Beeinflusst nur den Simulations-Modus")
train_btn = st.sidebar.button("Modell trainieren", type="primary")

# --- DATEN LADEN ---
@st.cache_data(ttl=60)
def load_data_pipeline(days_back):
    harvester = DataCollector()
    # 1. Rohdaten holen (Enthält auch geschlossene Bahnen)
    df_raw = harvester.fetch_historical_data(days_back=days_back)
    
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 2. Daten verarbeiten (Entfernt geschlossene Bahnen für KI-Training)
    engineer = FeatureEngineering()
    df_processed = engineer.process_data(df_raw)
    
    return df_raw, df_processed

with st.spinner("Synchronisiere Datenpipeline..."):
    # Wir laden beides: Rohdaten für Anzeige, KI-Daten für Modell
    df_raw, df_ai = load_data_pipeline(days)

# --- DASHBOARD HAUPTBEREICH ---
if df_raw.empty:
    st.error("Systemfehler: Keine Daten gefunden.")
else:
    # --- 1. KPI ZEILE (Basierend auf Rohdaten) ---
    last_update = df_raw['datetime'].max()
    
    # Filter auf den allerletzten Zeitstempel
    latest_snapshot_raw = df_raw[df_raw['datetime'] == last_update]
    
    # Zählen, wie viele davon WIRKLICH offen sind
    if 'is_open' in latest_snapshot_raw.columns:
        open_rides_count = latest_snapshot_raw[latest_snapshot_raw['is_open'] == True].shape[0]
    else:
        # Fallback
        open_rides_count = latest_snapshot_raw[latest_snapshot_raw['wait_time'] > 0].shape[0]

    # Temperatur aus Rohdaten
    current_temp = latest_snapshot_raw['temp'].mean()
    
    # Durchschnittswartezeit (nur von offenen Bahnen)
    if open_rides_count > 0:
        avg_wait = latest_snapshot_raw[latest_snapshot_raw['is_open'] == True]['wait_time'].mean()
    else:
        avg_wait = 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Letztes Update", str(last_update.strftime('%H:%M:%S')))
    
    # Dynamische Anzeige Status
    if open_rides_count == 0:
        kpi2.error(f"Offene Attraktionen: {open_rides_count} (PARK GESCHLOSSEN)")
    else:
        kpi2.metric("Offene Attraktionen", open_rides_count)
        
    kpi3.metric("Ø Wartezeit", f"{avg_wait:.1f} min")
    kpi4.metric("Aktuelle Temp", f"{current_temp:.1f} °C")

    st.markdown("---")

    # --- 2. TABS ---
    tab_overview, tab_prediction = st.tabs(["Aktueller Status & Analyse", "KI-Prognose-Simulator"])

    with tab_overview:
        if open_rides_count == 0:
            st.info("😴 Der Park ist derzeit geschlossen. Keine aktuellen Wartezeiten verfügbar.")
            st.subheader("Letzter bekannter Status (vor Schließung)")
            # Spalten umbenennen für Anzeige
            display_df = df_raw.tail(10).rename(columns={
                'datetime': 'Zeit', 'ride_name': 'Attraktion', 'wait_time': 'Wartezeit', 'temp': 'Temp', 'is_open': 'Offen'
            })
            st.dataframe(display_df, use_container_width=True)
        else:
            col_chart, col_raw = st.columns([2, 1])
            with col_chart:
                st.subheader("Aktuelle Wartezeiten")
                # Nur offene Bahnen für Chart
                open_snapshot = latest_snapshot_raw[latest_snapshot_raw['is_open'] == True].sort_values('wait_time', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=open_snapshot, x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_xlabel("Wartezeit (Minuten)")
                ax.set_ylabel("")
                st.pyplot(fig)
                
            with col_raw:
                st.subheader("Daten-Protokoll")
                display_df = df_ai[['datetime', 'ride_name', 'wait_time', 'temp']].tail(15).rename(columns={
                    'datetime': 'Zeit', 'ride_name': 'Attraktion', 'wait_time': 'Wartezeit', 'temp': 'Temp'
                })
                st.dataframe(display_df, use_container_width=True)

    with tab_prediction:
        # Prüfung: Genug Daten?
        if df_ai.empty or len(df_ai) < 5:
            st.warning("Nicht genügend historische Daten von offenen Attraktionen für das Training vorhanden (Modell benötigt Wartezeiten > 0).")
        
        elif 'model' not in st.session_state and not train_btn:
            st.info("Das KI-Modell ist noch nicht trainiert. Bitte klicken Sie auf 'Modell trainieren' in der Seitenleiste.")
        
        else:
            if train_btn:
                trainer = PredictionModel()
                with st.spinner("Trainiere Random Forest Regressor..."):
                    # Training mit bereinigten Daten
                    st.session_state['model'] = trainer.train_and_evaluate(df_ai)
                st.success("Modell erfolgreich trainiert!")
            
            if 'model' in st.session_state:
                st.subheader("Prädiktiver Simulator (Nowcasting)")
                
                c1, c2, c3 = st.columns(3)
                sim_temp = c1.slider("Temperatur (°C)", 0, 40, 22)
                sim_rain = c2.slider("Niederschlag (mm)", 0.0, 20.0, 0.0)
                sim_cloud = c3.slider("Bewölkung (%)", 0, 100, 50)
                
                # Simulation
                rides_list = df_raw['ride_name'].unique()
                predictions = []
                
                for ride in rides_list:
                    try:
                        # Metadaten aus KI-Datensatz holen
                        ride_data = df_ai[df_ai['ride_name'] == ride]
                        if ride_data.empty: continue
                        
                        ride_id = ride_data['ride_id'].iloc[0]
                        avg_lag = ride_data['wait_time'].mean()
                    except: continue
                    
                    # HCI Berechnung
                    tc = max(0, 10 - abs(sim_temp - 25) * 0.5)
                    a = (100 - sim_cloud) / 10
                    p = max(0, 10 - sim_rain * 2)
                    w = 10 
                    hci_score = (4 * tc) + (2 * a) + (3 * p) + (1 * w)
                    
                    input_vector = pd.DataFrame([{
                        'hour': 14, 'weekday': 5, 'month': 7, 'is_weekend': 1,
                        'holiday_de_bw': 0, 'holiday_fr_zone_b': 0, 'holiday_ch_bs': 0,
                        'temp': sim_temp, 'rain': sim_rain, 'HCI_Urban': hci_score,
                        'wait_time_lag_1': avg_lag, 'wait_time_lag_6': avg_lag,
                        'ride_id': ride_id
                    }])
                    
                    try:
                        pred_val = st.session_state['model'].predict(input_vector)[0]
                        predictions.append({
                            'Attraktion': ride,
                            'Prognose (Min)': int(pred_val),
                            'HCI Einfluss': f"{hci_score:.1f}"
                        })
                    except: pass
                
                if predictions:
                    st.table(pd.DataFrame(predictions).sort_values('Prognose (Min)', ascending=False).set_index('Attraktion'))
                else:
                    st.warning("Simulation noch nicht möglich - Modell benötigt diversere Trainingsdaten.")