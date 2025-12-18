import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_harvester import DataHarvester
from feature_engine import FeatureEngineer
from model_trainer import ModelTrainer

# --- CONFIGURATION ---
st.set_page_config(page_title="AIRide Analytics", layout="wide")

# --- HEADER SECTION ---
st.title("AIRide: Crowd Flow & Wait Time Analytics")
st.markdown("Real-time data monitoring and predictive modeling for Europa-Park.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("System Control")

if st.sidebar.button("🔄 Refresh Data / Clear Cache"):
    st.cache_data.clear()
    st.rerun()

if os.path.exists("real_waiting_times.csv"):
    st.sidebar.success("Status: ONLINE (Real Data)")
else:
    st.sidebar.warning("Status: SIMULATION (Synthetic Data)")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Management")
days = st.sidebar.slider("Training Window (Days)", 30, 90, 60, help="Only affects simulation mode")
train_btn = st.sidebar.button("Train / Retrain Model", type="primary")

# --- DATA LOADING ---
@st.cache_data(ttl=60)
def load_data_pipeline(days_back):
    harvester = DataHarvester()
    # 1. Hole die ROHDATEN (Hier sind auch geschlossene Bahnen drin!)
    df_raw = harvester.fetch_historical_data(days_back=days_back)
    
    if df_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # 2. Erstelle die PROCESSED DATEN (Hier fliegen geschlossene raus für die KI)
    engineer = FeatureEngineer()
    df_processed = engineer.enrich_data(df_raw)
    
    return df_raw, df_processed

with st.spinner("Synchronizing data pipeline..."):
    # Wir laden jetzt BEIDES
    df_raw, df_ai = load_data_pipeline(days)

# --- MAIN DASHBOARD ---
if df_raw.empty:
    st.error("System Error: No data found.")
else:
    # --- 1. KPI ROW (Basierend auf ROHDATEN) ---
    # Wir nutzen df_raw, damit wir auch sehen, wenn der Park zu ist
    last_update = df_raw['datetime'].max()
    
    # Filter auf den allerletzten Zeitstempel
    latest_snapshot_raw = df_raw[df_raw['datetime'] == last_update]
    
    # Zählen, wie viele davon WIRKLICH offen sind
    # (Wir schauen auf die Spalte 'is_open', falls vorhanden, sonst wait_time > 0)
    if 'is_open' in latest_snapshot_raw.columns:
        open_rides_count = latest_snapshot_raw[latest_snapshot_raw['is_open'] == True].shape[0]
    else:
        open_rides_count = latest_snapshot_raw[latest_snapshot_raw['wait_time'] > 0].shape[0]

    # Temperatur nehmen wir auch aus den Rohdaten
    current_temp = latest_snapshot_raw['temp'].mean()
    
    # Durchschnittswartezeit (nur von offenen Bahnen berechnen, sonst Division durch Null/Verfälschung)
    if open_rides_count > 0:
        avg_wait = latest_snapshot_raw[latest_snapshot_raw['is_open'] == True]['wait_time'].mean()
    else:
        avg_wait = 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Last Update", str(last_update.strftime('%H:%M:%S')))
    
    # Dynamische Farbe für den Status
    if open_rides_count == 0:
        kpi2.error(f"Open Attractions: {open_rides_count} (PARK CLOSED)")
    else:
        kpi2.metric("Open Attractions (Live)", open_rides_count)
        
    kpi3.metric("Avg Wait Time", f"{avg_wait:.1f} min")
    kpi4.metric("Current Temp", f"{current_temp:.1f} C")

    st.markdown("---")

    # --- 2. TABS (Nutzen df_ai für Charts, damit Nullen nicht stören) ---
    tab_overview, tab_prediction = st.tabs(["Current Status & Analysis", "AI Prediction Simulator"])

    with tab_overview:
        if open_rides_count == 0:
            st.info("😴 The Park is currently closed. No wait times to display.")
            st.subheader("Last Known Status (Before Closing)")
            st.dataframe(df_raw.tail(10))
        else:
            col_chart, col_raw = st.columns([2, 1])
            with col_chart:
                st.subheader("Current Wait Times")
                # Wir nehmen die Daten aus dem Snapshot (Rohdaten), aber filtern nur die offenen
                open_snapshot = latest_snapshot_raw[latest_snapshot_raw['is_open'] == True].sort_values('wait_time', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=open_snapshot, x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_xlabel("Wait Time (Minutes)")
                ax.set_ylabel("")
                st.pyplot(fig)
                
            with col_raw:
                st.subheader("Recent Data Log")
                st.dataframe(df_ai[['datetime', 'ride_name', 'wait_time', 'temp']].tail(15), use_container_width=True)

    with tab_prediction:
        # Prüfung: Haben wir überhaupt genug "offene" Daten zum Trainieren?
        if df_ai.empty or len(df_ai) < 5:
            st.warning("Not enough historical data from OPEN rides to train the AI yet. (The Model needs active wait times, not zeros).")
        
        elif 'model' not in st.session_state and not train_btn:
            st.info("The AI Model is not trained yet. Please click 'Train / Retrain Model' in the sidebar.")
        
        else:
            if train_btn:
                trainer = ModelTrainer()
                with st.spinner("Training Random Forest Regressor..."):
                    # WICHTIG: Training NUR mit df_ai (ohne geschlossene Bahnen)
                    st.session_state['model'] = trainer.train_and_evaluate(df_ai)
                st.success("Model trained successfully!")
            
            if 'model' in st.session_state:
                st.subheader("Predictive Simulator (Nowcasting)")
                
                c1, c2, c3 = st.columns(3)
                sim_temp = c1.slider("Temperature (C)", 0, 40, 22)
                sim_rain = c2.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
                sim_cloud = c3.slider("Cloud Cover (%)", 0, 100, 50)
                
                # Für die Simulation nehmen wir die Liste ALLER Bahnen aus den Rohdaten
                rides_list = df_raw['ride_name'].unique()
                predictions = []
                
                for ride in rides_list:
                    # Wir brauchen eine Ride ID aus den Trainingsdaten
                    try:
                        # Suche Metadaten im KI-Datensatz
                        ride_data = df_ai[df_ai['ride_name'] == ride]
                        if ride_data.empty: continue
                        
                        ride_id = ride_data['ride_id'].iloc[0]
                        avg_lag = ride_data['wait_time'].mean()
                    except: continue
                    
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
                            'Attraction': ride,
                            'Predicted Wait (min)': int(pred_val),
                            'HCI Impact': f"{hci_score:.1f}"
                        })
                    except: pass
                
                if predictions:
                    st.table(pd.DataFrame(predictions).sort_values('Predicted Wait (min)', ascending=False).set_index('Attraction'))
                else:
                    st.warning("Cannot simulate yet - model needs more diverse training data.")