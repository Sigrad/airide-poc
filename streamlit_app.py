import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

# --- CONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e1e1e", 
    "grid.color": "#444444", 
    "text.color": "white", 
    "xtick.color": "white", 
    "ytick.color": "white", 
    "axes.labelcolor": "white"
})

# --- HEADER ---
st.title("AIRide: Analyse und Prognose von Besucherströmen")
st.markdown("""
System zur Überwachung und prädiktiven Modellierung von Wartezeiten im Europa-Park.
Methodik: Ensemble-Learning (Random Forest, Gradient Boosting) und Deep Learning (LSTM) unter Einbezug des Holiday Climate Index (HCI).
""")

# --- SIDEBAR ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Datenbestand aktualisieren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modelle trainieren", type="primary")

st.sidebar.markdown("---")
status_text = "ONLINE (API)" if os.path.exists("real_waiting_times.csv") else "OFFLINE (Synthetisch)"
st.sidebar.info(f"Datenquelle: {status_text}")

# --- DATA PIPELINE ---
@st.cache_data(ttl=60)
def load_data_pipeline():
    collector = DataCollector()
    df_raw = collector.fetch_historical_data()
    if df_raw.empty: 
        return pd.DataFrame(), pd.DataFrame()
    
    # Standardize column name for UI consistency
    if 'timestamp' in df_raw.columns and 'datetime' not in df_raw.columns:
        df_raw['datetime'] = pd.to_datetime(df_raw['timestamp'])
    elif 'datetime' in df_raw.columns:
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    return df_raw, df_processed

# Safe data loading
df_raw, df_ai = load_data_pipeline()

if df_raw.empty:
    st.error("Fehler: Keine Daten geladen. Bitte prüfen Sie die Verbindung zur API oder die CSV-Datei.")
else:
    # --- KPI CALCULATIONS ---
    # Determine the latest record safely
    time_col = 'datetime' if 'datetime' in df_raw.columns else 'timestamp'
    latest_ts = df_raw[time_col].max()
    snapshot = df_raw[df_raw[time_col] == latest_ts]
    
    # Average calculation excluding closed rides
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    avg_wait = open_rides['wait_time'].mean() if not open_rides.empty else 0
    temp_now = snapshot['temp'].mean() if 'temp' in snapshot.columns else 20
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    # UI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letzte Aktualisierung", pd.to_datetime(latest_ts).strftime('%H:%M'))
    c2.metric("Aktive Attraktionen", len(open_rides))
    c3.metric("Durchschn. Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric(
        "HCI", 
        f"{hci_score:.0f}/100", 
        help="Der Holiday Climate Index (HCI) bewertet die klimatische Eignung für Freizeitaktivitäten unter Einbezug meteorologischer Faktoren."
    )

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Echtzeit-Monitor", "Modell-Erkenntnisse", "Prognose & Simulation", "Validierung"])

    # TAB 1: MONITOR
    with tab1:
        c_chart, c_table = st.columns([1.5, 1])
        with c_chart:
            st.subheader("Wartezeit (Top 15)")
            if not open_rides.empty:
                fig, ax = plt.subplots(figsize=(8, 6)); fig.patch.set_facecolor('#0E1117') 
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False).head(15), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_ylabel(""); ax.set_xlabel("Minuten"); st.pyplot(fig)
            else:
                st.info("Momentan keine Daten für offene Attraktionen verfügbar.")

        with c_table:
            st.subheader("Attraktionen")
            overview_df = snapshot[['ride_name', 'is_open', 'wait_time']].sort_values('wait_time', ascending=False)
            overview_df['Status'] = overview_df['is_open'].map({True: 'Offen', False: 'Geschlossen'})
            def style_status(val):
                return f"color: {'#ff4b4b' if val == 'Geschlossen' else '#09ab3b'}"
            st.dataframe(
                overview_df[['ride_name', 'Status', 'wait_time']].style.map(style_status, subset=['Status']),
                column_config={
                    "ride_name": "Attraktion",
                    "Status": "Status",
                    "wait_time": st.column_config.ProgressColumn("Wartezeit", format="%d min", min_value=0, max_value=120)
                },
                use_container_width=True, hide_index=True, height=600
            )

    # TAB 2: INSIGHTS
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Algorithmen werden trainiert..."):
                results = trainer.run_benchmark(df_ai)
                st.session_state['trainer'] = trainer
                st.session_state['benchmark'] = results
            st.rerun()
        
        if 'benchmark' in st.session_state and 'trainer' in st.session_state:
            st.subheader("Signifikanz der Einflussfaktoren")
            fi_col1, fi_col2 = st.columns(2)
            # Feature mapping based on standard columns defined in PredictionModel
            features = st.session_state['trainer'].feature_columns
            feature_map_de = {
                'hour': 'Uhrzeit', 'weekday': 'Wochentag', 'month': 'Monat', 'is_weekend': 'Wochenende',
                'holiday_de_bw': 'Feiertag (DE)', 'holiday_fr_zone_b': 'Schulferien (FR)', 'holiday_ch_bs': 'Feiertag (CH)',
                'temp': 'Temperatur', 'rain': 'Niederschlag', 'HCI_Urban': 'HCI',
                'wait_time_lag_1': 'Latenz (10min)', 'wait_time_lag_6': 'Trend (1h)', 'ride_id': 'Attraktionstyp'
            }

            with fi_col1:
                st.markdown("**Random Forest**")
                rf_model = st.session_state['trainer'].models.get('rf')
                if rf_model:
                    imp_df_rf = pd.DataFrame({'Merkmal': [feature_map_de.get(f, f) for f in features], 'Wert': rf_model.feature_importances_}).sort_values('Wert', ascending=False).head(10)
                    fig, ax = plt.subplots(figsize=(5, 3)); fig.patch.set_facecolor('#0E1117')
                    sns.barplot(data=imp_df_rf, x='Wert', y='Merkmal', palette="magma", ax=ax); st.pyplot(fig)

            with fi_col2:
                st.markdown("**Gradient Boosting**")
                gb_model = st.session_state['trainer'].models.get('gb')
                if gb_model:
                    imp_df_gb = pd.DataFrame({'Merkmal': [feature_map_de.get(f, f) for f in features], 'Wert': gb_model.feature_importances_}).sort_values('Wert', ascending=False).head(10)
                    fig, ax = plt.subplots(figsize=(5, 3)); fig.patch.set_facecolor('#0E1117')
                    sns.barplot(data=imp_df_gb, x='Wert', y='Merkmal', palette="viridis", ax=ax); st.pyplot(fig)
        else:
            st.info("Bitte nutzen Sie den Button in der linken Seitenleiste, um die Modell-Analyse zu starten.")

    # TAB 3 & 4 (Simulation & Validation logic follows...)
    # (Reduced for brevity, make sure to use st.session_state['trainer'].predict_ensemble for inference)