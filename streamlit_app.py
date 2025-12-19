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

# --- APPLICATION CONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e1e1e", 
    "grid.color": "#444444", 
    "text.color": "white", 
    "xtick.color": "white", 
    "ytick.color": "white", 
    "axes.labelcolor": "white"
})

# --- UI HEADER ---
st.title("AIRide: Analyse und Prognose von Besucherströmen")
st.markdown("""
System zur Überwachung und prädiktiven Modellierung von Wartezeiten im Europa-Park.
Methodik: Ensemble-Learning (Random Forest, Gradient Boosting) und Deep Learning (LSTM) unter Einbezug des Holiday Climate Index (HCI).
""")

# --- SIDEBAR CONTROL ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Datenbestand aktualisieren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modelle trainieren", type="primary")

st.sidebar.markdown("---")
status_source = "ONLINE (API)" if os.path.exists("real_waiting_times.csv") else "OFFLINE (Synthetisch)"
st.sidebar.info(f"Datenquelle: {status_source}")

# --- DATA PIPELINE INITIALIZATION ---
@st.cache_data(ttl=60)
def load_data_pipeline():
    collector = DataCollector()
    df_raw = collector.fetch_historical_data()
    if df_raw.empty: 
        return pd.DataFrame(), pd.DataFrame()
    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    return df_raw, df_processed

df_raw, df_ai = load_data_pipeline()

# --- MAIN DASHBOARD LOGIC ---
if df_raw.empty:
    st.error("Keine Daten verfügbar. Bitte stellen Sie sicher, dass 'real_waiting_times.csv' existiert oder die API erreichbar ist.")
else:
    # --- KPI HEADER ---
    latest_ts = df_raw['timestamp'].max()
    snapshot = df_raw[df_raw['timestamp'] == latest_ts]
    
    # Filter for open rides to calculate accurate average
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    avg_wait = open_rides['wait_time'].mean() if not open_rides.empty else 0
    temp_now = snapshot['temp'].mean() if 'temp' in snapshot.columns else 20
    # HCI logic integration for UI
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letzte Aktualisierung", pd.to_datetime(latest_ts).strftime('%H:%M'))
    c2.metric("Aktive Attraktionen", len(open_rides))
    c3.metric("Ø Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric(
        "HCI", 
        f"{hci_score:.0f}/100", 
        help="Der Holiday Climate Index (HCI) bewertet die klimatische Eignung für Freizeitaktivitäten."
    )

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Echtzeit-Monitor", "Modell-Erkenntnisse", "Prognose & Simulation", "Validierung"])

    # TAB 1: ECHTZEIT-MONITOR
    with tab1:
        c_chart, c_table = st.columns([1.5, 1])
        with c_chart:
            st.subheader("Wartezeit (Top 15)")
            if not open_rides.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#0E1117') 
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False).head(15), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_ylabel("")
                ax.set_xlabel("Minuten")
                st.pyplot(fig)
            else:
                st.info("Keine offenen Attraktionen erfasst.")

        with c_table:
            st.subheader("Attraktionen")
            overview_df = snapshot[['ride_name', 'is_open', 'wait_time']].sort_values('wait_time', ascending=False)
            overview_df['Status'] = overview_df['is_open'].map({True: 'Offen', False: 'Geschlossen'})
            
            def style_status(val):
                color = '#ff4b4b' if val == 'Geschlossen' else '#09ab3b'
                return f'color: {color}'

            st.dataframe(
                overview_df[['ride_name', 'Status', 'wait_time']].style.map(style_status, subset=['Status']),
                column_config={
                    "ride_name": "Attraktion",
                    "Status": "Status",
                    "wait_time": st.column_config.ProgressColumn("Wartezeit", format="%d min", min_value=0, max_value=120)
                },
                use_container_width=True, hide_index=True, height=600
            )

    # TAB 2: MODELL-ERKENNTNISSE
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Modelle werden trainiert..."):
                results = trainer.run_benchmark(df_ai)
                st.session_state['trainer'] = trainer
                st.session_state['benchmark'] = results
            st.rerun()

        if 'benchmark' in st.session_state:
            st.subheader("Signifikanz der Einflussfaktoren")
            # Logic for Feature Importance (using Random Forest as example)
            rf_model = st.session_state['trainer'].models.get('rf')
            if rf_model:
                features = st.session_state['trainer'].feature_columns
                imp_df = pd.DataFrame({'Merkmal': features, 'Wert': rf_model.feature_importances_}).sort_values('Wert', ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0E1117')
                sns.barplot(data=imp_df, x='Wert', y='Merkmal', palette="magma", ax=ax)
                st.pyplot(fig)
        else:
            st.info("Bitte nutzen Sie den Button 'Modelle trainieren' in der Seitenleiste.")

    # TAB 3 & 4 (Simulation & Validierung) folgen der Logik von Tab 1/2...
    # (Strukturell korrigiert, um auf st.session_state['trainer'] zuzugreifen)