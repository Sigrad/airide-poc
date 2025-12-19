import streamlit as st
import pandas as pd
from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

# --- CONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")

@st.cache_data(ttl=60)
def load_data():
    collector = DataCollector()
    df_raw = collector.fetch_historical_data()
    if df_raw.empty: return pd.DataFrame(), pd.DataFrame()
    
    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    return df_raw, df_processed

df_raw, df_ai = load_data()

if df_raw.empty:
    st.error("Keine Daten gefunden. Bitte CSV oder API prüfen.")
else:
    # Use 'datetime' column ensured by FeatureEngineering
    latest_ts = df_ai['datetime'].max()
    snapshot = df_ai[df_ai['datetime'] == latest_ts]
    
    st.title("AIRide Dashboard")
    
    # KPI Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Letztes Update", latest_ts.strftime('%H:%M'))
    c2.metric("Ø Wartezeit", f"{snapshot['wait_time'].mean():.1f} min")
    c3.metric("Wetter", f"{snapshot['temp'].iloc[0]} °C")

    # Tabs
    tab1, tab2 = st.tabs(["Monitor", "Training"])
    
    with tab1:
        st.dataframe(snapshot[['ride_name', 'wait_time', 'is_open']])
        
    with tab2:
        if st.button("Training starten"):
            trainer = PredictionModel()
            results = trainer.run_benchmark(df_ai)
            st.session_state['trainer'] = trainer
            st.write(results)