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

if os.path.exists("real_waiting_times.csv"):
    st.sidebar.success("Status: ONLINE (Real Data)")
else:
    st.sidebar.warning("Status: SIMULATION (Synthetic Data)")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Management")
days = st.sidebar.slider("Training Window (Days)", 30, 90, 60, help="Only affects simulation mode")
train_btn = st.sidebar.button("Train / Retrain Model", type="primary")

# --- DATA LOADING ---
@st.cache_data(ttl=300)
def load_and_process_data(days_back):
    harvester = DataHarvester()
    df_raw = harvester.fetch_historical_data(days_back=days_back)
    
    if df_raw.empty:
        return pd.DataFrame()
        
    engineer = FeatureEngineer()
    return engineer.enrich_data(df_raw)

with st.spinner("Synchronizing data pipeline..."):
    df = load_and_process_data(days)

# --- MAIN DASHBOARD ---
if df.empty or len(df) < 5:
    st.error("System Error: Insufficient data. Please ensure 'real_waiting_times.csv' is populated or API is accessible.")
else:
    # 1. KPI ROW
    # Note: df only contains OPEN rides due to FeatureEngineer filter
    last_update = df['datetime'].max()
    active_rides = df['ride_name'].nunique()
    
    # Calculate stats based on latest snapshot
    latest_snapshot = df[df['datetime'] == last_update]
    avg_wait = latest_snapshot['wait_time'].mean()
    current_temp = latest_snapshot['temp'].mean()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Last Update", str(last_update.strftime('%H:%M:%S')))
    kpi2.metric("Open Attractions", active_rides)
    kpi3.metric("Avg Wait Time", f"{avg_wait:.1f} min")
    kpi4.metric("Current Temp", f"{current_temp:.1f} C")

    st.markdown("---")

    # 2. TABS
    tab_overview, tab_prediction = st.tabs(["Current Status & Analysis", "AI Prediction Simulator"])

    with tab_overview:
        col_chart, col_raw = st.columns([2, 1])
        
        with col_chart:
            st.subheader("Current Wait Times (Latest Snapshot)")
            latest_df = df[df['datetime'] == df['datetime'].max()].sort_values('wait_time', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=latest_df, x='wait_time', y='ride_name', palette="viridis", ax=ax)
            ax.set_xlabel("Wait Time (Minutes)")
            ax.set_ylabel("")
            st.pyplot(fig)
            
        with col_raw:
            st.subheader("Recent Data Log")
            st.dataframe(df[['datetime', 'ride_name', 'wait_time', 'temp']].tail(15), use_container_width=True)

    with tab_prediction:
        if 'model' not in st.session_state and not train_btn:
            st.info("The AI Model is not trained yet. Please click 'Train / Retrain Model' in the sidebar.")
        
        else:
            if train_btn:
                trainer = ModelTrainer()
                with st.spinner("Training Random Forest Regressor..."):
                    st.session_state['model'] = trainer.train_and_evaluate(df)
                st.success("Model trained successfully!")
            
            if 'model' in st.session_state:
                st.subheader("Predictive Simulator (Nowcasting)")
                
                c1, c2, c3 = st.columns(3)
                sim_temp = c1.slider("Temperature (C)", 0, 40, 22)
                sim_rain = c2.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
                sim_cloud = c3.slider("Cloud Cover (%)", 0, 100, 50)
                
                rides_list = df['ride_name'].unique()
                predictions = []
                
                for ride in rides_list:
                    try:
                        ride_data = df[df['ride_name'] == ride]
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
                
                st.table(pd.DataFrame(predictions).sort_values('Predicted Wait (min)', ascending=False).set_index('Attraction'))