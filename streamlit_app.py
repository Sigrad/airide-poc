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

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("AIRide: Crowd Flow & Wait Time Analytics")
st.markdown("Real-time data monitoring and predictive modeling for Europa-Park.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("System Control")

# Data Source Status
if os.path.exists("real_waiting_times.csv"):
    st.sidebar.success("Status: ONLINE (Real Data)")
    data_mode = "real"
else:
    st.sidebar.warning("Status: SIMULATION (Synthetic Data)")
    data_mode = "sim"

# Training Control
st.sidebar.markdown("---")
st.sidebar.subheader("Model Management")
days = st.sidebar.slider("Training Window (Days)", 30, 90, 60, help="Only affects simulation mode")
train_btn = st.sidebar.button("Train / Retrain Model", type="primary")

# --- DATA LOADING ---
@st.cache_data(ttl=300)
def load_and_process_data(days_back):
    harvester = DataHarvester()
    # The harvester automatically handles CSV vs Synthetic logic
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
    last_update = df['datetime'].max()
    active_rides = df['ride_name'].nunique()
    avg_wait = df['wait_time'].mean()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Last Update", str(last_update.strftime('%H:%M:%S')))
    kpi2.metric("Active Attractions", active_rides)
    kpi3.metric("Avg Wait Time", f"{avg_wait:.1f} min")
    kpi4.metric("Data Points", len(df))

    st.markdown("---")

    # 2. TABS
    tab_overview, tab_prediction = st.tabs(["Current Status & Analysis", "AI Prediction Simulator"])

    with tab_overview:
        col_chart, col_raw = st.columns([2, 1])
        
        with col_chart:
            st.subheader("Current Wait Times (Latest Snapshot)")
            # Get only the latest data point for each ride
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
        # Check if model needs training
        if 'model' not in st.session_state and not train_btn:
            st.info("⚠️ The AI Model is not trained yet. Please click 'Train / Retrain Model' in the sidebar to initialize predictions.")
        
        else:
            # Training Logic
            if train_btn:
                trainer = ModelTrainer()
                with st.spinner("Training Random Forest Regressor..."):
                    st.session_state['model'] = trainer.train_and_evaluate(df)
                st.success("Model trained successfully! Ready for inference.")
            
            if 'model' in st.session_state:
                st.subheader("Predictive Simulator (Nowcasting)")
                st.markdown("Adjust weather parameters to simulate impact on crowd flow based on the Holiday Climate Index (HCI).")
                
                # Simulator Controls
                c1, c2, c3 = st.columns(3)
                sim_temp = c1.slider("Temperature (°C)", 0, 40, 22)
                sim_rain = c2.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
                sim_cloud = c3.slider("Cloud Cover (%)", 0, 100, 50)
                
                # Simulation Loop
                rides_list = df['ride_name'].unique()
                predictions = []
                
                for ride in rides_list:
                    # Get ride metadata
                    try:
                        ride_data = df[df['ride_name'] == ride]
                        ride_id = ride_data['ride_id'].iloc[0]
                        # Use historical average as lag proxy for simulation
                        avg_lag = ride_data['wait_time'].mean()
                    except:
                        continue
                    
                    # Calculate HCI on the fly
                    # HCI = 4*TC + 2*A + 3*P + 1*W
                    tc = max(0, 10 - abs(sim_temp - 25) * 0.5)
                    a = (100 - sim_cloud) / 10
                    p = max(0, 10 - sim_rain * 2)
                    w = 10 # Assume low wind for simulation
                    hci_score = (4 * tc) + (2 * a) + (3 * p) + (1 * w)
                    
                    # Build Feature Vector
                    input_vector = pd.DataFrame([{
                        'hour': 14, # Assume Peak Time
                        'weekday': 5, # Assume Saturday
                        'month': 7, # Assume July
                        'is_weekend': 1,
                        'holiday_de_bw': 0,
                        'holiday_fr_zone_b': 0,
                        'holiday_ch_bs': 0,
                        'temp': sim_temp,
                        'rain': sim_rain,
                        'HCI_Urban': hci_score,
                        'wait_time_lag_1': avg_lag, 
                        'wait_time_lag_6': avg_lag,
                        'ride_id': ride_id
                    }])
                    
                    # Predict
                    try:
                        pred_val = st.session_state['model'].predict(input_vector)[0]
                        predictions.append({
                            'Attraction': ride,
                            'Predicted Wait (min)': int(pred_val),
                            'HCI Impact': f"{hci_score:.1f}"
                        })
                    except Exception as e:
                        pass
                
                # Display Results
                st.table(pd.DataFrame(predictions).sort_values('Predicted Wait (min)', ascending=False).set_index('Attraction'))