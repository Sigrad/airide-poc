import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_harvester import DataHarvester
from feature_engine import FeatureEngineer
from model_trainer import ModelTrainer

st.set_page_config(page_title="AIRide PoC", layout="wide")

st.title("AIRide: Live Wait Time Prediction")

if os.path.exists("real_waiting_times.csv"):
    st.success("REAL DATA MODE active")
else:
    st.warning("SIMULATION MODE active (No CSV found)")

days = st.sidebar.slider("History Window (Days)", 30, 90, 60)
train_btn = st.sidebar.button("Train Model")

@st.cache_data(ttl=600)
def get_data(d):
    h = DataHarvester()
    df_raw = h.fetch_historical_data(days_back=d)
    if df_raw.empty: return pd.DataFrame()
    fe = FeatureEngineer()
    return fe.enrich_data(df_raw)

with st.spinner("Loading Pipeline..."):
    df = get_data(days)

if df.empty or len(df) < 5:
    st.error("Not enough data to run. Please run 'data_harvester.py' locally first or check API.")
else:
    tab1, tab2 = st.tabs(["Analysis", "Prediction & AI"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", len(df))
        c2.metric("Attractions", df['ride_name'].nunique())
        c3.metric("Last Data Point", str(df['datetime'].max()))
        
        st.subheader("Wait Time Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(data=df, x='wait_time', y='ride_name', ax=ax)
        st.pyplot(fig)

    with tab2:
        if train_btn or 'model' in st.session_state:
            st.write("Training Model...")
            if 'model' not in st.session_state:
                mt = ModelTrainer()
                st.session_state['model'] = mt.train_and_evaluate(df)
                st.success("Training Complete! Check Terminal for Diebold-Mariano Statistics.")
            
            st.divider()
            st.subheader("Live Simulator")
            
            c1, c2 = st.columns(2)
            temp = c1.slider("Temp (C)", 0, 40, 22)
            rain = c2.slider("Rain (mm)", 0.0, 20.0, 0.0)
            
            rides = df['ride_name'].unique()
            res = []
            
            for r in rides:
                try:
                    ride_id = df[df['ride_name']==r]['ride_id'].iloc[0]
                    avg = df[df['ride_name']==r]['wait_time'].mean()
                except: continue
                
                hci = (4*max(0,10-abs(temp-25)*0.5)) + (3*max(0,10-rain*2)) + 20
                
                inp = pd.DataFrame([{
                    'hour': 14, 'weekday': 5, 'month': 7, 'is_weekend': 1,
                    'holiday_de_bw': 0, 'holiday_fr_zone_b': 0, 'holiday_ch_bs': 0,
                    'temp': temp, 'rain': rain, 'HCI_Urban': hci,
                    'wait_time_lag_1': avg, 'wait_time_lag_6': avg, 'ride_id': ride_id
                }])
                
                pred = st.session_state['model'].predict(inp)[0]
                res.append({'Ride': r, 'Prediction (min)': int(pred)})
            
            st.table(pd.DataFrame(res).set_index('Ride'))
        else:
            st.info("Click 'Train Model' to start.")