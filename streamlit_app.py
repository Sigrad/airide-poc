import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime, time
from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

# --- CONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "grid.color": "#444444", "text.color": "white", "xtick.color": "white", "ytick.color": "white", "axes.labelcolor": "white"})

# --- HEADER ---
st.title("AIRide: Analyse von Besucherströmen & Wartezeiten")
st.markdown("""
Dashboard zur Überwachung und Prognose von Besucherströmen im Europa-Park.
Dieses Tool nutzt **mehrere KI-Modelle (Random Forest, Gradient Boosting, LSTM)** und den **Holiday Climate Index (HCI)** zur Präzisionssteigerung.
""")

# --- SIDEBAR ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Daten aktualisieren / Cache leeren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modelle trainieren", type="primary")

st.sidebar.markdown("---")

data_status = "ONLINE (Echtzeit-Daten)" if os.path.exists("real_waiting_times.csv") else "SIMULATION (Synthetische Daten)"
st.sidebar.info(f"Quelle: {data_status}")

# --- DATA LOADING ---
@st.cache_data(ttl=60)
def load_data_pipeline(days_back=60):
    collector = DataCollector()
    df_raw = collector.fetch_historical_data(days_back=days_back)
    if df_raw.empty: return pd.DataFrame(), pd.DataFrame()
    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    return df_raw, df_processed

with st.spinner("Verarbeite Datenpipeline..."):
    df_raw, df_ai = load_data_pipeline()

# --- MAIN ---
if df_raw.empty:
    st.error("Keine Daten verfügbar. Bitte aktivieren Sie den DataCollector.")
else:
    # --- METRICS ---
    latest_ts = df_raw['datetime'].max()
    snapshot = df_raw[df_raw['datetime'] == latest_ts]
    
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    avg_wait = open_rides['wait_time'].mean() if not open_rides.empty else 0
    temp_now = open_rides['temp'].mean() if 'temp' in open_rides.columns else 0
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letztes Update", latest_ts.strftime('%H:%M'))
    c2.metric("Offene Attraktionen", len(open_rides))
    c3.metric("Durchschn. Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric("HCI", f"{hci_score:.0f}/100", help="Der Holiday Climate Index (HCI) bewertet die Eignung des Wetters für Freizeitparks (0=Schlecht, 100=Ideal).")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Live Analyse", "KI & Modell Insights", "Prognose & Simulation", "Modell-Benchmark"])

    # TAB 1: LIVE
    with tab1:
        c_chart, c_table = st.columns([1.5, 1])
        with c_chart:
            st.subheader("Aktuelle Wartezeiten")
            if not open_rides.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#0E1117') 
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False).head(15), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_ylabel("", color='white')
                ax.set_xlabel("Minuten", color='white')
                st.pyplot(fig)
            else:
                st.info("Park geschlossen.")

        with c_table:
            st.subheader("Liste")
            overview_df = snapshot[['ride_name', 'is_open', 'wait_time']].sort_values('wait_time', ascending=False)
            def style_dark(styler):
                styler.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff', 'border-color': '#444444'})
                return styler
            st.dataframe(overview_df.style.pipe(style_dark), use_container_width=True, hide_index=True)

    # TAB 2: TRAINING
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Trainiere Random Forest, Gradient Boosting & LSTM..."):
                results = trainer.run_benchmark(df_ai)
                st.session_state['trainer'] = trainer
                st.session_state['benchmark'] = results
            st.success("Benchmark abgeschlossen!")
        
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            best_model_name = min(res, key=lambda k: res[k]['rmse'])
            
            st.subheader(f"Gewinner: {best_model_name}")
            m1, m2 = st.columns(2)
            m1.metric("Best RMSE", f"{res[best_model_name]['rmse']:.2f}")
            m2.metric("Best R²", f"{res[best_model_name]['r2']:.2f}")
            st.info("Details siehe Tab 'Modell-Benchmark'")
        else:
            st.info("Bitte Training starten.")

    # TAB 3: SIMULATION (Restored & Upgraded)
    with tab3:
        if 'trainer' not in st.session_state:
            st.warning("Kein Modell trainiert. Bitte in Tab 2 trainieren.")
        else:
            subtab_sim, subtab_live = st.tabs(["Manueller Simulator", "Tages-Replay (Heute)"])
            
            # --- SUBTAB 1: MANUAL SIMULATOR ---
            with subtab_sim:
                st.markdown("Konfigurieren Sie ein hypothetisches Szenario und vergleichen Sie alle Modelle.")
                
                with st.container(border=True):
                    col_weather, col_time = st.columns(2)
                    with col_weather:
                        st.markdown("#### Wetterdaten")
                        s_temp = st.slider("Temperatur (°C)", -5, 40, 25, key="s_temp")
                        s_rain = st.slider("Regen (mm)", 0.0, 20.0, 0.0, key="s_rain")
                        s_cloud = st.slider("Bewölkung (%)", 0, 100, 20, key="s_cloud")
                        sim_hci = (4 * max(0, 10-abs(s_temp-25)*0.5)) + (2 * (100-s_cloud)/10) + (3 * max(0, 10-s_rain*2)) + 10
                        st.metric("HCI Score", f"{sim_hci:.1f}")

                    with col_time:
                        st.markdown("#### Zeit & Kalender")
                        s_hour = st.slider("Uhrzeit", 9, 20, 14, key="s_hour")
                        s_month = st.select_slider("Monat", options=list(range(1, 13)), value=7, key="s_month")
                        days = {0: "Montag", 1: "Dienstag", 2: "Mittwoch", 3: "Donnerstag", 4: "Freitag", 5: "Samstag", 6: "Sonntag"}
                        s_day_label = st.selectbox("Wochentag", list(days.values()), index=5, key="s_day")
                        s_weekday = list(days.keys())[list(days.values()).index(s_day_label)]
                        s_is_weekend = 1 if s_weekday >= 5 else 0
                        
                        c1, c2, c3 = st.columns(3)
                        s_hol_de = c1.checkbox("DE", key="h1")
                        s_hol_fr = c2.checkbox("FR", key="h2")
                        s_hol_ch = c3.checkbox("CH", key="h3")

                if st.button("Szenario Simulieren (Alle Modelle)", type="primary", use_container_width=True):
                    rides = df_ai['ride_name'].unique()
                    rows = []
                    
                    with st.spinner("Berechne Vorhersagen..."):
                        for r in rides:
                            try:
                                rid_meta = df_ai[df_ai['ride_name']==r].iloc[0]
                                inp = pd.DataFrame([{
                                    'hour': s_hour, 'weekday': s_weekday, 'month': s_month, 'is_weekend': s_is_weekend,
                                    'holiday_de_bw': int(s_hol_de), 'holiday_fr_zone_b': int(s_hol_fr), 'holiday_ch_bs': int(s_hol_ch),
                                    'temp': s_temp, 'rain': s_rain, 'HCI_Urban': sim_hci,
                                    'wait_time_lag_1': rid_meta['wait_time_lag_1'], 'wait_time_lag_6': rid_meta['wait_time_lag_6'],
                                    'ride_id': rid_meta['ride_id']
                                }])
                                
                                # Get predictions from ALL models
                                preds = st.session_state['trainer'].predict_ensemble(inp)
                                
                                row = {'Attraktion': r}
                                row.update(preds) # Add RF, GB, LSTM columns
                                rows.append(row)
                            except: continue
                    
                    if rows:
                        res_df = pd.DataFrame(rows).set_index('Attraktion')
                        st.dataframe(res_df.style.highlight_max(axis=0), use_container_width=True)

            # --- SUBTAB 2: REALITY CHECK ---
            with subtab_live:
                st.markdown("### Realitäts-Check: Heute")
                latest_date = df_raw['datetime'].max().date()
                df_today = df_raw[df_raw['datetime'].dt.date == latest_date].copy()
                
                if df_today.empty:
                    st.error("Keine Daten für Heute.")
                else:
                    # Determine open times
                    open_mask = df_today[df_today['is_open'] == True]
                    if not open_mask.empty:
                        start_time = open_mask['datetime'].min().strftime('%H:%M')
                        st.info(f"Parköffnung heute erkannt um: {start_time}")
                    
                    # Get latest snapshot
                    snapshot_now = df_today[df_today['datetime'] == df_today['datetime'].max()]
                    current_weather = snapshot_now.iloc[0]
                    
                    # Prepare input params from reality
                    current_hour = pd.to_datetime(current_weather['datetime']).hour
                    current_weekday = pd.to_datetime(current_weather['datetime']).weekday()
                    
                    rows_live = []
                    valid_rides = snapshot_now[snapshot_now['is_open']==True]['ride_name'].unique()
                    
                    with st.spinner("Vergleiche Modelle mit Realität..."):
                        for r in valid_rides:
                            try:
                                real_val = snapshot_now[snapshot_now['ride_name']==r]['wait_time'].values[0]
                                rid_meta = df_ai[df_ai['ride_name']==r].iloc[-1]
                                
                                inp_now = pd.DataFrame([{
                                    'hour': current_hour, 
                                    'weekday': current_weekday, 
                                    'month': pd.to_datetime(current_weather['datetime']).month, 
                                    'is_weekend': 1 if current_weekday >= 5 else 0,
                                    'holiday_de_bw': rid_meta['holiday_de_bw'],
                                    'holiday_fr_zone_b': rid_meta['holiday_fr_zone_b'],
                                    'holiday_ch_bs': rid_meta['holiday_ch_bs'],
                                    'temp': current_weather['temp'] if 'temp' in current_weather else 20, 
                                    'rain': current_weather['rain'] if 'rain' in current_weather else 0, 
                                    'HCI_Urban': hci_score,
                                    'wait_time_lag_1': rid_meta['wait_time_lag_1'],
                                    'wait_time_lag_6': rid_meta['wait_time_lag_6'],
                                    'ride_id': rid_meta['ride_id']
                                }])
                                
                                # Predict all models
                                preds = st.session_state['trainer'].predict_ensemble(inp_now)
                                
                                row = {'Attraktion': r, 'Realität (Ist)': real_val}
                                row.update(preds)
                                rows_live.append(row)
                            except: continue
                    
                    if rows_live:
                        live_df = pd.DataFrame(rows_live).set_index('Attraktion')
                        
                        st.markdown("#### Vergleichstabelle")
                        st.caption("Vergleich: Echte Wartezeit vs. KI-Prognosen")
                        st.dataframe(live_df, use_container_width=True)

    # TAB 4: BENCHMARK
    with tab4:
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            st.markdown("### Battle of Algorithms")
            
            metrics_data = []
            for name, metrics in res.items():
                metrics_data.append({
                    "Modell": name,
                    "RMSE": metrics['rmse'],
                    "R²": metrics['r2'],
                    "MAE": metrics['mae']
                })
            st.dataframe(pd.DataFrame(metrics_data).set_index("Modell"), use_container_width=True)
            
            st.markdown("### Visueller Vergleich (Test-Set)")
            df_plot = pd.DataFrame()
            first_key = list(res.keys())[0]
            limit = 50
            df_plot['Ground Truth'] = res[first_key]['actuals'][:limit]
            
            for name, metrics in res.items():
                df_plot[name] = metrics['predictions'][:limit]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0E1117')
            sns.lineplot(data=df_plot, ax=ax, linewidth=2)
            ax.set_ylabel("Wartezeit (min)", color='white')
            st.pyplot(fig)
        else:
            st.info("Bitte Modell im Tab 2 trainieren.")