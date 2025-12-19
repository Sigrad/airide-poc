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
Dieses Tool nutzt **Random Forest Regression** und den **Holiday Climate Index (HCI)**.
""")

# --- SIDEBAR ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Daten aktualisieren / Cache leeren"):
    st.cache_data.clear()
    st.rerun()

# Source status
data_status = "ONLINE (Echtzeit-Daten)" if os.path.exists("real_waiting_times.csv") else "SIMULATION (Synthetische Daten)"
st.sidebar.info(f"Quelle: {data_status}")

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modell trainieren", type="primary")

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

# --- MAIN CONTENT ---
if df_raw.empty:
    st.error("Keine Daten verfügbar. Bitte aktivieren Sie den DataCollector.")
else:
    # --- GLOBAL METRICS ---
    latest_ts = df_raw['datetime'].max()
    snapshot = df_raw[df_raw['datetime'] == latest_ts]
    
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    count_open = len(open_rides)
    avg_wait = open_rides['wait_time'].mean() if count_open > 0 else 0
    temp_now = open_rides['temp'].mean() if 'temp' in open_rides.columns else 0
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Letztes Update", latest_ts.strftime('%H:%M:%S'))
    
    if count_open == 0:
        c2.error("Park Geschlossen")
    else:
        c2.metric("Offene Attraktionen", count_open)
        
    c3.metric("Durchschn. Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric("Aktueller HCI Score", f"{hci_score:.0f}/100", help="Holiday Climate Index: 100=Perfekt, 0=Schlecht")
    c5.metric("Datensätze (Total)", f"{len(df_raw):,}", f"{len(df_ai):,} Aktiv", help="Total geladene Historie vs. bereinigte Trainingsdaten.")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["Live Analyse", "KI & Modell Insights", "Prognose & Simulation"])

    # TAB 1: LIVE STATUS
    with tab1:
        c_chart, c_table = st.columns([1.5, 1])
        
        with c_chart:
            st.subheader("Aktuelle Wartezeiten (Top 15)")
            if count_open > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#0E1117') 
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False).head(15), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_xlabel("Minuten", color='white')
                ax.set_ylabel("", color='white')
                st.pyplot(fig)
            else:
                st.info("Keine offenen Attraktionen für das Diagramm verfügbar.")

        with c_table:
            st.subheader("Park Status")
            overview_df = snapshot[['ride_name', 'is_open', 'wait_time']].copy()
            overview_df = overview_df.sort_values(by=['is_open', 'wait_time', 'ride_name'], ascending=[False, False, True])
            
            def style_dark(styler):
                styler.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff', 'border-color': '#444444'})
                return styler

            st.dataframe(
                overview_df.style.pipe(style_dark), 
                column_config={
                    "ride_name": st.column_config.TextColumn("Attraktion"),
                    "is_open": st.column_config.CheckboxColumn("Geöffnet"),
                    "wait_time": st.column_config.ProgressColumn("Wartezeit", format="%d min", min_value=0, max_value=120)
                },
                hide_index=True, use_container_width=False, width=700, height=450
            )

    # TAB 2: AI INSIGHTS
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Trainiere Random Forest..."):
                results = trainer.train_and_evaluate(df_ai)
                st.session_state['model'] = results['model']
                st.session_state['metrics'] = results
            st.success("Training Abgeschlossen.")
        
        if 'metrics' in st.session_state:
            m = st.session_state['metrics']
            st.subheader("Modell Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{m['rmse']:.2f} min", help="Durchschnittlicher Fehler.")
            m2.metric("R² Score", f"{m['r2']:.2f}", help="Erklärungsquote.")
            m3.metric("Signifikanz (p-value)", f"{m['p_value']:.4f}", help="Diebold-Mariano Test.")
            
            st.subheader("Einflussfaktoren")
            fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
            fig_imp.patch.set_facecolor('#0E1117') 
            
            imp_df = m['feature_importance'].head(12).copy()
            feature_map = {
                'hour': 'Tageszeit (Stunde)', 'temp': 'Temperatur', 'ride_id': 'Attraktionstyp',
                'wait_time_lag_1': 'Momentum (10min)', 'wait_time_lag_6': 'Trend (1h)',          
                'HCI_Urban': 'Wetter Index (HCI)', 'weekday': 'Wochentag', 'is_weekend': 'Wochenende',               
                'month': 'Monat (Saison)', 'rain': 'Regenmenge', 'holiday_de_bw': 'Schulferien (DE)',
                'holiday_fr_zone_b': 'Schulferien (FR)', 'holiday_ch_bs': 'Schulferien (CH)'
            }
            imp_df['Feature'] = imp_df['Feature'].map(feature_map).fillna(imp_df['Feature'])
            sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette="magma", ax=ax_imp)
            ax_imp.set_xlabel("Einflussstärke", color='white')
            ax_imp.set_ylabel("Faktor", color='white')
            st.pyplot(fig_imp)
        else:
            st.info("Bitte trainieren Sie das Modell, um Analysen zu sehen.")

    # TAB 3: SIMULATION & VALIDATION
    with tab3:
        if 'model' not in st.session_state:
            st.warning("Modell nicht geladen. Bitte im Tab 'KI & Modell Insights' trainieren.")
        else:
            # Sub-tabs for distinct logic
            subtab_sim, subtab_live = st.tabs(["Manueller Simulator", "Tages-Replay (Heute vs. KI)"])
            
            # --- SUBTAB 1: MANUAL SIMULATOR ---
            with subtab_sim:
                st.markdown("Konfigurieren Sie ein hypothetisches Szenario.")
                
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

                if st.button("Szenario Simulieren", type="primary", use_container_width=True):
                    # Prediction Loop
                    rides = df_ai['ride_name'].unique()
                    preds = []
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
                            val = st.session_state['model'].predict(inp)[0]
                            preds.append({'Attraktion': r, 'Simulation': int(val)})
                        except: continue
                    
                    if preds:
                        sim_df = pd.DataFrame(preds).sort_values('Simulation', ascending=False)
                        st.dataframe(sim_df, use_container_width=True, hide_index=True)

            # --- SUBTAB 2: REALITY CHECK ---
            with subtab_live:
                st.markdown("### Realitäts-Check: Heute")
                st.markdown("Das Modell wird mit den **echten Wetter- und Zeitdaten von heute** gefüttert und gegen die Realität geprüft.")

                # Filter for today
                latest_date = df_raw['datetime'].max().date()
                df_today = df_raw[df_raw['datetime'].dt.date == latest_date].copy()
                
                if df_today.empty:
                    st.error("Keine Daten für den heutigen Tag gefunden.")
                else:
                    # Determine open/close times
                    open_mask = df_today[df_today['is_open'] == True]
                    
                    if open_mask.empty:
                        st.warning(f"Am {latest_date} war der Park scheinbar geschlossen (Keine offenen Attraktionen geloggt).")
                    else:
                        start_time = open_mask['datetime'].min().strftime('%H:%M')
                        current_time = df_today['datetime'].max().strftime('%H:%M')
                        
                        col_info1, col_info2, col_info3 = st.columns(3)
                        col_info1.metric("Datum", latest_date.strftime('%d.%m.%Y'))
                        col_info2.metric("Parköffnung (Erkannt)", start_time)
                        col_info3.metric("Aktueller Stand", current_time)
                        
                        st.divider()
                        
                        # Compare latest snapshot with AI prediction
                        snapshot_now = df_today[df_today['datetime'] == df_today['datetime'].max()]
                        
                        # Assume uniform weather across park
                        current_weather = snapshot_now.iloc[0] 
                        current_hour = pd.to_datetime(current_weather['datetime']).hour
                        current_weekday = pd.to_datetime(current_weather['datetime']).weekday()
                        current_month = pd.to_datetime(current_weather['datetime']).month
                        
                        preds_now = []
                        valid_rides_now = snapshot_now[snapshot_now['is_open']==True]['ride_name'].unique()
                        
                        for r in valid_rides_now:
                            try:
                                # Ground Truth
                                real_val = snapshot_now[snapshot_now['ride_name']==r]['wait_time'].values[0]
                                
                                # AI Prediction using metadata and current context
                                rid_meta = df_ai[df_ai['ride_name']==r].iloc[-1] 
                                
                                inp_now = pd.DataFrame([{
                                    'hour': current_hour, 
                                    'weekday': current_weekday, 
                                    'month': current_month, 
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
                                
                                pred_val = st.session_state['model'].predict(inp_now)[0]
                                
                                preds_now.append({
                                    'Attraktion': r,
                                    'KI-Prognose': int(pred_val),
                                    'Realität': int(real_val),
                                    'Abweichung': int(pred_val - real_val)
                                })
                            except: continue
                            
                        if preds_now:
                            res_df = pd.DataFrame(preds_now).set_index('Attraktion')
                            
                            st.markdown("#### Modell-Validierung (Live)")
                            st.caption("Vergleich der aktuellen echten Wartezeit mit der KI-Prognose.")
                            
                            # Chart
                            fig_check, ax_check = plt.subplots(figsize=(10, 5))
                            fig_check.patch.set_facecolor('#0E1117')
                            
                            melted_check = res_df.reset_index().melt(id_vars='Attraktion', value_vars=['KI-Prognose', 'Realität'], var_name='Quelle', value_name='Minuten')
                            sns.barplot(data=melted_check, x='Minuten', y='Attraktion', hue='Quelle', palette=['#4c72b0', '#55a868'], ax=ax_check)
                            ax_check.set_xlabel("Minuten", color='white')
                            ax_check.set_ylabel("", color='white')
                            st.pyplot(fig_check)
                            
                            # Table with Delta
                            st.dataframe(
                                res_df.sort_values('Abweichung', ascending=True),
                                column_config={
                                    "Abweichung": st.column_config.NumberColumn("Fehler (Delta)", format="%+d min")
                                },
                                use_container_width=True
                            )