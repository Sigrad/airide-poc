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
    tab1, tab2, tab3, tab4 = st.tabs(["Live Analyse", "KI & Modell Insights", "Prognose & Simulation", "Modell-Validierung"])

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

    # TAB 2: INSIGHTS & FEATURE ENGINEERING
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Trainiere Random Forest, Gradient Boosting & LSTM..."):
                results = trainer.run_benchmark(df_ai)
                st.session_state['trainer'] = trainer
                st.session_state['benchmark'] = results
            st.success("Training abgeschlossen!")
        
        # Display Feature Analysis regardless of model training
        st.subheader("Feature Engineering & Daten-Analyse")
        
        col_corr, col_dist = st.columns(2)
        
        with col_corr:
            st.markdown("**Korrelations-Matrix (Einflussfaktoren)**")
            st.caption("Wie stark hängen Wetter, Zeit und Wartezeiten zusammen?")
            # Filter numerical cols
            num_cols = ['wait_time', 'temp', 'rain', 'HCI_Urban', 'hour', 'month']
            corr = df_ai[num_cols].corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
            fig_corr.patch.set_facecolor('#0E1117')
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, cbar=False)
            st.pyplot(fig_corr)

        with col_dist:
            st.markdown("**Verteilung der Wartezeiten (Ground Truth)**")
            st.caption("Häufigkeitsverteilung der gemessenen Wartezeiten im Datensatz.")
            fig_dist, ax_dist = plt.subplots(figsize=(6, 5))
            fig_dist.patch.set_facecolor('#0E1117')
            sns.histplot(df_ai['wait_time'], bins=30, kde=True, color="#4c72b0", ax=ax_dist)
            ax_dist.set_xlabel("Wartezeit (Minuten)", color='white')
            st.pyplot(fig_dist)

        # Feature Importance (if model is trained)
        if 'benchmark' in st.session_state:
            st.subheader("Feature Importance (Random Forest)")
            # Extract Feature Importance from Random Forest (assumed to be available)
            # Since the new benchmark structure is different, we access the RF model directly if possible
            # or we re-calculate/access stored importance if saved. 
            # For simplicity, we assume the trainer object stores the RF model.
            
            if 'trainer' in st.session_state and 'rf' in st.session_state['trainer'].models:
                rf_model = st.session_state['trainer'].models['rf']
                features = ['hour', 'weekday', 'month', 'is_weekend', 
                            'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
                            'temp', 'rain', 'HCI_Urban', 
                            'wait_time_lag_1', 'wait_time_lag_6', 'ride_id']
                
                imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
                fig_imp.patch.set_facecolor('#0E1117')
                sns.barplot(data=imp_df, x='Importance', y='Feature', palette="magma", ax=ax_imp)
                ax_imp.set_xlabel("Signifikanz", color='white')
                ax_imp.set_ylabel("", color='white')
                st.pyplot(fig_imp)
        else:
            st.info("Trainieren Sie die Modelle, um die Feature Importance zu sehen.")

    # TAB 3: SIMULATION
    with tab3:
        if 'trainer' not in st.session_state:
            st.warning("Kein Modell trainiert. Bitte in Tab 2 trainieren.")
        else:
            subtab_sim, subtab_live = st.tabs(["Manueller Simulator", "Tages-Replay (Heute)"])
            
            # --- SUBTAB 1: MANUAL ---
            with subtab_sim:
                st.markdown("Konfigurieren Sie ein hypothetisches Szenario. Die Grafik vergleicht die Prognosen aller Modelle.")
                
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
                                
                                preds = st.session_state['trainer'].predict_ensemble(inp)
                                row = {'Attraktion': r}
                                row.update(preds)
                                rows.append(row)
                            except: continue
                    
                    if rows:
                        res_df = pd.DataFrame(rows)
                        
                        # Graph Visualization
                        st.markdown("#### Modell-Vergleich (Simulation)")
                        melted_sim = res_df.melt(id_vars='Attraktion', var_name='Modell', value_name='Minuten')
                        
                        fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
                        fig_sim.patch.set_facecolor('#0E1117')
                        sns.barplot(data=melted_sim, x='Minuten', y='Attraktion', hue='Modell', palette="viridis", ax=ax_sim)
                        ax_sim.set_ylabel("", color='white')
                        ax_sim.set_xlabel("Prognostizierte Wartezeit (min)", color='white')
                        st.pyplot(fig_sim)

                        # Data Table
                        st.dataframe(res_df.set_index('Attraktion'), use_container_width=True)

            # --- SUBTAB 2: REALITY CHECK ---
            with subtab_live:
                st.markdown("### Realitäts-Check: Heute")
                latest_date = df_raw['datetime'].max().date()
                df_today = df_raw[df_raw['datetime'].dt.date == latest_date].copy()
                
                if df_today.empty:
                    st.error("Keine Daten für Heute.")
                else:
                    snapshot_now = df_today[df_today['datetime'] == df_today['datetime'].max()]
                    current_weather = snapshot_now.iloc[0]
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
                                
                                preds = st.session_state['trainer'].predict_ensemble(inp_now)
                                row = {'Attraktion': r, 'Realität (Ist)': real_val}
                                row.update(preds)
                                rows_live.append(row)
                            except: continue
                    
                    if rows_live:
                        live_df = pd.DataFrame(rows_live)
                        
                        # Visualization
                        st.markdown("#### Vergleich: Modelle vs Realität")
                        st.caption("Vergleich aller Modelle gegen die echte Wartezeit (Momentaufnahme).")
                        
                        melted = live_df.melt(id_vars='Attraktion', var_name='Quelle', value_name='Minuten')
                        
                        fig_check, ax_check = plt.subplots(figsize=(12, 6))
                        fig_check.patch.set_facecolor('#0E1117')
                        
                        # Highlight Reality with specific color, models with palette
                        sns.barplot(data=melted, x='Minuten', y='Attraktion', hue='Quelle', ax=ax_check)
                        ax_check.set_ylabel("", color='white')
                        ax_check.set_xlabel("Minuten", color='white')
                        st.pyplot(fig_check)
                        
                        st.dataframe(live_df.set_index('Attraktion'), use_container_width=True)

    # TAB 4: BENCHMARK
    with tab4:
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            st.markdown("### Modell-Validierung & Performance")
            
            # 1. Metrics Comparison Chart
            metrics_list = []
            for name, metrics in res.items():
                metrics_list.append({'Modell': name, 'Metric': 'RMSE', 'Value': metrics['rmse']})
                metrics_list.append({'Modell': name, 'Metric': 'R²', 'Value': metrics['r2']})
            
            met_df = pd.DataFrame(metrics_list)
            
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.markdown("**Fehlermetrik (RMSE)** (niedriger = besser)")
                fig_rmse, ax_rmse = plt.subplots(figsize=(5, 3))
                fig_rmse.patch.set_facecolor('#0E1117')
                sns.barplot(data=met_df[met_df['Metric']=='RMSE'], x='Modell', y='Value', palette="Reds_d", ax=ax_rmse)
                ax_rmse.set_xlabel("", color='white')
                ax_rmse.set_ylabel("Minuten", color='white')
                st.pyplot(fig_rmse)
                
            with col_met2:
                st.markdown("**Erklärungsquote (R²)** (höher = besser)")
                fig_r2, ax_r2 = plt.subplots(figsize=(5, 3))
                fig_r2.patch.set_facecolor('#0E1117')
                sns.barplot(data=met_df[met_df['Metric']=='R²'], x='Modell', y='Value', palette="Greens_d", ax=ax_r2)
                ax_r2.set_xlabel("", color='white')
                ax_r2.set_ylabel("Score (0-1)", color='white')
                st.pyplot(fig_r2)
            
            st.markdown("---")
            
            # 2. Residual Plot (Professional Validation)
            st.markdown("### Fehlerverteilung (Residual Plot)")
            st.caption("Verteilung der Abweichungen (Ist - Soll). Eine Zentrierung um 0 deutet auf ein gutes Modell hin.")
            
            # Calculate residuals
            res_data = pd.DataFrame()
            first_key = list(res.keys())[0]
            # Use limited slice for cleaner plot
            limit = 200 
            actuals = res[first_key]['actuals'][:limit]
            
            for name, metrics in res.items():
                preds = metrics['predictions'][:limit]
                residuals = actuals - preds
                temp_df = pd.DataFrame({'Residuals': residuals, 'Modell': name})
                res_data = pd.concat([res_data, temp_df])
            
            fig_res, ax_res = plt.subplots(figsize=(10, 5))
            fig_res.patch.set_facecolor('#0E1117')
            sns.kdeplot(data=res_data, x='Residuals', hue='Modell', fill=True, alpha=0.3, ax=ax_res)
            ax_res.axvline(0, color='white', linestyle='--', linewidth=1)
            ax_res.set_xlabel("Abweichung in Minuten (Residual)", color='white')
            st.pyplot(fig_res)

            # 3. Line Chart
            st.markdown("### Zeitreihen-Vergleich")
            st.caption("Vergleich der Vorhersagen auf einem Ausschnitt des Test-Sets.")
            
            df_plot = pd.DataFrame()
            df_plot['Ground Truth'] = actuals[:50] # Zoom in
            for name, metrics in res.items():
                df_plot[name] = metrics['predictions'][:50]
            
            fig_line, ax_line = plt.subplots(figsize=(12, 5))
            fig_line.patch.set_facecolor('#0E1117')
            sns.lineplot(data=df_plot, ax=ax_line, linewidth=2)
            ax_line.set_ylabel("Wartezeit (min)", color='white')
            st.pyplot(fig_line)
            
        else:
            st.info("Bitte Modell im Tab 2 trainieren.")