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

# --- KONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")
# Setze Seaborn Design auf dunkel für Konsistenz
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "grid.color": "#444444", "text.color": "white", "xtick.color": "white", "ytick.color": "white", "axes.labelcolor": "white"})

# --- KOPFZEILE (HEADER) ---
st.title("AIRide: Analyse und Prognose von Besucherströmen")
st.markdown("""
Dieses Dashboard dient der **Überwachung und prädiktiven Modellierung** von Wartezeiten im Europa-Park.
Zur Anwendung kommen **Ensemble-Learning-Verfahren** (Random Forest, Gradient Boosting) sowie **rekurrente neuronale Netze** (LSTM) unter Einbezug des **Holiday Climate Index (HCI)**.
""")

# --- SEITENLEISTE (SIDEBAR) ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Datenbestand aktualisieren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modelle trainieren (Initialisierung)", type="primary")

st.sidebar.markdown("---")

# Statusanzeige Quelle
status_text = "ONLINE (Echtzeit-Schnittstelle)" if os.path.exists("real_waiting_times.csv") else "SIMULATION (Synthetische Datenbasis)"
st.sidebar.info(f"Datenquelle: {status_text}")

# --- DATENLADEN ---
@st.cache_data(ttl=60)
def load_data_pipeline(days_back=60):
    collector = DataCollector()
    df_raw = collector.fetch_historical_data(days_back=days_back)
    if df_raw.empty: return pd.DataFrame(), pd.DataFrame()
    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    return df_raw, df_processed

with st.spinner("Initialisiere Datenpipeline..."):
    df_raw, df_ai = load_data_pipeline()

# --- HAUPTINHALT ---
if df_raw.empty:
    st.error("Keine Daten verfügbar. Bitte starten Sie den Datensammler (DataCollector).")
else:
    # --- KPI METRIKEN ---
    latest_ts = df_raw['datetime'].max()
    snapshot = df_raw[df_raw['datetime'] == latest_ts]
    
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    avg_wait = open_rides['wait_time'].mean() if not open_rides.empty else 0
    temp_now = open_rides['temp'].mean() if 'temp' in open_rides.columns else 0
    # HCI Berechnung (Approximation für KPI)
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letzte Aktualisierung", latest_ts.strftime('%H:%M'))
    c2.metric("Aktive Attraktionen", len(open_rides))
    c3.metric("Ø Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric("HCI (Klima-Index)", f"{hci_score:.0f}/100", help="Der Holiday Climate Index (HCI) quantifiziert die Eignung der Witterung für Freizeitaktivitäten (0=Ungünstig, 100=Ideal).")

    st.markdown("---")

    # --- REGISTERKARTEN (TABS) ---
    # Übersetzung der Tab-Namen in akademisches Deutsch
    tab1, tab2, tab3, tab4 = st.tabs(["Echtzeit-Monitor", "Modell-Erkenntnisse", "Prognose & Simulation", "Validierung & Benchmarking"])

    # TAB 1: ECHTZEIT-MONITOR
    with tab1:
        c_chart, c_table = st.columns([1.5, 1])
        with c_chart:
            st.subheader("Wartezeit-Verteilung (Top 15)")
            if not open_rides.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#0E1117') 
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False).head(15), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_ylabel("", color='white')
                ax.set_xlabel("Wartezeit in Minuten", color='white')
                st.pyplot(fig)
            else:
                st.info("Der Park ist derzeit geschlossen oder es liegen keine Daten vor.")

        with c_table:
            st.subheader("Tabellarische Übersicht")
            overview_df = snapshot[['ride_name', 'is_open', 'wait_time']].sort_values('wait_time', ascending=False)
            
            # Umbenennung für die Anzeige
            overview_df.columns = ['Attraktion', 'Status', 'Wartezeit']
            
            def style_dark(styler):
                styler.set_properties(**{'background-color': '#1e1e1e', 'color': '#ffffff', 'border-color': '#444444'})
                return styler
            st.dataframe(overview_df.style.pipe(style_dark), use_container_width=True, hide_index=True)

    # TAB 2: MODELL-ERKENNTNISSE
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Training der Algorithmen (Random Forest, Gradient Boosting, LSTM)..."):
                results = trainer.run_benchmark(df_ai)
                st.session_state['trainer'] = trainer
                st.session_state['benchmark'] = results
            st.success("Modellierung erfolgreich abgeschlossen.")
        
        st.subheader("Explorative Datenanalyse & Merkmalsrelevanz")
        
        col_corr, col_dist = st.columns(2)
        
        with col_corr:
            st.markdown("**Korrelationsmatrix**")
            st.caption("Pearson-Korrelationskoeffizienten ausgewählter Variablen.")
            # Filter auf numerische Spalten
            num_cols = ['wait_time', 'temp', 'rain', 'HCI_Urban', 'hour', 'month']
            # Umbenennung für Plot
            rename_map = {'wait_time': 'Wartezeit', 'temp': 'Temperatur', 'rain': 'Niederschlag', 'hour': 'Stunde', 'month': 'Monat'}
            corr = df_ai[num_cols].rename(columns=rename_map).corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
            fig_corr.patch.set_facecolor('#0E1117')
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, cbar=False)
            st.pyplot(fig_corr)

        with col_dist:
            st.markdown("**Verteilung der Wartezeiten (Empirie)**")
            st.caption("Histogramm der historisch gemessenen Wartezeiten.")
            fig_dist, ax_dist = plt.subplots(figsize=(6, 5))
            fig_dist.patch.set_facecolor('#0E1117')
            sns.histplot(df_ai['wait_time'], bins=30, kde=True, color="#4c72b0", ax=ax_dist)
            ax_dist.set_xlabel("Wartezeit (Minuten)", color='white')
            ax_dist.set_ylabel("Häufigkeit", color='white')
            st.pyplot(fig_dist)

        # Feature Importance
        if 'benchmark' in st.session_state:
            st.subheader("Signifikanz der Einflussfaktoren (Random Forest)")
            
            if 'trainer' in st.session_state and 'rf' in st.session_state['trainer'].models:
                rf_model = st.session_state['trainer'].models['rf']
                features = ['hour', 'weekday', 'month', 'is_weekend', 
                            'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
                            'temp', 'rain', 'HCI_Urban', 
                            'wait_time_lag_1', 'wait_time_lag_6', 'ride_id']
                
                # Mapping auf Deutsch
                feature_map_de = {
                    'hour': 'Uhrzeit', 'weekday': 'Wochentag', 'month': 'Monat', 'is_weekend': 'Wochenende',
                    'holiday_de_bw': 'Feiertag (DE)', 'holiday_fr_zone_b': 'Schulferien (FR)', 'holiday_ch_bs': 'Feiertag (CH)',
                    'temp': 'Temperatur', 'rain': 'Niederschlag', 'HCI_Urban': 'HCI Index',
                    'wait_time_lag_1': 'Systemträgheit (10min)', 'wait_time_lag_6': 'Trend (1h)', 'ride_id': 'Attraktionstyp'
                }
                
                imp_df = pd.DataFrame({
                    'Merkmal': [feature_map_de.get(f, f) for f in features],
                    'Relevanz': rf_model.feature_importances_
                }).sort_values('Relevanz', ascending=False).head(10)
                
                fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
                fig_imp.patch.set_facecolor('#0E1117')
                sns.barplot(data=imp_df, x='Relevanz', y='Merkmal', palette="magma", ax=ax_imp)
                ax_imp.set_xlabel("Gini-Impurity Reduktion (Relative Wichtigkeit)", color='white')
                ax_imp.set_ylabel("", color='white')
                st.pyplot(fig_imp)
        else:
            st.info("Bitte führen Sie das Training durch, um die Merkmalsanalyse einzusehen.")

    # TAB 3: PROGNOSE & SIMULATION
    with tab3:
        if 'trainer' not in st.session_state:
            st.warning("Kein trainiertes Modell gefunden. Bitte Initialisierung in Tab 2 durchführen.")
        else:
            subtab_sim, subtab_live = st.tabs(["Szenario-Simulation (Hypothetisch)", "Echtzeit-Validierung (Aktuell)"])
            
            # --- SUBTAB 1: MANUELLE SIMULATION ---
            with subtab_sim:
                st.markdown("Definition eines hypothetischen Szenarios zur **Sensitivitätsanalyse** der Modelle.")
                
                with st.container(border=True):
                    col_weather, col_time = st.columns(2)
                    with col_weather:
                        st.markdown("#### Meteorologische Parameter")
                        s_temp = st.slider("Temperatur (°C)", -5, 40, 25, key="s_temp")
                        s_rain = st.slider("Niederschlag (mm)", 0.0, 20.0, 0.0, key="s_rain")
                        s_cloud = st.slider("Bewölkungsgrad (%)", 0, 100, 20, key="s_cloud")
                        sim_hci = (4 * max(0, 10-abs(s_temp-25)*0.5)) + (2 * (100-s_cloud)/10) + (3 * max(0, 10-s_rain*2)) + 10
                        st.metric("Resultierender HCI", f"{sim_hci:.1f}")

                    with col_time:
                        st.markdown("#### Temporale Parameter")
                        s_hour = st.slider("Tageszeit", 9, 20, 14, key="s_hour")
                        s_month = st.select_slider("Monat", options=list(range(1, 13)), value=7, key="s_month")
                        days = {0: "Montag", 1: "Dienstag", 2: "Mittwoch", 3: "Donnerstag", 4: "Freitag", 5: "Samstag", 6: "Sonntag"}
                        s_day_label = st.selectbox("Wochentag", list(days.values()), index=5, key="s_day")
                        s_weekday = list(days.keys())[list(days.values()).index(s_day_label)]
                        s_is_weekend = 1 if s_weekday >= 5 else 0
                        
                        c1, c2, c3 = st.columns(3)
                        s_hol_de = c1.checkbox("Feiertag DE", key="h1")
                        s_hol_fr = c2.checkbox("Ferien FR", key="h2")
                        s_hol_ch = c3.checkbox("Feiertag CH", key="h3")

                if st.button("Simulation starten (Alle Algorithmen)", type="primary", use_container_width=True):
                    rides = df_ai['ride_name'].unique()
                    rows = []
                    
                    with st.spinner("Berechne Inferenz..."):
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
                        
                        st.markdown("#### Modellvergleich (Simulierte Prognose)")
                        melted_sim = res_df.melt(id_vars='Attraktion', var_name='Algorithmus', value_name='Prognose (Min)')
                        
                        fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
                        fig_sim.patch.set_facecolor('#0E1117')
                        sns.barplot(data=melted_sim, x='Prognose (Min)', y='Attraktion', hue='Algorithmus', palette="viridis", ax=ax_sim)
                        ax_sim.set_ylabel("", color='white')
                        ax_sim.set_xlabel("Prognostizierte Wartezeit (Minuten)", color='white')
                        st.pyplot(fig_sim)

                        st.dataframe(res_df.set_index('Attraktion'), use_container_width=True)

            # --- SUBTAB 2: ECHTZEIT-VALIDIERUNG ---
            with subtab_live:
                st.markdown("### Abgleich mit Echtzeitdaten")
                latest_date = df_raw['datetime'].max().date()
                df_today = df_raw[df_raw['datetime'].dt.date == latest_date].copy()
                
                if df_today.empty:
                    st.error("Für das heutige Datum liegen keine Datensätze vor.")
                else:
                    # Parköffnung ermitteln
                    open_mask = df_today[df_today['is_open'] == True]
                    if not open_mask.empty:
                        start_time = open_mask['datetime'].min().strftime('%H:%M')
                        st.info(f"Erkannte Betriebsaufnahme heute: {start_time}")
                    
                    # Letzten Snapshot holen
                    snapshot_now = df_today[df_today['datetime'] == df_today['datetime'].max()]
                    current_weather = snapshot_now.iloc[0]
                    current_hour = pd.to_datetime(current_weather['datetime']).hour
                    current_weekday = pd.to_datetime(current_weather['datetime']).weekday()
                    
                    rows_live = []
                    valid_rides = snapshot_now[snapshot_now['is_open']==True]['ride_name'].unique()
                    
                    with st.spinner("Validiere Modelle gegen Ist-Zustand..."):
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
                                row = {'Attraktion': r, 'Messwert (Ist)': real_val}
                                row.update(preds)
                                rows_live.append(row)
                            except: continue
                    
                    if rows_live:
                        live_df = pd.DataFrame(rows_live)
                        
                        st.markdown("#### Diskrepanz-Analyse: Modell vs. Empirie")
                        st.caption("Vergleich der algorithmischen Vorhersagen mit den tatsächlich gemessenen Wartezeiten.")
                        
                        melted = live_df.melt(id_vars='Attraktion', var_name='Datenquelle', value_name='Minuten')
                        
                        fig_check, ax_check = plt.subplots(figsize=(12, 6))
                        fig_check.patch.set_facecolor('#0E1117')
                        
                        sns.barplot(data=melted, x='Minuten', y='Attraktion', hue='Datenquelle', ax=ax_check)
                        ax_check.set_ylabel("", color='white')
                        ax_check.set_xlabel("Wartezeit (Minuten)", color='white')
                        st.pyplot(fig_check)
                        
                        st.dataframe(live_df.set_index('Attraktion'), use_container_width=True)

    # TAB 4: VALIDIERUNG & BENCHMARKING
    with tab4:
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            st.markdown("### Vergleichende Evaluation der Algorithmen")
            
            # 1. Metriken Charts
            metrics_list = []
            for name, metrics in res.items():
                metrics_list.append({'Algorithmus': name, 'Metrik': 'RMSE', 'Wert': metrics['rmse']})
                metrics_list.append({'Algorithmus': name, 'Metrik': 'R²', 'Wert': metrics['r2']})
            
            met_df = pd.DataFrame(metrics_list)
            
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.markdown("**Fehlermetrik (RMSE)** (Niedriger ist besser)")
                fig_rmse, ax_rmse = plt.subplots(figsize=(5, 3))
                fig_rmse.patch.set_facecolor('#0E1117')
                sns.barplot(data=met_df[met_df['Metrik']=='RMSE'], x='Algorithmus', y='Wert', palette="Reds_d", ax=ax_rmse)
                ax_rmse.set_xlabel("", color='white')
                ax_rmse.set_ylabel("Fehler in Minuten", color='white')
                st.pyplot(fig_rmse)
                
            with col_met2:
                st.markdown("**Bestimmtheitsmaß (R²)** (Höher ist besser)")
                fig_r2, ax_r2 = plt.subplots(figsize=(5, 3))
                fig_r2.patch.set_facecolor('#0E1117')
                sns.barplot(data=met_df[met_df['Metrik']=='R²'], x='Algorithmus', y='Wert', palette="Greens_d", ax=ax_r2)
                ax_r2.set_xlabel("", color='white')
                ax_r2.set_ylabel("Score (0-1)", color='white')
                st.pyplot(fig_r2)
            
            st.markdown("---")
            
            # 2. Residuenanalyse
            st.markdown("### Residuenanalyse (Fehlerverteilung)")
            st.caption("Kerndichteschätzung der Prognosefehler (Residuals = Ist - Soll). Eine Zentrierung um 0 indiziert hohe Modellgüte ohne systematischen Bias.")
            
            res_data = pd.DataFrame()
            first_key = list(res.keys())[0]
            limit = 200 
            actuals = res[first_key]['actuals'][:limit]
            
            for name, metrics in res.items():
                preds = metrics['predictions'][:limit]
                residuals = actuals - preds
                temp_df = pd.DataFrame({'Residuum': residuals, 'Algorithmus': name})
                res_data = pd.concat([res_data, temp_df])
            
            fig_res, ax_res = plt.subplots(figsize=(10, 5))
            fig_res.patch.set_facecolor('#0E1117')
            sns.kdeplot(data=res_data, x='Residuum', hue='Algorithmus', fill=True, alpha=0.3, ax=ax_res)
            ax_res.axvline(0, color='white', linestyle='--', linewidth=1)
            ax_res.set_xlabel("Abweichung in Minuten", color='white')
            ax_res.set_ylabel("Dichte", color='white')
            st.pyplot(fig_res)

            # 3. Zeitreihen-Plot
            st.markdown("### Zeitreihen-Validierung (Test-Sample)")
            st.caption("Visueller Vergleich der Modellprognosen gegenüber den empirischen Messwerten auf einem Ausschnitt des Testdatensatzes.")
            
            df_plot = pd.DataFrame()
            df_plot['Messwert (Ist)'] = actuals[:50] 
            for name, metrics in res.items():
                df_plot[name] = metrics['predictions'][:50]
            
            fig_line, ax_line = plt.subplots(figsize=(12, 5))
            fig_line.patch.set_facecolor('#0E1117')
            sns.lineplot(data=df_plot, ax=ax_line, linewidth=2)
            ax_line.set_ylabel("Wartezeit (Minuten)", color='white')
            ax_line.set_xlabel("Zeitpunkte (Test-Sequenz)", color='white')
            st.pyplot(fig_line)
            
        else:
            st.info("Bitte initialisieren Sie das Training im Reiter 'Modell-Erkenntnisse'.")