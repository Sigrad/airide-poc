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
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1e1e1e", "grid.color": "#444444", "text.color": "white", "xtick.color": "white", "ytick.color": "white", "axes.labelcolor": "white"})

# --- KOPFZEILE ---
st.title("AIRide: Analyse und Prognose von Besucherströmen")
st.markdown("""
System zur Überwachung und prädiktiven Modellierung von Wartezeiten (Europa-Park).
Methodik: Ensemble-Learning (Random Forest, Gradient Boosting) und Deep Learning (LSTM) unter Einbezug des Holiday Climate Index (HCI).
""")

# --- SEITENLEISTE ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Datenbestand aktualisieren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modelle trainieren", type="primary")

st.sidebar.markdown("---")
status_text = "ONLINE (API)" if os.path.exists("real_waiting_times.csv") else "OFFLINE (Synthetisch)"
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

with st.spinner("Lade Datenpipeline..."):
    df_raw, df_ai = load_data_pipeline()

# --- HAUPTINHALT ---
if df_raw.empty:
    st.error("Keine Daten verfügbar.")
else:
    # --- KPI HEADER ---
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
    c1.metric("Letzte Aktualisierung", latest_ts.strftime('%H:%M'))
    c2.metric("Offene Attraktionen", len(open_rides))
    c3.metric("Ø Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    
    c4.metric(
        "HCI", 
        f"{hci_score:.0f}/100", 
        help="Der Holiday Climate Index (HCI) bewertet die klimatische Eignung für Freizeitaktivitäten. Das Modell integriert diesen Index mit kalendarischen Faktoren (Feiertage, Schulferien), um sowohl den meteorologischen als auch den saisonalen Einfluss auf die Besuchernachfrage abzubilden."
    )

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Echtzeit-Monitor", "Modell-Erkenntnisse", "Prognose & Simulation", "Validierung"])

    # TAB 1: ECHTZEIT
    with tab1:
        c_chart, c_table = st.columns([1.5, 1])
        with c_chart:
            st.subheader("Wartezeit (Top 15)")
            if not open_rides.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                fig.patch.set_facecolor('#0E1117') 
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False).head(15), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_ylabel("", color='white')
                ax.set_xlabel("Minuten", color='white')
                st.pyplot(fig)
            else:
                st.info("Keine Daten verfügbar.")

        with c_table:
            st.subheader("Attraktionen")
            overview_df = snapshot[['ride_name', 'is_open', 'wait_time']].sort_values('wait_time', ascending=False)
            
            # Status Mapping
            overview_df['Status'] = overview_df['is_open'].map({True: 'Offen', False: 'Geschlossen'})
            
            def style_status(val):
                color = '#ff4b4b' if val == 'Geschlossen' else '#09ab3b'
                return f'color: {color}'

            styled_df = overview_df[['ride_name', 'Status', 'wait_time']].style.map(style_status, subset=['Status'])
            
            st.dataframe(
                styled_df,
                column_config={
                    "ride_name": "Attraktion",
                    "Status": "Status",
                    "wait_time": st.column_config.ProgressColumn("Wartezeit", format="%d min", min_value=0, max_value=120)
                },
                use_container_width=True,
                hide_index=True,
                height=600
            )

    # TAB 2: INSIGHTS
    with tab2:
        if train_btn:
            trainer = PredictionModel()
            with st.spinner("Training läuft..."):
                results = trainer.run_benchmark(df_ai)
                st.session_state['trainer'] = trainer
                st.session_state['benchmark'] = results
            st.success("Training beendet.")
        
        # 1. Feature Importance
        if 'benchmark' in st.session_state and 'trainer' in st.session_state:
            st.subheader("Signifikanz der Einflussfaktoren")
            
            fi_col1, fi_col2 = st.columns(2)
            
            features = ['hour', 'weekday', 'month', 'is_weekend', 
                        'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
                        'temp', 'rain', 'HCI_Urban', 
                        'wait_time_lag_1', 'wait_time_lag_6', 'ride_id']
            
            feature_map_de = {
                'hour': 'Uhrzeit', 'weekday': 'Wochentag', 'month': 'Monat', 'is_weekend': 'Wochenende',
                'holiday_de_bw': 'Feiertag (DE)', 'holiday_fr_zone_b': 'Schulferien (FR)', 'holiday_ch_bs': 'Feiertag (CH)',
                'temp': 'Temperatur', 'rain': 'Niederschlag', 'HCI_Urban': 'HCI Index',
                'wait_time_lag_1': 'Latenz (10min)', 'wait_time_lag_6': 'Trend (1h)', 'ride_id': 'Attraktionstyp'
            }

            with fi_col1:
                if 'rf' in st.session_state['trainer'].models:
                    st.markdown("**Random Forest**")
                    rf_model = st.session_state['trainer'].models['rf']
                    imp_df_rf = pd.DataFrame({'Merkmal': [feature_map_de.get(f, f) for f in features], 'Wert': rf_model.feature_importances_}).sort_values('Wert', ascending=False).head(10)
                    
                    fig_imp1, ax_imp1 = plt.subplots(figsize=(5, 3))
                    fig_imp1.patch.set_facecolor('#0E1117')
                    sns.barplot(data=imp_df_rf, x='Wert', y='Merkmal', palette="magma", ax=ax_imp1)
                    ax_imp1.set_xlabel("Gini-Impurity", color='white')
                    ax_imp1.set_ylabel("", color='white')
                    st.pyplot(fig_imp1)

            with fi_col2:
                if 'gb' in st.session_state['trainer'].models:
                    st.markdown("**Gradient Boosting**")
                    gb_model = st.session_state['trainer'].models['gb']
                    imp_df_gb = pd.DataFrame({'Merkmal': [feature_map_de.get(f, f) for f in features], 'Wert': gb_model.feature_importances_}).sort_values('Wert', ascending=False).head(10)
                    
                    fig_imp2, ax_imp2 = plt.subplots(figsize=(5, 3))
                    fig_imp2.patch.set_facecolor('#0E1117')
                    sns.barplot(data=imp_df_gb, x='Wert', y='Merkmal', palette="viridis", ax=ax_imp2)
                    ax_imp2.set_xlabel("Relative Wichtigkeit", color='white')
                    ax_imp2.set_ylabel("", color='white')
                    st.pyplot(fig_imp2)
        else:
            st.info("Modelle müssen trainiert werden.")

        st.divider()
        
        # 2. Correlation & Distribution
        col_corr, col_dist = st.columns(2)
        with col_corr:
            st.markdown("**Korrelationsmatrix**")
            num_cols = ['wait_time', 'temp', 'rain', 'HCI_Urban', 'hour']
            corr = df_ai[num_cols].rename(columns={'wait_time': 'Wartezeit', 'temp': 'Temp', 'rain': 'Regen', 'hour': 'Std'}).corr()
            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
            fig_corr.patch.set_facecolor('#0E1117')
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, cbar=False)
            st.pyplot(fig_corr)

        with col_dist:
            st.markdown("**Verteilung (Empirie)**")
            fig_dist, ax_dist = plt.subplots(figsize=(5, 4))
            fig_dist.patch.set_facecolor('#0E1117')
            sns.histplot(df_ai['wait_time'], bins=30, kde=True, color="#4c72b0", ax=ax_dist)
            ax_dist.set_xlabel("Minuten", color='white')
            ax_dist.set_ylabel("Anzahl", color='white')
            st.pyplot(fig_dist)

    # TAB 3: PROGNOSE
    with tab3:
        if 'trainer' not in st.session_state:
            st.warning("Bitte Modelle in Tab 2 initialisieren.")
        else:
            subtab_sim, subtab_live = st.tabs(["Szenario-Simulation", "Echtzeit-Validierung"])
            
            # --- SUBTAB 1: SIMULATION ---
            with subtab_sim:
                with st.container(border=True):
                    c_w, c_t = st.columns(2)
                    with c_w:
                        s_temp = st.slider("Temperatur (°C)", -5, 40, 25)
                        s_rain = st.slider("Regen (mm)", 0.0, 20.0, 0.0)
                        s_cloud = st.slider("Bewölkung (%)", 0, 100, 20)
                        sim_hci = (4 * max(0, 10-abs(s_temp-25)*0.5)) + (2 * (100-s_cloud)/10) + (3 * max(0, 10-s_rain*2)) + 10
                        st.caption(f"HCI: {sim_hci:.1f}")
                    with c_t:
                        s_hour = st.slider("Stunde", 9, 20, 14)
                        s_weekday = st.slider("Wochentag (0=Mo, 6=So)", 0, 6, 5)
                        s_month = st.slider("Monat", 1, 12, 7)
                        
                        cc1, cc2, cc3 = st.columns(3)
                        s_hol_de = cc1.checkbox("Feiertag DE")
                        s_hol_fr = cc2.checkbox("Ferien FR")
                        s_hol_ch = cc3.checkbox("Feiertag CH")

                if st.button("Simulation starten", type="primary", use_container_width=True):
                    rides = df_ai['ride_name'].unique()
                    rows = []
                    for r in rides:
                        try:
                            rid_meta = df_ai[df_ai['ride_name']==r].iloc[0]
                            inp = pd.DataFrame([{
                                'hour': s_hour, 'weekday': s_weekday, 'month': s_month, 'is_weekend': 1 if s_weekday>=5 else 0,
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
                        melted_sim = res_df.melt(id_vars='Attraktion', var_name='Modell', value_name='Prognose')
                        
                        fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
                        fig_sim.patch.set_facecolor('#0E1117')
                        top_attractions = res_df.head(10)['Attraktion'].tolist()
                        melted_filtered = melted_sim[melted_sim['Attraktion'].isin(top_attractions)]
                        sns.barplot(data=melted_filtered, x='Prognose', y='Attraktion', hue='Modell', palette="viridis", ax=ax_sim)
                        ax_sim.set_ylabel("", color='white')
                        ax_sim.set_xlabel("Prognostizierte Wartezeit (Minuten)", color='white')
                        st.pyplot(fig_sim)

            # --- SUBTAB 2: REALITY CHECK ---
            with subtab_live:
                latest_date = df_raw['datetime'].max().date()
                df_today = df_raw[df_raw['datetime'].dt.date == latest_date].copy()
                
                if df_today.empty:
                    st.error("Keine Daten für Heute.")
                else:
                    snapshot_now = df_today[df_today['datetime'] == df_today['datetime'].max()]
                    current_weather = snapshot_now.iloc[0]
                    rows_live = []
                    valid_rides = snapshot_now[snapshot_now['is_open']==True]['ride_name'].unique()
                    
                    for r in valid_rides:
                        try:
                            real_val = snapshot_now[snapshot_now['ride_name']==r]['wait_time'].values[0]
                            rid_meta = df_ai[df_ai['ride_name']==r].iloc[-1]
                            inp_now = pd.DataFrame([{
                                'hour': pd.to_datetime(current_weather['datetime']).hour, 
                                'weekday': pd.to_datetime(current_weather['datetime']).weekday(), 
                                'month': pd.to_datetime(current_weather['datetime']).month, 
                                'is_weekend': 1 if pd.to_datetime(current_weather['datetime']).weekday() >= 5 else 0,
                                'holiday_de_bw': rid_meta['holiday_de_bw'], 'holiday_fr_zone_b': rid_meta['holiday_fr_zone_b'], 'holiday_ch_bs': rid_meta['holiday_ch_bs'],
                                'temp': current_weather['temp'], 'rain': current_weather['rain'], 'HCI_Urban': hci_score,
                                'wait_time_lag_1': rid_meta['wait_time_lag_1'], 'wait_time_lag_6': rid_meta['wait_time_lag_6'],
                                'ride_id': rid_meta['ride_id']
                            }])
                            preds = st.session_state['trainer'].predict_ensemble(inp_now)
                            row = {'Attraktion': r, 'Messwert (Ist)': real_val}
                            row.update(preds)
                            rows_live.append(row)
                        except: continue
                    
                    if rows_live:
                        live_df = pd.DataFrame(rows_live)
                        st.markdown(f"#### Diskrepanz-Analyse (Referenzzeitpunkt: {latest_date})")
                        melted = live_df.melt(id_vars='Attraktion', var_name='Datenquelle', value_name='Minuten')
                        
                        fig_check, ax_check = plt.subplots(figsize=(12, 6))
                        fig_check.patch.set_facecolor('#0E1117')
                        palette = {"Messwert (Ist)": "#ff4b4b", "Random Forest": "#4c72b0", "Gradient Boosting": "#55a868", "LSTM": "#8172b2"}
                        top_attractions_live = live_df.sort_values('Messwert (Ist)', ascending=False).head(10)['Attraktion'].tolist()
                        melted_filtered_live = melted[melted['Attraktion'].isin(top_attractions_live)]

                        sns.barplot(data=melted_filtered_live, x='Minuten', y='Attraktion', hue='Datenquelle', palette=palette, ax=ax_check)
                        ax_check.set_ylabel("", color='white')
                        ax_check.set_xlabel("Wartezeit (Minuten)", color='white')
                        st.pyplot(fig_check)

    # TAB 4: VALIDIERUNG
    with tab4:
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            
            # 1. KPI CARDS
            st.subheader("Performance-Metriken (Testdatensatz)")
            best_model_name = min(res, key=lambda k: res[k]['rmse'])
            cols = st.columns(len(res))
            
            for idx, (name, metrics) in enumerate(res.items()):
                with cols[idx]:
                    rmse_val = metrics['rmse']
                    r2_val = metrics['r2']
                    if name == best_model_name:
                        st.success(f"🏆 {name} (Beste Präzision)")
                        st.metric(label="RMSE (Fehler)", value=f"{rmse_val:.2f} min", delta="Minimum", delta_color="inverse")
                        st.metric(label="R² (Erklärungskraft)", value=f"{r2_val:.2f}", delta="Maximum")
                    else:
                        st.info(f"{name}")
                        st.metric(label="RMSE (Fehler)", value=f"{rmse_val:.2f} min")
                        st.metric(label="R² (Erklärungskraft)", value=f"{r2_val:.2f}")

            st.divider()

            # 2. Zeitreihen
            st.subheader("Zeitreihen-Validierung (Test-Sample)")
            first_key = list(res.keys())[0]
            limit = 100 
            actuals = res[first_key]['actuals'][:limit]
            
            df_plot = pd.DataFrame({'Messwert (Ist)': actuals})
            for name, metrics in res.items():
                df_plot[name] = metrics['predictions'][:limit]
            
            fig_line, ax_line = plt.subplots(figsize=(12, 4))
            fig_line.patch.set_facecolor('#0E1117')
            sns.lineplot(data=df_plot, ax=ax_line, linewidth=1.5)
            ax_line.set_ylabel("Minuten", color='white')
            ax_line.set_xlabel("Zeitpunkte", color='white')
            st.pyplot(fig_line)

            # 3. Residuen
            st.subheader("Residuenanalyse (Fehlerverteilung)")
            res_data = pd.DataFrame()
            limit_res = 300
            actuals_res = res[first_key]['actuals'][:limit_res]
            
            for name, metrics in res.items():
                residuals = actuals_res - metrics['predictions'][:limit_res]
                temp = pd.DataFrame({'Residuum': residuals, 'Algorithmus': name})
                res_data = pd.concat([res_data, temp])
            
            fig_res, ax_res = plt.subplots(figsize=(10, 4))
            fig_res.patch.set_facecolor('#0E1117')
            sns.kdeplot(data=res_data, x='Residuum', hue='Algorithmus', fill=True, alpha=0.3, ax=ax_res)
            ax_res.axvline(0, color='white', linestyle='--', linewidth=1)
            ax_res.set_xlabel("Abweichung (Minuten)", color='white')
            ax_res.set_ylabel("Dichte", color='white')
            st.pyplot(fig_res)
                
        else:
            st.info("Bitte Training durchführen.")