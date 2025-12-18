import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

# --- CONFIGURATION ---
st.set_page_config(page_title="AIRide Analyse", layout="wide")
sns.set_theme(style="whitegrid") 

# --- HEADER SECTION ---
st.title("AIRide: Analyse von Besucherströmen & Wartezeiten")
st.markdown("""
Dashboard zur Überwachung und Prognose von Besucherströmen von Europa Park.
Dieses Tool nutzt **Random Forest Regression** und den **Holiday Climate Index (HCI)**.
""")

# --- SIDEBAR CONTROL ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Daten aktualisieren / Cache leeren"):
    st.cache_data.clear()
    st.rerun()

# Data Source Status
data_status = "ONLINE (Echtzeit-Daten)" if os.path.exists("real_waiting_times.csv") else "SIMULATION (Synthetische Daten)"
st.sidebar.info(f"Quelle: {data_status}")

st.sidebar.subheader("Modell-Konfiguration")
train_btn = st.sidebar.button("Modell trainieren", type="primary")

# --- DATA LOADING ---
@st.cache_data(ttl=60)
def load_data_pipeline(days_back=60):
    collector = DataCollector()
    # 1. Fetch raw data
    df_raw = collector.fetch_historical_data(days_back=days_back)
    
    if df_raw.empty: return pd.DataFrame(), pd.DataFrame()
        
    # 2. Process data
    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    
    return df_raw, df_processed

with st.spinner("Verarbeite Datenpipeline..."):
    df_raw, df_ai = load_data_pipeline()

# --- MAIN CONTENT ---
if df_raw.empty:
    st.error("Keine Daten verfügbar. Bitte aktivieren Sie den DataCollector.")
else:
    # --- KPI CALCULATION ---
    latest_ts = df_raw['datetime'].max()
    snapshot = df_raw[df_raw['datetime'] == latest_ts]
    
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    count_open = len(open_rides)
    avg_wait = open_rides['wait_time'].mean() if count_open > 0 else 0
    temp_now = open_rides['temp'].mean() if 'temp' in open_rides.columns else 0
    
    # Calculate HCI approximation
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    # Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Letztes Update", latest_ts.strftime('%H:%M:%S'))
    
    if count_open == 0:
        c2.error("Park Geschlossen")
    else:
        c2.metric("Offene Attraktionen", count_open)
        
    c3.metric("Ø Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric("Aktueller HCI Score", f"{hci_score:.0f}/100", help="Holiday Climate Index: 100=Perfekt, 0=Schlecht")
    c5.metric("Datensätze (Total)", f"{len(df_raw):,}", f"{len(df_ai):,} Aktiv", help="Total geladene Historie vs. bereinigte Trainingsdaten (nur offene Bahnen).")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["Live Analyse", "KI & Modell Insights", "Prognose Simulator & Vergleich"])

    # TAB 1: LIVE STATUS
    with tab1:
        if count_open > 0:
            c_chart, c_table = st.columns([2, 1])
            
            with c_chart:
                st.subheader("Aktuelle Wartezeiten")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=open_rides.sort_values('wait_time', ascending=False), 
                            x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_xlabel("Minuten")
                ax.set_ylabel("")
                st.pyplot(fig)
                st.caption("Echtzeit-Auslastung der offenen Attraktionen.")
                
            with c_table:
                st.subheader("Rohdaten Log")
                
                # Prepare Table Data
                log_df = df_raw.tail(15)[['datetime', 'ride_name', 'wait_time', 'is_open']].copy()
                
                # Format Timestamp (remove milliseconds)
                log_df['datetime'] = log_df['datetime'].dt.strftime('%H:%M:%S')
                
                # Display with nice formatting
                st.dataframe(
                    log_df,
                    column_config={
                        "datetime": "Zeitpunkt",
                        "ride_name": "Attraktion",
                        "wait_time": st.column_config.NumberColumn(
                            "Wartezeit", format="%d min"
                        ),
                        "is_open": st.column_config.CheckboxColumn(
                            "Status",
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("Park ist derzeit geschlossen. Historisches Log wird unten angezeigt.")
            
            # Same formatting for closed state log
            log_df = df_raw.tail(20)[['datetime', 'ride_name', 'wait_time', 'is_open']].copy()
            log_df['datetime'] = log_df['datetime'].dt.strftime('%d.%m. %H:%M')
            
            st.dataframe(
                log_df,
                column_config={
                    "datetime": "Zeitstempel",
                    "ride_name": "Attraktion",
                    "wait_time": st.column_config.NumberColumn("Wartezeit", format="%d min"),
                    "is_open": st.column_config.CheckboxColumn("Geöffnet?")
                },
                hide_index=True,
                use_container_width=True
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
            m1.metric("RMSE", f"{m['rmse']:.2f} min", help="Root Mean Squared Error")
            m2.metric("R² Score", f"{m['r2']:.2f}", help="Bestimmtheitsmaß (1.0 = Perfekt)")
            m3.metric("Signifikanz (p-value)", f"{m['p_value']:.4f}", help="Diebold-Mariano Test")
            
            st.subheader("Einflussfaktoren (Feature Importance)")
            st.markdown("Welche Faktoren beeinflussen die Wartezeit am stärksten?")
            
            fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
            
            # Mapping feature names to German
            imp_df = m['feature_importance'].head(10).copy()
            feature_map = {
                'hour': 'Tageszeit (Stunde)', 
                'temp': 'Temperatur', 
                'ride_id': 'Attraktionstyp',
                'wait_time_lag_1': 'Vorherige Wartezeit',
                'HCI_Urban': 'Wetter Index (HCI)',
                'weekday': 'Wochentag',
                'rain': 'Regenmenge',
                'holiday_de_bw': 'Schulferien (BW)',
                'holiday_fr_zone_b': 'Schulferien (FR)',
                'holiday_ch_bs': 'Feiertage (CH)'
            }
            imp_df['Feature'] = imp_df['Feature'].map(feature_map).fillna(imp_df['Feature'])
            
            sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette="magma", ax=ax_imp)
            st.pyplot(fig_imp)
            st.caption("Visualisierung der Random Forest Entscheidungsbaum-Gewichtung.")
        else:
            st.info("Bitte trainieren Sie das Modell, um Analysen zu sehen.")

    # TAB 3: SIMULATOR
    with tab3:
        if 'model' in st.session_state:
            st.subheader("Szenario-Analyse & Vergleich")
            st.markdown("Vergleichen Sie die **simulierte Prognose** mit dem **letzten realen Betriebszustand**.")
            
            col_controls, col_vis = st.columns([1, 2])
            
            with col_controls:
                with st.container(border=True):
                    st.markdown("#### Simulations-Parameter")
                    s_temp = st.slider("Temperatur (°C)", 0, 40, 25)
                    s_rain = st.slider("Regen (mm)", 0.0, 15.0, 0.0)
                    s_cloud = st.slider("Bewölkung (%)", 0, 100, 20)
                    
                    sim_hci = (4 * max(0, 10-abs(s_temp-25)*0.5)) + (2 * (100-s_cloud)/10) + (3 * max(0, 10-s_rain*2)) + 10
                    st.metric("Simulierter HCI", f"{sim_hci:.1f}/100")
            
            # --- FIND REFERENCE DATA ---
            if 'is_open' in df_raw.columns:
                valid_data = df_raw[df_raw['is_open'] == True]
            else:
                valid_data = df_raw[df_raw['wait_time'] > 0]
            
            if not valid_data.empty:
                last_open_ts = valid_data['datetime'].max()
                last_open_snapshot = df_raw[df_raw['datetime'] == last_open_ts]
                reference_status = last_open_snapshot[['ride_name', 'wait_time']].set_index('ride_name')
                ref_label_short = "Basis"
                ref_label_long = f"Letzter Real-Stand ({last_open_ts.strftime('%d.%m. %H:%M')})"
            else:
                reference_status = pd.DataFrame(columns=['wait_time'])
                ref_label_short = "N/A"
                ref_label_long = "Keine Daten"

            # --- PREDICTION LOGIC ---
            rides = df_ai['ride_name'].unique()
            preds = []
            
            for r in rides:
                try:
                    rid_meta = df_ai[df_ai['ride_name']==r].iloc[0]
                    inp = pd.DataFrame([{
                        'hour': 14, 'weekday': 5, 'month': 7, 'is_weekend': 1,
                        'holiday_de_bw': 0, 'holiday_fr_zone_b': 0, 'holiday_ch_bs': 0,
                        'temp': s_temp, 'rain': s_rain, 'HCI_Urban': sim_hci,
                        'wait_time_lag_1': rid_meta['wait_time_lag_1'],
                        'wait_time_lag_6': rid_meta['wait_time_lag_6'],
                        'ride_id': rid_meta['ride_id']
                    }])
                    val = st.session_state['model'].predict(inp)[0]
                    preds.append({'Attraktion': r, 'Simulation': int(val)})
                except: continue
            
            # --- MERGE & VISUALIZE ---
            if preds:
                sim_df = pd.DataFrame(preds).set_index('Attraktion')
                comp_df = sim_df.join(reference_status, how='left').fillna(0)
                comp_df = comp_df.rename(columns={'wait_time': ref_label_short})
                comp_df['Differenz'] = comp_df['Simulation'] - comp_df[ref_label_short]
                
                with col_vis:
                    avg_diff = comp_df['Differenz'].mean()
                    delta_color = "normal" if avg_diff < 0 else "inverse"
                    
                    st.markdown(f"#### Vergleich: Simulation vs. {ref_label_long}")
                    m1, m2 = st.columns(2)
                    m1.metric("Ø Wartezeit (Simuliert)", f"{comp_df['Simulation'].mean():.1f} min")
                    m2.metric("Veränderung zur Basis", f"{avg_diff:+.1f} min", delta_color=delta_color)
                    
                    top_comp = comp_df.sort_values(ref_label_short, ascending=False).head(10).reset_index()
                    melted = top_comp.melt(id_vars='Attraktion', value_vars=[ref_label_short, 'Simulation'], var_name='Szenario', value_name='Minuten')
                    
                    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
                    sns.barplot(data=melted, x='Minuten', y='Attraktion', hue='Szenario', palette=["#9e9e9e", "#4c72b0"], ax=ax_comp)
                    ax_comp.set_ylabel("")
                    st.pyplot(fig_comp)

            # Detailed Table
            st.markdown("#### Detaillierte Prognose-Tabelle")
            display_df = comp_df.sort_values('Simulation', ascending=False).reset_index()
            
            st.dataframe(
                display_df,
                column_config={
                    "Attraktion": st.column_config.TextColumn("Attraktion"),
                    ref_label_short: st.column_config.NumberColumn("Basis (Ist)", format="%d min"),
                    "Simulation": st.column_config.ProgressColumn(
                        "Simulation (Soll)", format="%d min", min_value=0, max_value=120
                    ),
                    "Differenz": st.column_config.NumberColumn("Delta", format="%+d min")
                },
                hide_index=True,
                use_container_width=True
            )
            
        else:
            st.warning("Modell erforderlich. Bitte in Sidebar/Tab 2 trainieren.")