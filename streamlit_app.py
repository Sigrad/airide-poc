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
Dieses Tool nutzt **Random Forest**, **Gradient Boosting** und **Deep Learning (LSTM)**.
""")

# --- SIDEBAR ---
st.sidebar.header("Systemsteuerung")
if st.sidebar.button("Daten aktualisieren / Cache leeren"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Modell-Konfiguration")
# REMOVED: (Benchmark) label
train_btn = st.sidebar.button("Modelle trainieren", type="primary")

# Visual separator to push Source info to bottom
st.sidebar.markdown("---")

# MOVED: Source status to bottom
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
    c1.metric("Update", latest_ts.strftime('%H:%M'))
    c2.metric("Offen", len(open_rides))
    c3.metric("Ø Wartezeit", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric("HCI Score", f"{hci_score:.0f}/100")

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

    # TAB 2: TRAINING LOGIC
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

    # TAB 3: SIMULATOR (Using Ensemble or Best Model)
    with tab3:
        if 'trainer' not in st.session_state:
            st.warning("Kein Modell trainiert.")
        else:
            st.markdown("### Simulator (Multi-Model Check)")
            with st.container(border=True):
                c_w, c_t = st.columns(2)
                s_temp = c_w.slider("Temp", 0, 40, 25)
                s_hour = c_t.slider("Stunde", 9, 20, 14)
                
            if st.button("Prognose für 'Silver Star'"):
                # Mockup input based on Silver Star metadata
                rides = df_ai[df_ai['ride_name'].str.contains("Silver Star")]
                if not rides.empty:
                    meta = rides.iloc[-1]
                    inp = pd.DataFrame([{
                        'hour': s_hour, 'weekday': 5, 'month': 7, 'is_weekend': 1,
                        'holiday_de_bw': 0, 'holiday_fr_zone_b': 0, 'holiday_ch_bs': 0,
                        'temp': s_temp, 'rain': 0, 'HCI_Urban': 80,
                        'wait_time_lag_1': meta['wait_time_lag_1'], 
                        'wait_time_lag_6': meta['wait_time_lag_6'],
                        'ride_id': meta['ride_id']
                    }])
                    
                    preds = st.session_state['trainer'].predict_ensemble(inp)
                    st.write(preds)
                else:
                    st.error("Silver Star nicht gefunden.")

    # TAB 4: BENCHMARK
    with tab4:
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            
            st.markdown("### Battle of Algorithms")
            
            # 1. Metrics Table
            metrics_data = []
            for name, metrics in res.items():
                metrics_data.append({
                    "Modell": name,
                    "RMSE (Fehler in Min)": metrics['rmse'],
                    "R² (Erklärungskraft)": metrics['r2'],
                    "MAE (Absolutfehler)": metrics['mae']
                })
            
            st.dataframe(pd.DataFrame(metrics_data).set_index("Modell"), use_container_width=True)
            
            # 2. Visual Comparison (Predictions vs Actuals)
            st.markdown("### Visueller Vergleich (Test-Set Sample)")
            st.caption("Vergleich der Vorhersagen auf den letzten 50 Datenpunkten des Test-Sets.")
            
            # Prepare plotting data
            df_plot = pd.DataFrame()
            first_key = list(res.keys())[0]
            # Take slice of actuals
            limit = 50
            actuals = res[first_key]['actuals'][:limit]
            
            df_plot['Ground Truth'] = actuals
            
            for name, metrics in res.items():
                df_plot[name] = metrics['predictions'][:limit]
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0E1117')
            sns.lineplot(data=df_plot, ax=ax, linewidth=2)
            ax.set_ylabel("Wartezeit (min)", color='white')
            ax.set_xlabel("Test-Samples (Zeitachse)", color='white')
            st.pyplot(fig)
            
            st.markdown("""
            **Analyse:**
            * **Random Forest:** Robust, neigt aber bei extremen Werten zur Glättung.
            * **Gradient Boosting:** Oft präziser, kann aber anfälliger für Ausreißer sein.
            * **LSTM:** Benötigt viel mehr Daten. Bei kleinen Datensätzen oft instabiler als RF/GB ("Overfitting" oder Unteranpassung wenn Epochen zu gering).
            """)
            
        else:
            st.info("Bitte Modell im Tab 2 trainieren.")