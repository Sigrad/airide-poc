import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

# --- KONFIGURATION ---
st.set_page_config(
    page_title="AIRide Analyse",
    layout="wide",
    initial_sidebar_state="expanded"
)
sns.set_theme(style="whitegrid", palette="muted") # Akademischer, ruhiger Plot-Stil

# --- CSS HACK FÜR PROFESSIONELLEN LOOK ---
# Entfernt unnötiges Padding und macht Tabellen lesbarer
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
    """, unsafe_allow_html=True)

# --- HEADER BEREICH ---
st.title("AIRide: Prädiktive Analyse von Besucherströmen")
st.markdown("""
**Projektarbeit HS 25 | Sakir Dönmez**

Dieses Dashboard dient als **Proof of Concept (PoC)** für den Einsatz von Machine Learning in der Freizeitpark-Logistik.
Es kombiniert Echtzeit-Datenströme mit meteorologischen Indizes, um Wartezeiten nicht nur zu überwachen, sondern mittels **Random Forest Regression** vorherzusagen.
""")

# --- SIDEBAR STEUERUNG ---
st.sidebar.header("Systemsteuerung")

if st.sidebar.button("🔄 Datenpipeline aktualisieren"):
    st.cache_data.clear()
    st.rerun()

# Status Anzeige mit Erklärung
st.sidebar.markdown("### Datenquelle")
if os.path.exists("real_waiting_times.csv"):
    st.sidebar.success("ONLINE: Live-Daten (API Ingestion)")
    st.sidebar.caption("Daten werden via `DataCollector` Modul alle 10min von der Queue-Times API und OpenMeteo geerntet.")
else:
    st.sidebar.warning("SIMULATION: Synthetische Daten")
    st.sidebar.caption("Fallback-Modus aktiv, da keine CSV-Datenbank gefunden wurde.")

st.sidebar.divider()
st.sidebar.subheader("Modellierung")
train_btn = st.sidebar.button("Modell neu trainieren", type="primary")
st.sidebar.caption("Startet das Training des Random Forest Regressors auf den aktuellsten Daten.")

# --- DATEN LADEN (Pipeline) ---
@st.cache_data(ttl=60)
def load_data_pipeline(days_back=60):
    collector = DataCollector()
    # 1. Rohdaten holen (Enthält auch geschlossene Bahnen)
    df_raw = collector.fetch_historical_data(days_back=days_back)
    
    if df_raw.empty: return pd.DataFrame(), pd.DataFrame()
        
    # 2. Daten verarbeiten (Feature Engineering)
    engine = FeatureEngineering()
    df_processed = engine.process_data(df_raw)
    
    return df_raw, df_processed

with st.spinner("Initialisiere Data Warehouse & Feature Engineering Pipeline..."):
    df_raw, df_ai = load_data_pipeline()

# --- HAUPTINHALT ---
if df_raw.empty:
    st.error("Kritischer Fehler: Keine Daten im Data Lake gefunden. Bitte starten Sie den DataCollector.")
else:
    # --- KPI BERECHNUNG ---
    latest_ts = df_raw['datetime'].max()
    snapshot = df_raw[df_raw['datetime'] == latest_ts]
    
    # Filterung: Wir betrachten für KPIs nur relevante (offene) Attraktionen
    if 'is_open' in snapshot.columns:
        open_rides = snapshot[snapshot['is_open'] == True]
    else:
        open_rides = snapshot[snapshot['wait_time'] > 0]
        
    count_open = len(open_rides)
    
    if count_open > 0:
        avg_wait = open_rides['wait_time'].mean()
        temp_now = open_rides['temp'].mean()
    else:
        avg_wait = 0
        temp_now = snapshot['temp'].mean() if not snapshot.empty else 0
    
    # HCI Berechnung für KPI (vereinfacht)
    hci_score = (4 * max(0, 10-abs(temp_now-25)*0.5)) + 20 
    
    # Metriken Zeile
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letzte Ingestion", latest_ts.strftime('%H:%M:%S'))
    
    if count_open == 0:
        c2.error("Park Geschlossen")
    else:
        c2.metric("Aktive Attraktionen", count_open)
        
    c3.metric("Ø Wartezeit (Aggregiert)", f"{avg_wait:.1f} min", delta_color="inverse")
    c4.metric("Holiday Climate Index", f"{hci_score:.0f}/100", help="HCI:Urban nach Scott et al. (0=Ungünstig, 100=Ideal)")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Echtzeit-Analyse", "🧠 Modell-Validierung", "🔮 Inferenz-Simulator"])

    # TAB 1: LIVE STATUS
    with tab1:
        st.subheader("Aktueller operativer Status")
        st.markdown("Übersicht der momentanen Wartezeiten basierend auf dem letzten API-Abruf.")
        
        if count_open > 0:
            c_chart, c_table = st.columns([1.5, 1])
            
            with c_chart:
                # Chart Visualisierung
                fig, ax = plt.subplots(figsize=(8, 6))
                open_sorted = open_rides.sort_values('wait_time', ascending=False)
                sns.barplot(data=open_sorted, x='wait_time', y='ride_name', palette="viridis", ax=ax)
                ax.set_xlabel("Wartezeit in Minuten")
                ax.set_ylabel("")
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
            with c_table:
                # Schöne Tabelle mit Progress Bar
                st.markdown("#### Detaildaten")
                # Spaltenauswahl und Umbenennung
                display_df = open_sorted[['ride_name', 'wait_time']].copy()
                
                st.dataframe(
                    display_df,
                    column_config={
                        "ride_name": "Attraktion",
                        "wait_time": st.column_config.ProgressColumn(
                            "Wartezeit (Min)",
                            help="Aktuelle Wartezeit visualisiert",
                            format="%d min",
                            min_value=0,
                            max_value=120, # Annahme max Wartezeit
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
            with st.expander("Methodische Anmerkung zur Datenbasis"):
                st.markdown("""
                Die Daten stammen von der Queue-Times API und repräsentieren die offiziell kommunizierten Wartezeiten. 
                Geschlossene Attraktionen (Status 'False' oder 0 Min) werden in dieser Ansicht gefiltert, 
                um den Durchschnittswert nicht künstlich zu senken (Bias Reduction).
                """)
                
        else:
            st.info("Der Park ist derzeit geschlossen. Es liegen keine Echtzeit-Wartezeiten vor.")
            st.markdown("**Historischer Log (Letzte 10 Einträge):**")
            st.dataframe(df_raw.tail(10))

    # TAB 2: KI MODEL VALIDIERUNG
    with tab2:
        st.subheader("Modell-Training & Evaluation")
        st.markdown("""
        Hier wird das **Random Forest Regressionsmodell** trainiert und validiert. 
        Zur Bewertung der Prognosegüte wird ein 80/20 Time-Series Split verwendet, um Data Leakage zu vermeiden.
        """)

        if train_btn:
            if df_ai.empty or len(df_ai) < 10:
                st.error("Nicht genügend Trainingsdaten verfügbar (n < 10).")
            else:
                trainer = PredictionModel()
                with st.spinner("Trainiere Random Forest (n_estimators=100, depth=20)..."):
                    results = trainer.train_and_evaluate(df_ai)
                    st.session_state['model'] = results['model']
                    st.session_state['metrics'] = results
                st.success(f"Modell erfolgreich trainiert auf {len(df_ai)} Datensätzen.")
        
        if 'metrics' in st.session_state:
            m = st.session_state['metrics']
            
            # KPI Row für Modell
            k1, k2, k3 = st.columns(3)
            k1.metric("RMSE (Genauigkeit)", f"{m['rmse']:.2f} min", 
                     help="Root Mean Squared Error. Durchschnittlicher Fehler der Vorhersage in Minuten.")
            k2.metric("R² Score (Erklärkraft)", f"{m['r2']:.2f}", 
                     help="Anteil der Varianz, der durch das Modell erklärt wird (1.0 = Perfekt).")
            k3.metric("Diebold-Mariano p-value", f"{m['p_value']:.4f}", 
                     help="Statistischer Test gegen Baseline. p < 0.05 zeigt signifikante Überlegenheit.")
            
            col_feat, col_exp = st.columns([2, 1])
            
            with col_feat:
                st.markdown("#### Feature Importance Analyse")
                fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                
                # Deutsche Labels für den Plot
                imp_df = m['feature_importance'].head(10).copy()
                imp_df['Feature'] = imp_df['Feature'].map({
                    'hour': 'Tageszeit (Stunde)', 
                    'ride_id': 'Attraktion (ID)',
                    'temp': 'Temperatur',
                    'wait_time_lag_1': 'Auto-Regressiv (Lag 1)',
                    'wait_time_lag_6': 'Auto-Regressiv (Lag 6)',
                    'HCI_Urban': 'Wetter Index (HCI)',
                    'weekday': 'Wochentag',
                    'is_weekend': 'Wochenende',
                    'rain': 'Niederschlag',
                    'month': 'Monat'
                }).fillna(imp_df['Feature'])
                
                sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette="magma", ax=ax_imp)
                ax_imp.set_xlabel("Relativer Einfluss auf Prognose")
                ax_imp.set_ylabel("")
                st.pyplot(fig_imp)
                
            with col_exp:
                st.markdown("#### Interpretation")
                st.info("""
                **Feature Importance:** Zeigt, welche Variablen die Entscheidung der Bäume im Random Forest dominieren.
                Hohe Werte bei **Lag-Features** deuten auf eine hohe Systemträgheit hin (die Wartezeit vor 10 Min ist ein guter Prädiktor für jetzt).
                Der **HCI** fasst komplexe Wettereinflüsse zusammen.
                """)

        else:
            st.warning("Kein trainiertes Modell im Speicher. Bitte führen Sie das Training durch.")

    # TAB 3: SIMULATOR
    with tab3:
        st.subheader("Prädiktive Simulation (Nowcasting)")
        st.markdown("""
        Dieses Modul ermöglicht **Was-wäre-wenn-Szenarien**. 
        Durch Anpassung der meteorologischen Parameter wird der **Holiday Climate Index (HCI)** neu berechnet 
        und in das ML-Modell gespeist, um die erwartete Auslastung zu prognostizieren.
        """)
        
        # Formel Erklärung
        with st.expander("Methodik: Holiday Climate Index (HCI:Urban)"):
            st.latex(r'''
            HCI = 4 \cdot TC + 2 \cdot A + 3 \cdot P + 1 \cdot W
            ''')
            st.caption("TC: Thermischer Komfort, A: Ästhetik (Bewölkung), P: Niederschlag, W: Wind")
        
        if 'model' in st.session_state:
            # Simulator Controls in einer Box
            with st.container(border=True):
                st.markdown("**Simulations-Parameter**")
                sc1, sc2, sc3 = st.columns(3)
                s_temp = sc1.slider("Temperatur (°C)", 0, 40, 25)
                s_rain = sc2.slider("Niederschlag (mm)", 0.0, 15.0, 0.0)
                s_cloud = sc3.slider("Bewölkung (%)", 0, 100, 20)
                
                # Berechne HCI für Sim
                sim_hci = (4 * max(0, 10-abs(s_temp-25)*0.5)) + (2 * (100-s_cloud)/10) + (3 * max(0, 10-s_rain*2)) + 10
                st.metric("Resultierender HCI Score", f"{sim_hci:.1f}/100")
            
            # Prognose Loop
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
                    preds.append({'Attraktion': r, 'Prognose': int(val)})
                except: continue
            
            # Ergebnis Tabelle
            if preds:
                st.subheader("Simulierte Wartezeiten")
                res_df = pd.DataFrame(preds).sort_values('Prognose', ascending=False)
                
                st.dataframe(
                    res_df,
                    column_config={
                        "Attraktion": st.column_config.TextColumn("Attraktion"),
                        "Prognose": st.column_config.NumberColumn(
                            "Prognose (Min)",
                            format="%d min",
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
        else:
            st.warning("Bitte trainieren Sie zuerst das Modell im Tab 'Modell-Validierung' oder in der Sidebar.")