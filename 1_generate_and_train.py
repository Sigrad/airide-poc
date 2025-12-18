import pandas as pd
import numpy as np
import joblib
import holidays
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- KONFIGURATION ---
CSV_FILE = "real_waiting_times.csv"
MODEL_FILE = "airide_model_scientific.pkl"
LAT, LON = 48.26, 7.72 # Rust

def load_harvested_data():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"❌ Datei '{CSV_FILE}' nicht gefunden! Bitte lass erst den Harvester laufen.")
    
    print(f"📂 Lade '{CSV_FILE}'...")
    df = pd.read_csv(CSV_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter: Nur offene Attraktionen
    df = df[df['is_open'] == True]
    
    if len(df) < 50:
        print("⚠️ WARNUNG: Sehr wenige Daten (<50). Das Modell wird ungenau sein.")
        
    # AGGREGATION: Wir bilden den Mittelwert aller Attraktionen pro 10-Minuten-Slot
    # (Alternativ könntest du hier auf eine spezifische Ride-ID filtern)
    df_agg = df.groupby('timestamp')['wait_time'].mean().reset_index()
    df_agg['wait_time'] = df_agg['wait_time'].astype(int)
    
    print(f"✅ {len(df_agg)} Zeitpunkte zum Trainieren gefunden.")
    return df_agg

def add_features(df):
    print("🛠️ Feature Engineering (Kalender & Lag)...")
    
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    # 1. Kalender (Dreiländereck)
    de_bw = holidays.DE(subdiv='BW')
    fr = holidays.FR()
    ch = holidays.CH(subdiv='BS')
    
    df['is_weekend'] = df['timestamp'].dt.dayofweek.apply(lambda x: 1 if x>=5 else 0)
    df['is_holiday_BW'] = df['date'].apply(lambda x: 1 if x in de_bw else 0)
    df['is_school_holiday_FR'] = df['date'].apply(lambda x: 1 if x in fr else 0)
    df['is_holiday_CH'] = df['date'].apply(lambda x: 1 if x in ch else 0)
    
    # 2. Lag-Feature (Wartezeit vor 1 Stunde)
    # Da wir echte Daten haben, müssen wir das berechnen (Shift)
    # Wir sortieren erst sicherheitshalber
    df = df.sort_values('timestamp')
    # Wir nehmen den Wert von vor 6 Zeilen (bei 10 Min Takt = 60 Min)
    df['wait_time_lag_1h'] = df['wait_time'].shift(6).fillna(method='bfill')
    
    return df

def fetch_historical_weather(df):
    print("☁️ Hole echte Wetterdaten für die Zeitpunkte im CSV...")
    
    # Zeitspanne ermitteln
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    
    # API Setup
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["temperature_2m", "rain", "cloud_cover", "wind_speed_10m"]
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Wetter-DataFrame bauen
        hourly = response.Hourly()
        w_dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        
        weather_df = pd.DataFrame({
            "timestamp_h": w_dates,
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "rain_mm": hourly.Variables(1).ValuesAsNumpy(),
            "clouds": hourly.Variables(2).ValuesAsNumpy(),
            "wind_kmh": hourly.Variables(3).ValuesAsNumpy()
        })
        
        # Merge: Wir runden die Wartezeit-Timestamps auf die volle Stunde für das Wetter
        # (Einfacher als Interpolation für den PoC)
        df['timestamp_h'] = df['timestamp'].dt.floor('H').dt.tz_localize('UTC')
        
        # Merge
        df_merged = pd.merge(df, weather_df, left_on='timestamp_h', right_on='timestamp_h', how='left')
        
        # Falls Wetterdaten fehlen (z.B. heute noch nicht im Archiv), füllen wir auf (Forward Fill)
        df_merged = df_merged.fillna(method='ffill').fillna(method='bfill')
        
        return df_merged
        
    except Exception as e:
        print(f"⚠️ Wetter-Fehler: {e}. Nutze Durchschnittswerte als Fallback.")
        df['temp'] = 20
        df['rain_mm'] = 0
        df['clouds'] = 50
        df['wind_kmh'] = 10
        return df

def calculate_hci(df):
    print("⚗️ Berechne HCI...")
    # HCI = 4*TC + 2*A + 3*P + 1*W (Skaliert auf 0-100)
    tc = (df['temp'].clip(0, 35) / 35) * 10
    a = ((100 - df['clouds']) / 100) * 10
    p = np.where(df['rain_mm'] > 0.5, 0, 10)
    w = ((60 - df['wind_kmh']) / 60).clip(0,1) * 10
    
    df['HCI_Urban'] = (4*tc + 2*a + 3*p + 1*w) / 10 * 10
    return df

def main():
    # 1. Pipeline
    df = load_harvested_data()
    df = add_features(df)
    df = fetch_historical_weather(df)
    df = calculate_hci(df)
    
    # 2. Training
    features = ['hour', 'is_weekend', 'is_holiday_BW', 'is_school_holiday_FR', 'is_holiday_CH',
                'temp', 'clouds', 'rain_mm', 'wind_kmh', 'HCI_Urban', 'wait_time_lag_1h']
    
    print(f"🧠 Starte Training mit {len(df)} echten Datensätzen...")
    
    X = df[features]
    y = df['wait_time']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y) # Wir nutzen alle Daten zum Training (da wir wenig haben)
    
    # Speichern
    joblib.dump(model, MODEL_FILE)
    print(f"🎉 Erfolg! Modell '{MODEL_FILE}' wurde nur mit echten Daten trainiert.")

if __name__ == "__main__":
    main()