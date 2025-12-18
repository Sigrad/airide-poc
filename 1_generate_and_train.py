import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import joblib
import holidays
from retry_requests import retry
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- KONFIGURATION ---
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"  # 2 Jahre echte Datenbasis
LAT = 48.26  # Koordinaten Rust
LON = 7.72

# Setup Open-Meteo API Client mit Cache
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_real_weather():
    print(f"☁️ Lade echte Wetterdaten für Rust ({START_DATE}-{END_DATE})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": ["temperature_2m", "rain", "cloud_cover", "wind_speed_10m"]
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Datenverarbeitung
    hourly = response.Hourly()
    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    
    df = pd.DataFrame({
        "date": dates,
        "temp": hourly.Variables(0).ValuesAsNumpy(),
        "rain_mm": hourly.Variables(1).ValuesAsNumpy(),
        "clouds": hourly.Variables(2).ValuesAsNumpy(),
        "wind_kmh": hourly.Variables(3).ValuesAsNumpy()
    })
    
    # Filter: Nur Parköffnungszeiten (08:00 - 20:00)
    df['hour'] = df['date'].dt.hour
    df = df[(df['hour'] >= 8) & (df['hour'] <= 20)]
    return df

def add_real_holidays(df):
    print("📅 Berechne echte Feiertage (Dreiländereck)...")
    dates = df['date'].dt.date
    
    # Kalender laden
    de_bw = holidays.DE(subdiv='BW') # Baden-Württemberg
    fr = holidays.FR()               # Frankreich
    ch = holidays.CH(subdiv='BS')    # Schweiz (Basel)

    df['is_holiday_BW'] = dates.apply(lambda x: 1 if x in de_bw else 0)
    df['is_school_holiday_FR'] = dates.apply(lambda x: 1 if x in fr else 0) 
    df['is_holiday_CH'] = dates.apply(lambda x: 1 if x in ch else 0)
    df['is_weekend'] = df['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def calculate_hci(df):
    print("⚗️ Feature Engineering: HCI:Urban...")
    # Formel aus Paper: HCI = 4*TC + 2*A + 3*P + 1*W
    
    # Normalisierung auf 0-10 Punkte
    tc_score = (df['temp'].clip(0, 35) / 35) * 10
    a_score = ((100 - df['clouds']) / 100) * 10
    p_score = np.where(df['rain_mm'] > 0.5, 0, 10) # Strenger Abzug bei Regen
    w_score = ((60 - df['wind_kmh']) / 60).clip(0, 1) * 10
    
    # Gewichtete Summe
    df['HCI_Urban'] = (4 * tc_score + 2 * a_score + 3 * p_score + 1 * w_score) / 10 * 10
    return df

def simulate_wait_times(df):
    print("🎢 Simuliere Wartezeiten basierend auf echten Faktoren...")
    np.random.seed(42)
    
    # Faktoren-Gewichtung (Logik aus dem Paper)
    peak = df['hour'].apply(lambda h: 25 if 12 <= h <= 15 else 0)
    
    # Dreiländereck-Effekt
    calendar = (df['is_holiday_BW'] * 20 + df['is_holiday_CH'] * 15 + df['is_weekend'] * 20)
    
    # Wetter & Kipppunkte (Hitze/Sturm)
    weather = (df['HCI_Urban'] / 100) * 30
    tipping_point = np.where((df['temp'] > 32) | (df['wind_kmh'] > 45), -30, 0)
    
    # Lag (Trägheit)
    df['wait_time_lag_1h'] = np.random.normal(30, 15, len(df)).clip(0, 120)
    
    # Gesamtsumme
    df['wait_time'] = 15 + peak + calendar + weather + tipping_point + (df['wait_time_lag_1h']*0.5) + np.random.normal(0, 5, len(df))
    df['wait_time'] = df['wait_time'].clip(0, 180).astype(int)
    
    return df

def main():
    # 1. Pipeline ausführen
    df = fetch_real_weather()
    df = add_real_holidays(df)
    df = calculate_hci(df)
    df = simulate_wait_times(df)
    
    # 2. Training
    features = ['hour', 'is_weekend', 'is_holiday_BW', 'is_school_holiday_FR', 'is_holiday_CH',
                'temp', 'clouds', 'rain_mm', 'wind_kmh', 'HCI_Urban', 'wait_time_lag_1h']
    
    print(f"🧠 Trainiere Modell mit {len(df)} Datensätzen...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(df[features], df['wait_time'], test_size=0.2)
    model.fit(X_train, y_train)
    
    # 3. Evaluation & Speichern
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"📊 RMSE: {rmse:.2f} Min | R²: {r2_score(y_test, model.predict(X_test)):.2f}")
    
    joblib.dump(model, 'airide_model_scientific.pkl')
    print("✅ Modell 'airide_model_scientific.pkl' gespeichert.")

if __name__ == "__main__":
    main()