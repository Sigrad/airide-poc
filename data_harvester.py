import pandas as pd
import numpy as np
import os
import requests
import time
import subprocess  # NEU: Für Git Befehle
from datetime import datetime, timedelta, date
from data_loader import DataLoader
import random

class DataHarvester:
    def __init__(self):
        self.loader = DataLoader()
        self.csv_path = "real_waiting_times.csv"
        self.api_url = "https://queue-times.com/parks/51/queue_times.json"
        self.user_agent = {'User-Agent': 'AIRide-PoC-StudentProject/1.0'}
        self.synthetic_rides = ['Silver Star', 'Blue Fire', 'Wodan', 'Arthur', 'Euro-Mir']
        
        # NEU: Konfiguration für Auto-Push
        self.last_push_time = time.time()
        self.push_interval = 1800  # 1800 Sekunden = 30 Minuten

    # --- MODE A: DATA PROVIDER (Used by App) ---
    def fetch_historical_data(self, days_back=60):
        if os.path.exists(self.csv_path):
            print(f"Reading real data from: {self.csv_path}")
            try:
                df_real = pd.read_csv(self.csv_path)
                
                if df_real.empty:
                    print("CSV is empty. Switching to synthetic.")
                else:
                    df_real['datetime'] = pd.to_datetime(df_real['timestamp'])
                    df_real = df_real.sort_values('datetime')
                    
                    start_date = df_real['datetime'].min().date()
                    end_date = df_real['datetime'].max().date()
                    w_start = (start_date - timedelta(days=1)).isoformat()
                    w_end = (end_date + timedelta(days=1)).isoformat()
                    
                    print(f"Fetching weather: {w_start} to {w_end}")
                    df_weather = self.loader.fetch_weather_history(w_start, w_end)
                    
                    if df_weather.empty:
                        print("No weather data. Filling defaults.")
                        df_real['temp'] = 20
                        df_real['rain'] = 0
                        df_real['wind'] = 5
                        df_real['cloud_cover'] = 50
                        return df_real
                    
                    df_real = df_real.sort_values('datetime')
                    df_weather = df_weather.sort_values('datetime')
                    
                    df_merged = pd.merge_asof(
                        df_real, 
                        df_weather, 
                        on='datetime', 
                        direction='nearest',
                        tolerance=pd.Timedelta('1h')
                    )
                    
                    df_merged['temp'] = df_merged['temp'].fillna(20)
                    df_merged['rain'] = df_merged['rain'].fillna(0)
                    df_merged['wind'] = df_merged['wind'].fillna(5)
                    df_merged['cloud_cover'] = df_merged['cloud_cover'].fillna(50)
                    
                    print(f"Merged successfully: {len(df_merged)} rows.")
                    return df_merged

            except Exception as e:
                print(f"Error reading CSV: {e}. Switching to synthetic.")

        print("No CSV found. Generating synthetic data...")
        return self._generate_synthetic_data(days_back)

    def _generate_synthetic_data(self, days_back):
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        df_weather = self.loader.fetch_weather_history(start_date.isoformat(), end_date.isoformat())
        if df_weather.empty: return pd.DataFrame()

        data = []
        for index, row in df_weather.iterrows():
            current_dt = row['datetime']
            if 9 <= current_dt.hour <= 18:
                is_weekend = current_dt.weekday() >= 5
                for ride in self.synthetic_rides:
                    wait = 15
                    if is_weekend: wait += 20
                    if 11 <= current_dt.hour <= 15: wait += 15
                    if row['rain'] > 1.5:
                        if ride in ['Silver Star', 'Blue Fire', 'Wodan']: wait -= 25
                        else: wait += 20
                    if row['temp'] > 30 and ride != 'Arthur': wait -= 10
                    wait += random.randint(-5, 15)
                    data.append({
                        'datetime': current_dt, 'ride_name': ride,
                        'wait_time': max(0, int(wait)), 'temp': row['temp'],
                        'rain': row['rain'], 'wind': row['wind'], 'cloud_cover': row['cloud_cover'],
                        'is_open': True
                    })
        return pd.DataFrame(data)

    # --- MODE B: ACTIVE HARVESTER (Run via Main) ---
    def start_harvesting(self, interval=600):
        print("AIRide Harvester started...")
        print(f"Storage: {os.path.abspath(self.csv_path)}")
        print("Auto-Push: Enabled (every 30 mins)")
        print("Press CTRL+C to stop.\n")
        
        while True:
            # 1. Daten holen
            self._fetch_and_save()
            
            # 2. Prüfen ob Zeit für GitHub Push
            if time.time() - self.last_push_time > self.push_interval:
                self._auto_push_to_github()
                self.last_push_time = time.time()
            
            print(f"Sleeping for {interval} seconds...")
            time.sleep(interval)

    def _fetch_and_save(self):
        try:
            response = requests.get(self.api_url, headers=self.user_agent, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            records = []
            now = datetime.now()
            
            for land in data.get('lands', []):
                for ride in land.get('rides', []):
                    records.append({
                        "timestamp": now,
                        "ride_id": ride['id'],
                        "ride_name": ride['name'],
                        "is_open": ride['is_open'],
                        "wait_time": ride['wait_time'],
                        "last_updated": ride['last_updated']
                    })
            
            if records:
                df_new = pd.DataFrame(records)
                header = not os.path.exists(self.csv_path)
                df_new.to_csv(self.csv_path, mode='a', header=header, index=False)
                print(f"{now.strftime('%H:%M:%S')}: Saved {len(records)} records.")
            
        except Exception as e:
            print(f"Harvest Error: {e}")

    def _auto_push_to_github(self):
        print("--- Starting Auto-Push to GitHub ---")
        try:
            # 1. Git Add
            subprocess.run(["git", "add", "real_waiting_times.csv"], check=True)
            
            # 2. Git Commit
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            commit_msg = f"Auto-update data: {timestamp}"
            # Wir fangen den Fehler ab, falls es nichts zu committen gibt (Exit Code 1)
            subprocess.run(["git", "commit", "-m", commit_msg], check=False)
            
            # 3. Git Push
            subprocess.run(["git", "push"], check=True)
            print("--- GitHub Push Successful! ---")
            
        except subprocess.CalledProcessError as e:
            print(f"Git Error (Push failed): {e}")
        except FileNotFoundError:
            print("Git not found. Please install Git and add it to PATH.")

if __name__ == "__main__":
    h = DataHarvester()
    h.start_harvesting()