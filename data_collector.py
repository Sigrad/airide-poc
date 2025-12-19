"""
Data Collection and Persistence Module for AIRide PoC.

This module orchestrates the automated harvesting of real-time queue data from
the Europa-Park API, enriches it with meteorological information, and manages
data persistence through CSV storage with integrated Git versioning.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

import pandas as pd
import os
import requests
import time
import subprocess
from datetime import datetime, timedelta, date
from weather_service import WeatherService
import random

class DataCollector:
    """
    Orchestrates data harvesting from API and persistence to CSV.
    Includes automated Git operations for data versioning.
    """
    def __init__(self):
        self.weather_service = WeatherService()
        self.csv_path = "real_waiting_times.csv"
        self.api_url = "https://queue-times.com/parks/51/queue_times.json"
        self.user_agent = {'User-Agent': 'AIRide-PoC-StudentProject/1.0'}
        self.synthetic_rides = ['Silver Star', 'Blue Fire', 'Wodan', 'Arthur', 'Euro-Mir']
        
        self.last_push_time = time.time()
        self.push_interval = 1800  # 30 minutes

    def fetch_historical_data(self, days_back=60):
        """Loads and merges CSV data with historical weather."""
        if os.path.exists(self.csv_path):
            try:
                df_real = pd.read_csv(self.csv_path)
                if df_real.empty: return self._generate_synthetic_data(days_back)
                
                # Mark as Real Data (Ground Truth)
                df_real['is_synthetic'] = 0 
                
                df_real['datetime'] = pd.to_datetime(df_real['timestamp'])
                df_real = df_real.sort_values('datetime')
                
                # Fetch Weather Window
                start_date = df_real['datetime'].min().date() - timedelta(days=1)
                end_date = df_real['datetime'].max().date() + timedelta(days=1)
                
                df_weather = self.weather_service.fetch_weather_data(
                    start_date.isoformat(), 
                    end_date.isoformat()
                )
                
                if df_weather.empty: return df_real 
                
                # Merge
                df_real = df_real.sort_values('datetime')
                df_weather = df_weather.sort_values('datetime')
                
                df_merged = pd.merge_asof(
                    df_real, 
                    df_weather, 
                    on='datetime', 
                    direction='nearest',
                    tolerance=pd.Timedelta('1h')
                )
                return df_merged.dropna(subset=['temp'])

            except Exception:
                return self._generate_synthetic_data(days_back)
        
        return self._generate_synthetic_data(days_back)

    def _generate_synthetic_data(self, days_back):
        """Fallback generator based on heuristics."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        df_weather = self.weather_service.fetch_weather_data(start_date.isoformat(), end_date.isoformat())
        if df_weather.empty: return pd.DataFrame()

        data = []
        for _, row in df_weather.iterrows():
            if 9 <= row['datetime'].hour <= 18:
                is_weekend = row['datetime'].weekday() >= 5
                for ride in self.synthetic_rides:
                    wait = 15 + (20 if is_weekend else 0) + random.randint(-5, 15)
                    data.append({
                        'datetime': row['datetime'], 
                        'ride_name': ride,
                        'wait_time': max(0, wait), 
                        'temp': row['temp'],
                        'rain': row['rain'], 
                        'wind': row['wind'], 
                        'cloud_cover': row['cloud_cover'], 
                        'is_open': True,
                        'is_synthetic': 1  # Flag as Synthetic
                    })
        return pd.DataFrame(data)

    def start_loop(self, interval=600):
        print(f"Collector started. Storage: {self.csv_path}")
        while True:
            self._collect_step()
            if time.time() - self.last_push_time > self.push_interval:
                self._git_push()
                self.last_push_time = time.time()
            time.sleep(interval)

    def _collect_step(self):
        try:
            resp = requests.get(self.api_url, headers=self.user_agent, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
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
                    df = pd.DataFrame(records)
                    hdr = not os.path.exists(self.csv_path)
                    df.to_csv(self.csv_path, mode='a', header=hdr, index=False)
                    print(f"Collected {len(records)} records at {now.strftime('%H:%M')}")
        except Exception as e:
            print(f"Collection Error: {e}")

    def _git_push(self):
        try:
            subprocess.run(["git", "add", "real_waiting_times.csv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            commit_res = subprocess.run(
                ["git", "commit", "-m", f"Auto-update: {timestamp}"], 
                check=False, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            if commit_res.returncode == 0:
                subprocess.run(["git", "push"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

if __name__ == "__main__":
    DataCollector().start_loop()