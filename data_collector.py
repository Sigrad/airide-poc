import pandas as pd
import os
import requests
import time
from datetime import datetime, timedelta
from weather_service import WeatherService

class DataCollector:
    """
    Handles data acquisition from external APIs and local CSV storage.
    Synchronizes theme park wait times with meteorological data.
    """
    def __init__(self):
        self.weather_service = WeatherService()
        self.storage_path = "real_waiting_times.csv"
        self.base_url = "https://queue-times.com/parks/51/queue_times.json"
        self.headers = {'User-Agent': 'AIRide-Academic-Project/1.0'}

    def fetch_historical_data(self, days_back: int = 60) -> pd.DataFrame:
        """Retrieves and merges local cache with historical weather data."""
        if not os.path.exists(self.storage_path):
            return pd.DataFrame()

        try:
            df_historical = pd.read_csv(self.storage_path)
            if df_historical.empty:
                return pd.DataFrame()
                
            df_historical['is_synthetic'] = 0
            df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
            
            # Weather synchronization logic
            start_date = df_historical['timestamp'].min().date()
            end_date = df_historical['timestamp'].max().date()
            
            df_weather = self.weather_service.fetch_weather_data(
                start_date.isoformat(), 
                end_date.isoformat()
            )
            
            if not df_weather.empty:
                df_historical['merge_key'] = df_historical['timestamp'].dt.round('H')
                df_weather['merge_key'] = pd.to_datetime(df_weather['datetime']).dt.round('H')
                df_merged = pd.merge(df_historical, df_weather, on='merge_key', how='left')
                return df_merged.drop(columns=['merge_key', 'datetime'])
            
            return df_historical
        except Exception as e:
            print(f"Data Retrieval Error: {str(e)}")
            return pd.DataFrame()

    def collect_current_snapshot(self):
        """Polls current API state and appends to local storage."""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                current_time = datetime.now()
                records = []
                
                for land in data.get('lands', []):
                    for ride in land.get('rides', []):
                        records.append({
                            "timestamp": current_time,
                            "ride_id": ride['id'],
                            "ride_name": ride['name'],
                            "is_open": ride['is_open'],
                            "wait_time": ride['wait_time']
                        })
                
                if records:
                    df_new = pd.DataFrame(records)
                    write_header = not os.path.exists(self.storage_path)
                    df_new.to_csv(self.storage_path, mode='a', header=write_header, index=False)
        except requests.RequestException as e:
            print(f"API Connectivity Error: {str(e)}")