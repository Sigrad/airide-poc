import pandas as pd
import os
import requests
from datetime import datetime
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

    def fetch_historical_data(self) -> pd.DataFrame:
        """Retrieves and merges local cache with historical weather data."""
        if not os.path.exists(self.storage_path):
            return pd.DataFrame()

        try:
            df_historical = pd.read_csv(self.storage_path)
            if df_historical.empty:
                return pd.DataFrame()
                
            # Standardize time column
            df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
            
            # Weather synchronization based on data range
            start_date = df_historical['timestamp'].min().strftime('%Y-%m-%d')
            end_date = df_historical['timestamp'].max().strftime('%Y-%m-%d')
            
            df_weather = self.weather_service.fetch_weather_data(start_date, end_date)
            
            if not df_weather.empty:
                df_historical['merge_key'] = df_historical['timestamp'].dt.round('H')
                df_weather['merge_key'] = pd.to_datetime(df_weather['datetime']).dt.round('H')
                
                df_merged = pd.merge(df_historical, df_weather, on='merge_key', how='left')
                return df_merged.drop(columns=['merge_key', 'datetime'])
            
            return df_historical
        except Exception as e:
            print(f"Data Retrieval Error: {str(e)}")
            return pd.DataFrame()