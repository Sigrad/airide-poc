import os
import time
import random
import logging
import subprocess
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
from weather_service import WeatherService

# Configure logging for production-level feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollector:
    """
    Orchestrates data harvesting from external APIs and manages data persistence.
    
    Includes automated Git operations for data versioning and a fallback 
    synthetic data generator for development environments.
    """

    def __init__(self):
        self.weather_service = WeatherService()
        self.csv_path = "real_waiting_times.csv"
        self.api_url = "https://queue-times.com/parks/51/queue_times.json"
        self.user_agent = {'User-Agent': 'AIRide-PoC-StudentProject/1.0'}
        self.synthetic_rides = ['Silver Star', 'Blue Fire', 'Wodan', 'Arthur', 'Euro-Mir']
        
        self.last_push_time = time.time()
        self.push_interval = 1800  # 30 minutes

    def fetch_historical_data(self, days_back: int = 60) -> pd.DataFrame:
        """
        Loads and merges CSV data with historical weather information.
        
        Args:
            days_back (int): Number of days to generate if no local data exists.
            
        Returns:
            pd.DataFrame: Merged dataset containing wait times and weather features.
        """
        if os.path.exists(self.csv_path):
            try:
                df_real = pd.read_csv(self.csv_path)
                if df_real.empty:
                    return self._generate_synthetic_data(days_back)
                
                df_real['is_synthetic'] = 0 
                df_real['datetime'] = pd.to_datetime(df_real['timestamp'])
                df_real = df_real.sort_values('datetime')
                
                # Define weather timeframe
                start_date = df_real['datetime'].min().date() - timedelta(days=1)
                end_date = df_real['datetime'].max().date() + timedelta(days=1)
                
                df_weather = self.weather_service.fetch_weather_data(
                    start_date.isoformat(), 
                    end_date.isoformat()
                )
                
                if df_weather.empty:
                    return df_real 
                
                # Merge datasets using asof to match nearest timestamps
                df_weather = df_weather.sort_values('datetime')
                df_merged = pd.merge_asof(
                    df_real.sort_values('datetime'), 
                    df_weather, 
                    on='datetime', 
                    direction='nearest',
                    tolerance=pd.Timedelta('1h')
                )
                return df_merged.dropna(subset=['temp'])

            except Exception as e:
                logging.error(f"Error loading historical data: {e}")
                return self._generate_synthetic_data(days_back)
        
        return self._generate_synthetic_data(days_back)

    def _generate_synthetic_data(self, days_back: int) -> pd.DataFrame:
        """Heuristic-based fallback generator for local development."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        df_weather = self.weather_service.fetch_weather_data(start_date.isoformat(), end_date.isoformat())
        
        if df_weather.empty:
            return pd.DataFrame()

        data = []
        for _, row in df_weather.iterrows():
            # Only simulate data during park opening hours
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
                        'is_synthetic': 1
                    })
        return pd.DataFrame(data)

    def start_loop(self, interval: int = 600):
        """Main execution loop for the data collection service."""
        logging.info(f"Collector service started. Targeting: {self.csv_path}")
        while True:
            self._collect_step()
            if time.time() - self.last_push_time > self.push_interval:
                self._git_push()
                self.last_push_time = time.time()
            time.sleep(interval)

    def _collect_step(self):
        """Performs a single API request and persists results."""
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
                    file_exists = os.path.exists(self.csv_path)
                    df.to_csv(self.csv_path, mode='a', header=not file_exists, index=False)
                    logging.info(f"Successfully collected {len(records)} records.")
        except Exception as e:
            logging.error(f"Collection Step Failed: {e}")

    def _git_push(self):
        """Automated version control for data persistence."""
        try:
            subprocess.run(["git", "add", self.csv_path], check=True, capture_output=True)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            commit_res = subprocess.run(
                ["git", "commit", "-m", f"Auto-update dataset: {timestamp}"], 
                check=False, capture_output=True
            )
            if commit_res.returncode == 0:
                subprocess.run(["git", "push"], check=True, capture_output=True)
                logging.info("Data pushed to remote repository.")
        except Exception as e:
            logging.warning(f"Git synchronization skipped: {e}")

if __name__ == "__main__":
    DataCollector().start_loop()