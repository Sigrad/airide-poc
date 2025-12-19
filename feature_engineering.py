import pandas as pd
import numpy as np
import holidays
from datetime import date

class FeatureEngineering:
    """
    Transforms raw data into a feature matrix for Machine Learning.
    Implements HCI (Holiday Climate Index) and regional holiday logic.
    """
    def __init__(self):
        self.de_holidays = holidays.Germany(subdiv='BW')
        self.ch_holidays = holidays.Switzerland(subdiv='BS') 
        self.fr_holidays = [(date(2024, 12, 21), date(2025, 1, 6))] # Simplified example

    def calculate_hci(self, row):
        """Calculates a simplified urban HCI score."""
        t = row.get('temp', 25)
        # Ideal temp 25°C, penalty for deviation
        thermal = 4 * max(0, 10 - abs(t - 25) * 0.5)
        return thermal + 20 # Base attraction score

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        
        # Consistent time column naming
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        
        df = df.sort_values(['ride_name', 'datetime'])
        
        # Temporal Features
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Regional Holidays
        df['holiday_de_bw'] = df['datetime'].dt.date.apply(lambda x: 1 if x in self.de_holidays else 0)
        df['holiday_ch_bs'] = df['datetime'].dt.date.apply(lambda x: 1 if x in self.ch_holidays else 0)
        df['holiday_fr_zone_b'] = 0 # Expand as needed
        
        # Ride ID mapping
        df['ride_id'] = pd.factorize(df['ride_name'])[0]
        
        # Weather & Lags
        df['HCI_Urban'] = df.apply(self.calculate_hci, axis=1)
        df['wait_time_lag_1'] = df.groupby('ride_name')['wait_time'].shift(1).fillna(0)
        df['wait_time_lag_6'] = df.groupby('ride_name')['wait_time'].shift(6).fillna(0)
        
        return df.dropna(subset=['wait_time'])