import pandas as pd
import numpy as np
import holidays
from datetime import date

class FeatureEngineer:
    def __init__(self):
        self.de_holidays = holidays.Germany(subdiv='BW')
        self.ch_holidays = holidays.Switzerland(subdiv='BS') 
        self.fr_zone_b_holidays = [
            (date(2024, 2, 24), date(2024, 3, 11)), (date(2024, 4, 20), date(2024, 5, 6)),
            (date(2024, 7, 6), date(2024, 9, 2)), (date(2024, 10, 19), date(2024, 11, 4)),
            (date(2024, 12, 21), date(2025, 1, 6))
        ]

    def _is_fr(self, d):
        dt = d.date()
        for s, e in self.fr_zone_b_holidays:
            if s <= dt <= e: return 1
        return 0

    def calculate_hci(self, r):
        tc = max(0, 10 - abs(r['temp'] - 25) * 0.5)
        a = (100 - r.get('cloud_cover', 50)) / 10
        p = max(0, 10 - r['rain'] * 2)
        w = max(0, 10 - r['wind'] * 0.5)
        return (4 * tc) + (2 * a) + (3 * p) + (1 * w)

    def enrich_data(self, df_in):
        df = df_in.copy()
        
        # --- CRITICAL FILTER ---
        # Only process rides that are OPEN. 
        # Removing closed rides prevents training on 0-wait times.
        if 'is_open' in df.columns:
            df = df[df['is_open'] == True]
            if df.empty: return pd.DataFrame()
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
             df['datetime'] = pd.to_datetime(df['timestamp'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        df = df.sort_values(['ride_name', 'datetime'])
        
        # Time & Calendar
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
        
        df['holiday_de_bw'] = df['datetime'].apply(lambda x: 1 if x in self.de_holidays else 0)
        df['holiday_ch_bs'] = df['datetime'].apply(lambda x: 1 if x in self.ch_holidays else 0)
        df['holiday_fr_zone_b'] = df['datetime'].apply(self._is_fr)
        
        # HCI
        df['HCI_Urban'] = df.apply(self.calculate_hci, axis=1)
        
        # Lag Features
        df['wait_time_lag_1'] = df.groupby('ride_name')['wait_time'].shift(1)
        df['wait_time_lag_6'] = df.groupby('ride_name')['wait_time'].shift(6)
        
        if 'ride_id' not in df.columns:
            df['ride_id'] = df['ride_name'].astype('category').cat.codes
        else:
            df['ride_id'] = pd.to_numeric(df['ride_id'], errors='coerce').fillna(0)
            
        return df.dropna().reset_index(drop=True)