"""
Feature Engineering and Domain Logic Module for AIRide PoC.

This module transforms raw queue and weather data into a structured feature 
matrix for machine learning. It implements domain-specific calculations 
such as the Holiday Climate Index (HCI) and regional holiday logic.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

import pandas as pd
import numpy as np
import holidays
from datetime import date
from typing import List, Tuple

class FeatureEngineering:
    """
    Transforms raw input data into a feature-rich matrix for predictive modeling.
    
    This class implements a specialized preprocessing pipeline that:
    1. Filters and cleans raw observation data
    2. Maps regional school holidays for DE, CH, and FR
    3. Calculates the Holiday Climate Index (HCI:Urban)
    4. Generates temporal lag features to capture system inertia
    """

    # Regional Holiday Configuration
    GERMAN_SUBDIVISION = 'BW'  # Baden-Württemberg
    SWISS_SUBDIVISION = 'BS'   # Basel-Stadt
    
    # French Zone B (Alsace/Grand Est) School Holiday Calendar 2024/25
    FR_ZONE_B_HOLIDAYS = [
        (date(2024, 2, 24), date(2024, 3, 11)),
        (date(2024, 4, 20), date(2024, 5, 6)),
        (date(2024, 7, 6), date(2024, 9, 2)),
        (date(2024, 10, 19), date(2024, 11, 4)),
        (date(2024, 12, 21), date(2025, 1, 6))
    ]

    def __init__(self):
        """Initialize regional holiday lookup engines."""
        self.de_holidays = holidays.Germany(subdiv=self.GERMAN_SUBDIVISION)
        self.ch_holidays = holidays.Switzerland(subdiv=self.SWISS_SUBDIVISION)
        self.fr_holidays = self.FR_ZONE_B_HOLIDAYS

    def _is_fr_holiday(self, dt: pd.Timestamp) -> int:
        """
        Check if a timestamp falls within French Zone B school holidays.
        
        Args:
            dt (pd.Timestamp): The observation timestamp.
        Returns:
            int: 1 if holiday, 0 otherwise.
        """
        d = dt.date()
        for start, end in self.fr_holidays:
            if start <= d <= end:
                return 1
        return 0

    def calculate_hci(self, row: pd.Series) -> float:
        """
        Calculate the Holiday Climate Index (HCI:Urban) for a given observation.
        
        Formula: 4*TC + 2*A + 3*P + 1*W
        TC (Thermal Comfort), A (Aesthetics), P (Precipitation), W (Wind).
        
        Args:
            row (pd.Series): Row containing weather features (temp, rain, wind, cloud_cover).
        Returns:
            float: Calculated HCI score.
        """
        # Thermal Comfort: Peak at 25°C
        tc = max(0, 10 - abs(row['temp'] - 25) * 0.5)
        # Aesthetics: Cloud cover impact
        a = (100 - row.get('cloud_cover', 50)) / 10
        # Precipitation impact
        p = max(0, 10 - row['rain'] * 2)
        # Wind speed impact
        w = max(0, 10 - row['wind'] * 0.5)
        
        return (4 * tc) + (2 * a) + (3 * p) + (1 * w)

    def process_data(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Main execution pipeline for feature transformation.
        
        Workflow:
        1. Operational Filtering: Exclude closed attractions
        2. Temporal Decomposition: Extract hour, month, weekday
        3. Regional Mapping: Apply holiday flags (DE/CH/FR)
        4. Weather Indexing: Compute HCI score
        5. Time-Series Engineering: Generate 10min and 60min lags
        
        Args:
            df_in (pd.DataFrame): Raw merged dataset.
        Returns:
            pd.DataFrame: Processed feature matrix ready for ML.
        """
        if df_in.empty:
            return pd.DataFrame()

        df = df_in.copy()
        
        # 1. Filter: Remove non-operational periods
        if 'is_open' in df.columns:
            df = df[df['is_open'] == True]
            if df.empty: return pd.DataFrame()
        
        # 2. Temporal Normalization
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
             df['datetime'] = pd.to_datetime(df['timestamp'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['ride_name', 'datetime'])
        
        # 3. Temporal Feature Extraction
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # 4. Regional Holiday Flags
        df['holiday_de_bw'] = df['datetime'].apply(lambda x: 1 if x in self.de_holidays else 0)
        df['holiday_ch_bs'] = df['datetime'].apply(lambda x: 1 if x in self.ch_holidays else 0)
        df['holiday_fr_zone_b'] = df['datetime'].apply(self._is_fr_holiday)
        
        # 5. Weather Index Calculation
        df['HCI_Urban'] = df.apply(self.calculate_hci, axis=1)
        
        # 6. Lag Feature Engineering (System Inertia)
        df['wait_time_lag_1'] = df.groupby('ride_name')['wait_time'].shift(1)  # T-10 min
        df['wait_time_lag_6'] = df.groupby('ride_name')['wait_time'].shift(6)  # T-60 min
        
        # 7. Identity Encoding
        if 'ride_id' not in df.columns:
            df['ride_id'] = df['ride_name'].astype('category').cat.codes
        else:
            df['ride_id'] = pd.to_numeric(df['ride_id'], errors='coerce').fillna(0)
            
        # Clean up NaN records created by lagging
        return df.dropna().reset_index(drop=True)