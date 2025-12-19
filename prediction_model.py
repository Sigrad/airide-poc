import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any

class PredictionModel:
    def __init__(self):
        self.models = {}
        self.feature_columns = [
            'hour', 'weekday', 'month', 'is_weekend', 
            'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
            'temp', 'rain', 'HCI_Urban', 
            'wait_time_lag_1', 'wait_time_lag_6', 'ride_id'
        ]

    def run_benchmark(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains and evaluates RF and GB models."""
        X = df[self.feature_columns]
        y = df['wait_time']
        
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['rf'] = rf

        # Gradient Boosting
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X_train, y_train)
        self.models['gb'] = gb

        return {
            'Random Forest': self._get_metrics(rf, X_test, y_test),
            'Gradient Boosting': self._get_metrics(gb, X_test, y_test)
        }

    def _get_metrics(self, model, X_test, y_test):
        pred = model.predict(X_test)
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'r2': r2_score(y_test, pred),
            'predictions': pred,
            'actuals': y_test.values
        }

    def predict_ensemble(self, input_df: pd.DataFrame):
        results = {}
        if 'rf' in self.models:
            results['Random Forest'] = self.models['rf'].predict(input_df)[0]
        if 'gb' in self.models:
            results['Gradient Boosting'] = self.models['gb'].predict(input_df)[0]
        return results