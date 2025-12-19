import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from typing import Dict, Any, Tuple

# TensorFlow imports (wrapped in try-except to avoid crashes if not installed)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class PredictionModel:
    """
    Manages training and evaluation of multiple model architectures:
    1. Random Forest (Baseline)
    2. Gradient Boosting (Booster)
    3. LSTM (Deep Learning)
    """

    def __init__(self):
        self.models = {}
        self.scalers = {} # Needed for LSTM







    def run_benchmark(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Trains all available models and compares metrics."""
        if df.empty: raise ValueError("Empty DataFrame")

        # 1. Prepare Data





















        features = ['hour', 'weekday', 'month', 'is_weekend', 
                    'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
                    'temp', 'rain', 'HCI_Urban', 
                    'wait_time_lag_1', 'wait_time_lag_6', 'ride_id']

        

        for f in features:
            if f not in df.columns: df[f] = 0
            

        X = df[features]
        y = df['wait_time']
        
        # Chronological Split (80/20)

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]



































        results = {}

        # --- MODEL 1: Random Forest ---
        print("Training Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        rf.fit(X_train, y_train)
        results['Random Forest'] = self._evaluate(rf, X_test, y_test, 'rf')
        self.models['rf'] = rf

        # --- MODEL 2: Gradient Boosting ---
        print("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        results['Gradient Boosting'] = self._evaluate(gb, X_test, y_test, 'gb')
        self.models['gb'] = gb

        # --- MODEL 3: LSTM (Optional) ---
        if TF_AVAILABLE:
            print("Training LSTM...")
            # Scale data for NN
            scaler_X = StandardScaler()
            X_train_sc = scaler_X.fit_transform(X_train)
            X_test_sc = scaler_X.transform(X_test)
            self.scalers['lstm'] = scaler_X

            # Reshape for LSTM [samples, timesteps, features]
            X_train_re = X_train_sc.reshape((X_train_sc.shape[0], 1, X_train_sc.shape[1]))
            X_test_re = X_test_sc.reshape((X_test_sc.shape[0], 1, X_test_sc.shape[1]))

            # Build Net
            lstm = Sequential([
                Input(shape=(1, X_train_sc.shape[1])),
                LSTM(50, activation='relu', return_sequences=False),
                Dense(1)
            ])
            lstm.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            
            # Train (Verbose=0 to keep logs clean)
            lstm.fit(X_train_re, y_train, epochs=10, batch_size=32, verbose=0, shuffle=False)
            
            results['LSTM'] = self._evaluate(lstm, X_test_re, y_test, 'lstm')
            self.models['lstm'] = lstm
        
        return results

    def _evaluate(self, model, X_test, y_test, model_type):
        """Generic evaluation helper."""
        if model_type == 'lstm':
            pred = model.predict(X_test, verbose=0).flatten()
        else:
            pred = model.predict(X_test)
            
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'r2': r2_score(y_test, pred),
            'predictions': pred,
            'actuals': y_test.values


        }

    def predict_ensemble(self, input_df: pd.DataFrame) -> Dict[str, float]:
        """Returns predictions from all trained models for a single input."""
        preds = {}
        # RF
        if 'rf' in self.models:
            preds['Random Forest'] = self.models['rf'].predict(input_df)[0]
        
        # GB
        if 'gb' in self.models:
            preds['Gradient Boosting'] = self.models['gb'].predict(input_df)[0]

        # LSTM
        if 'lstm' in self.models and TF_AVAILABLE:
            sc = self.scalers['lstm']
            input_sc = sc.transform(input_df)
            input_re = input_sc.reshape((input_sc.shape[0], 1, input_sc.shape[1]))
            preds['LSTM'] = float(self.models['lstm'].predict(input_re, verbose=0)[0][0])
            
        return preds