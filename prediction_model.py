"""
Machine Learning Core Module for AIRide PoC.

This module implements the training, evaluation, and inference logic for 
the ensemble model approach, combining Random Forest, Gradient Boosting, 
and LSTM architectures.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any

# Conditional Deep Learning Stack Initialization
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
    Manages the lifecycle of multiple predictive models for wait time forecasting.
    
    Supported Architectures:
    - Random Forest: Robust baseline for non-linear feature capture
    - Gradient Boosting: High-performance tree boosting
    - LSTM: Recurrent architecture for temporal dependencies (if TF available)
    """

    # Model Hyperparameters
    RF_ESTIMATORS = 100
    RF_MAX_DEPTH = 20
    GB_LEARNING_RATE = 0.1
    LSTM_UNITS = 50
    LSTM_LEARNING_RATE = 0.01

    def __init__(self):
        """Initialize model storage and scaling utilities."""
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def run_benchmark(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all available models and generate comparative performance metrics.
        
        Args:
            df (pd.DataFrame): Processed feature matrix.
        Returns:
            Dict[str, Any]: Nested dictionary containing model metrics and predictions.
        """
        # Feature Selection for Modeling
        features = [
            'hour', 'weekday', 'month', 'is_weekend', 'holiday_de_bw', 
            'holiday_fr_zone_b', 'holiday_ch_bs', 'temp', 'rain', 
            'HCI_Urban', 'wait_time_lag_1', 'wait_time_lag_6', 'ride_id'
        ]
        
        X = df[features].fillna(0)
        y = df['wait_time']
        
        # Chronological Split (80% Train, 20% Test)
        split_point = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        results = {}

        # 1. Random Forest Training
        rf = RandomForestRegressor(
            n_estimators=self.RF_ESTIMATORS, 
            max_depth=self.RF_MAX_DEPTH, 
            random_state=42
        )
        rf.fit(X_train, y_train)
        results['Random Forest'] = self._evaluate(rf, X_test, y_test, 'rf')
        self.models['rf'] = rf

        # 2. Gradient Boosting Training
        gb = GradientBoostingRegressor(
            n_estimators=self.RF_ESTIMATORS, 
            learning_rate=self.GB_LEARNING_RATE, 
            max_depth=5, 
            random_state=42
        )
        gb.fit(X_train, y_train)
        results['Gradient Boosting'] = self._evaluate(gb, X_test, y_test, 'gb')
        self.models['gb'] = gb

        # 3. LSTM Training (Optional)
        if TF_AVAILABLE:
            # Neural network data preparation: Scaling is mandatory
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            self.scalers['lstm'] = scaler

            # Reshape for LSTM: [samples, time_steps, features]
            X_train_re = X_train_sc.reshape((X_train_sc.shape[0], 1, X_train_sc.shape[1]))
            X_test_re = X_test_sc.reshape((X_test_sc.shape[0], 1, X_test_sc.shape[1]))

            lstm_model = Sequential([
                Input(shape=(1, X_train_sc.shape[1])),
                LSTM(self.LSTM_UNITS, activation='relu'),
                Dense(1)
            ])
            lstm_model.compile(optimizer=Adam(learning_rate=self.LSTM_LEARNING_RATE), loss='mse')
            lstm_model.fit(X_train_re, y_train, epochs=10, batch_size=32, verbose=0, shuffle=False)
            
            results['LSTM'] = self._evaluate(lstm_model, X_test_re, y_test, 'lstm')
            self.models['lstm'] = lstm_model
        
        return results

    def _evaluate(self, model: Any, X_test: np.ndarray, y_test: pd.Series, mode: str) -> Dict[str, Any]:
        """
        Calculate regression metrics for model performance assessment.
        
        Args:
            model (Any): Trained model object.
            X_test (np.ndarray): Test features.
            y_test (pd.Series): Test ground truth.
            mode (str): Architecture type identifier.
        Returns:
            Dict: RMSE, MAE, R2 scores and raw predictions.
        """
        if mode == 'lstm':
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
        """
        Generate wait time forecasts using all trained ensemble members.
        
        Args:
            input_df (pd.DataFrame): Single observation feature set.
        Returns:
            Dict[str, float]: Model name to wait time prediction mapping.
        """
        preds = {}
        if 'rf' in self.models:
            preds['Random Forest'] = self.models['rf'].predict(input_df)[0]
        if 'gb' in self.models:
            preds['Gradient Boosting'] = self.models['gb'].predict(input_df)[0]
        if 'lstm' in self.models and TF_AVAILABLE:
            sc_input = self.scalers['lstm'].transform(input_df)
            re_input = sc_input.reshape((sc_input.shape[0], 1, sc_input.shape[1]))
            preds['LSTM'] = float(self.models['lstm'].predict(re_input, verbose=0)[0][0])
        return preds