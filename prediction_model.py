import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List

# Conditional import for TensorFlow to maintain environment flexibility
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
    Manages the training, evaluation, and inference of multiple machine learning 
    architectures including Random Forest, Gradient Boosting, and LSTM.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = [
            'hour', 'weekday', 'month', 'is_weekend', 
            'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
            'temp', 'rain', 'HCI_Urban', 
            'wait_time_lag_1', 'wait_time_lag_6', 'ride_id'
        ]

    def run_benchmark(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes a comparative training and evaluation run for all available models.
        
        Args:
            dataframe (pd.DataFrame): Preprocessed features and target variable.
            
        Returns:
            Dict[str, Any]: Performance metrics and prediction results per model.
        """
        if dataframe.empty:
            raise ValueError("Provided DataFrame is empty.")

        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in dataframe.columns:
                dataframe[feature] = 0
            
        x_data = dataframe[self.feature_columns]
        y_data = dataframe['wait_time']
        
        # Chronological split (80% training, 20% testing)
        split_index = int(len(dataframe) * 0.8)
        x_train, x_test = x_data.iloc[:split_index], x_data.iloc[:split_index]
        y_train, y_test = y_data.iloc[:split_index], y_data.iloc[:split_index]

        benchmark_results = {}

        # Architecture 1: Random Forest
        rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        rf_regressor.fit(x_train, y_train)
        self.models['rf'] = rf_regressor
        benchmark_results['Random Forest'] = self._evaluate_model(rf_regressor, x_test, y_test, 'rf')

        # Architecture 2: Gradient Boosting
        gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        gb_regressor.fit(x_train, y_train)
        self.models['gb'] = gb_regressor
        benchmark_results['Gradient Boosting'] = self._evaluate_model(gb_regressor, x_test, y_test, 'gb')

        # Architecture 3: LSTM (Deep Learning)
        if TF_AVAILABLE:
            lstm_results = self._train_lstm(x_train, y_train, x_test, y_test)
            benchmark_results['LSTM'] = lstm_results
        
        return benchmark_results

    def _train_lstm(self, x_train: pd.DataFrame, y_train: pd.Series, 
                   x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Internal helper to manage LSTM specific data transformations and training."""
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        self.scalers['lstm'] = scaler

        # Reshape for LSTM: [samples, time_steps, features]
        x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
        x_test_reshaped = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))

        model = Sequential([
            Input(shape=(1, x_train_scaled.shape[1])),
            LSTM(50, activation='relu', return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        model.fit(x_train_reshaped, y_train, epochs=10, batch_size=32, verbose=0, shuffle=False)
        
        self.models['lstm'] = model
        return self._evaluate_model(model, x_test_reshaped, y_test, 'lstm')

    def _evaluate_model(self, model: Any, x_test: np.ndarray, 
                       y_test: pd.Series, model_type: str) -> Dict[str, Any]:
        """Calculates standard regression metrics for model validation."""
        if model_type == 'lstm':
            predictions = model.predict(x_test, verbose=0).flatten()
        else:
            predictions = model.predict(x_test)
            
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'predictions': predictions,
            'actuals': y_test.values
        }

    def predict_ensemble(self, input_features: pd.DataFrame) -> Dict[str, float]:
        """Generates predictions across all initialized models."""
        ensemble_predictions = {}
        
        if 'rf' in self.models:
            ensemble_predictions['Random Forest'] = self.models['rf'].predict(input_features)[0]
        
        if 'gb' in self.models:
            ensemble_predictions['Gradient Boosting'] = self.models['gb'].predict(input_features)[0]
            
        if 'lstm' in self.models and TF_AVAILABLE:
            scaler = self.scalers['lstm']
            scaled_input = scaler.transform(input_features)
            reshaped_input = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
            lstm_pred = self.models['lstm'].predict(reshaped_input, verbose=0)[0][0]
            ensemble_predictions['LSTM'] = float(lstm_pred)
            
        return ensemble_predictions