import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from typing import Dict, Any, Tuple

class PredictionModel:
    """
    Wrapper class for the Random Forest Regressor tailored for time-series forecasting.
    Includes automated hyperparameter tuning via RandomizedSearchCV and statistical testing.
    """

    def __init__(self):
        """Initializes the PredictionModel with a placeholder estimator."""
        self.model = RandomForestRegressor(random_state=42)
        self.best_params = {}

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains the Random Forest model using RandomizedSearchCV for hyperparameter optimization
        and evaluates it on a hold-out test set.

        Args:
            df (pd.DataFrame): The preprocessed feature matrix containing target 'wait_time'.

        Returns:
            Dict[str, Any]: A dictionary containing the trained model object, performance metrics
                            (RMSE, MAE, R2), p-value of the DM-test, and feature importances.
        
        Raises:
            ValueError: If the dataframe is empty or lacks required columns.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # --- 1. Sanity Check for Synthetic Data ---
        # Checks if the dataset relies heavily on heuristics (bootstrap data).
        if 'is_synthetic' in df.columns:
            synthetic_ratio = df['is_synthetic'].mean()
            if synthetic_ratio > 0.5:
                warnings.warn(
                    f"WARNING: High reliance on synthetic data ({synthetic_ratio:.1%}). "
                    "Model validity may be compromised and reflect heuristics rather than reality.",
                    UserWarning
                )
        
        # --- 2. Feature Selection ---
        features = ['hour', 'weekday', 'month', 'is_weekend', 
                    'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
                    'temp', 'rain', 'HCI_Urban', 
                    'wait_time_lag_1', 'wait_time_lag_6',
                    'ride_id']
        
        # Ensure robustness: Fill missing columns with 0
        for f in features:
            if f not in df.columns:
                df[f] = 0
                
        X = df[features]
        y = df['wait_time']
        
        # --- 3. Chronological Train-Test Split ---
        # We perform a strict chronological split (last 20%) to avoid look-ahead bias.
        split_idx = int(len(df) * 0.8)
        
        X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # --- 4. Hyperparameter Tuning (RandomizedSearchCV) ---
        print(f"Starting Hyperparameter Tuning on {len(X_train_full)} rows...")
        
        # Define search space
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        # Use TimeSeriesSplit for CV to respect temporal order during tuning
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_dist,
            n_iter=15,  # Control computational cost
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train_full, y_train_full)
        
        # Update model with best estimator
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        print(f"Best Parameters found: {self.best_params}")

        # --- 5. Evaluation on Hold-Out Set ---
        rf_pred = self.model.predict(X_test)
        
        # Baseline: Mean prediction (Naive Forecast)
        baseline_pred = np.full(len(y_test), y_train_full.mean())
        
        # Metrics Calculation
        rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        mae = mean_absolute_error(y_test, rf_pred)
        r2 = r2_score(y_test, rf_pred)
        
        # Feature Importance Extraction
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Statistical Significance Test
        dm_stat, p_value = self._diebold_mariano_test(y_test, baseline_pred, rf_pred)
        
        return {
            'model': self.model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'p_value': p_value,
            'feature_importance': feature_importance,
            'best_params': self.best_params
        }

    def _diebold_mariano_test(self, y_true: np.array, y_pred1: np.array, y_pred2: np.array) -> Tuple[float, float]:
        """
        Performs the Diebold-Mariano test to compare the predictive accuracy of two models.
        
        Args:
            y_true (np.array): Ground truth values.
            y_pred1 (np.array): Predictions from model 1 (Reference/Baseline).
            y_pred2 (np.array): Predictions from model 2 (Candidate).

        Returns:
            Tuple[float, float]: A tuple containing (DM-statistic, p-value).
            Returns (0.0, 1.0) if variance is zero (models are identical).
        """
        y_true = np.array(y_true)
        y_pred1 = np.array(y_pred1)
        y_pred2 = np.array(y_pred2)
        
        # Squared error loss
        e1 = (y_true - y_pred1)**2
        e2 = (y_true - y_pred2)**2
        
        # Loss differential
        d = e1 - e2
        
        d_mean = np.mean(d)
        gamma0 = np.var(d)
        
        # Robustness Check: Avoid Division by Zero
        if gamma0 < 1e-9: 
            return 0.0, 1.0 
            
        dm_stat = d_mean / np.sqrt(gamma0 / len(y_true))
        
        # Two-sided p-value (Normal distribution approximation)
        p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
        
        return dm_stat, p_value