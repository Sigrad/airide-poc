import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
        
    def train_and_evaluate(self, df):
        features = ['hour', 'weekday', 'month', 'is_weekend', 
                    'holiday_de_bw', 'holiday_fr_zone_b', 'holiday_ch_bs',
                    'temp', 'rain', 'HCI_Urban', 
                    'wait_time_lag_1', 'wait_time_lag_6',
                    'ride_id']
        
        for f in features:
            if f not in df.columns: df[f] = 0
                
        X = df[features]
        y = df['wait_time']
        
        print(f"Training on {len(df)} rows...")
        
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        baseline_pred = np.full(len(y_test), y_train.mean())
        
        self.model.fit(X_train, y_train)
        rf_pred = self.model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        mae = mean_absolute_error(y_test, rf_pred)
        r2 = r2_score(y_test, rf_pred)
        
        print(f"Result: RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.2f}")
        
        dm_stat, p_value = self.diebold_mariano_test(y_test, baseline_pred, rf_pred)
        print(f"Diebold-Mariano Test: p-value={p_value:.6f}")
        if p_value < 0.05:
            print("   -> SIGNIFICANT: Model beats baseline.")
        else:
            print("   -> NOT SIGNIFICANT: Model not better than baseline.")
            
        return self.model

    def diebold_mariano_test(self, y_true, y_pred1, y_pred2):
        y_true = np.array(y_true)
        y_pred1 = np.array(y_pred1)
        y_pred2 = np.array(y_pred2)
        
        e1 = (y_true - y_pred1)**2
        e2 = (y_true - y_pred2)**2
        
        d = e1 - e2
        d_mean = np.mean(d)
        gamma0 = np.var(d)
        
        if gamma0 == 0: return 0, 1
        
        dm_stat = d_mean / np.sqrt(gamma0 / len(y_true))
        p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
        
        return dm_stat, p_value