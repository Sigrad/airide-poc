import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- SCHRITT 1: DATEN GENERIEREN (Synthetischer Europa-Park Datensatz) ---
print("Generiere synthetische Daten...")
np.random.seed(42)
n_samples = 10000

data = pd.DataFrame({
    'hour': np.random.randint(9, 19, n_samples),  # 09:00 bis 18:00
    'temp': np.random.uniform(5, 35, n_samples),  # 5°C bis 35°C
    'rain': np.random.uniform(0, 1, n_samples),   # 0% bis 100% Regen
    'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]), # 30% Wochenende
    'is_holiday': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% Ferien
})

# Feature Engineering: Wetter-Score (Dein USP aus dem Paper)
# Formel: Je wärmer und trockener, desto höher der Score (0 bis 1)
data['weather_score'] = (data['temp'] / 40) * (1 - data['rain'])

# Zielvariable simulieren: Wartezeit
# Basiswert + Zeitfaktor + Wetterfaktor + Wochenendzuschlag
data['wait_time'] = 10 + \
                    (data['hour'].apply(lambda h: 20 if 11 <= h <= 15 else 0)) + \
                    (data['weather_score'] * 30) + \
                    (data['is_weekend'] * 25) + \
                    (data['is_holiday'] * 15) + \
                    np.random.normal(0, 5, n_samples) # Zufallsrauschen

data['wait_time'] = data['wait_time'].clip(0, 120).astype(int) # Keine negativen Zeiten

# --- SCHRITT 2: MODELL TRAINIEREN ---
print("Trainiere Random Forest...")

X = data[['hour', 'temp', 'rain', 'is_weekend', 'is_holiday', 'weather_score']]
y = data['wait_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation (für dein Paper!)
y_pred = model.predict(X_test)
print(f"--- ERGEBNISSE ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} Minuten")

# --- SCHRITT 3: SPEICHERN ---
joblib.dump(model, 'airide_model.pkl')
print("Modell gespeichert als 'airide_model.pkl'. BEREIT FÜR GITHUB!")