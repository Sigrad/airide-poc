import requests
import pandas as pd
import time
import os
from datetime import datetime

# --- KONFIGURATION (Europa-Park) ---
API_URL = "https://queue-times.com/parks/51/queue_times.json"
CSV_FILE = "real_waiting_times.csv"
POLLING_INTERVAL = 600  # Alle 10 Minuten (wissenschaftlicher Standard)

def fetch_live_data():
    try:
        print(f"📡 Request: {datetime.now().strftime('%H:%M:%S')}...")
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        records = []
        timestamp = datetime.now()
        
        # Parsing der JSON-Struktur (Lands -> Rides)
        for land in data.get('lands', []):
            for ride in land.get('rides', []):
                records.append({
                    "timestamp": timestamp,
                    "ride_id": ride['id'],
                    "ride_name": ride['name'],
                    "is_open": ride['is_open'],
                    "wait_time": ride['wait_time'],
                    "last_updated": ride['last_updated']
                })
        return records
    except Exception as e:
        print(f"⚠️ Fehler: {e}")
        return []

def save_to_csv(new_records):
    if not new_records: return
    
    df_new = pd.DataFrame(new_records)
    
    # Wenn Datei noch nicht existiert -> Header schreiben
    if not os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
        print(f"🆕 Datei '{CSV_FILE}' erstellt.")
    else:
        # Anhängen ohne Header
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
        print(f"💾 {len(new_records)} Datensätze gespeichert.")

def main():
    print("🚜 AIRide Data Harvester gestartet...")
    print("Drücke STRG+C zum Beenden.")
    
    while True:
        records = fetch_live_data()
        save_to_csv(records)
        time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    main()