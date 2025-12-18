import requests
import pandas as pd
import time
import os
from datetime import datetime

# --- KONFIGURATION ---
# Europa-Park ID = 51
API_URL = "https://queue-times.com/parks/51/queue_times.json"
CSV_FILE = "real_waiting_times.csv"
POLLING_INTERVAL = 600  # Alle 10 Minuten

def fetch_live_data():
    try:
        # User-Agent ist wichtig, damit wir nicht geblockt werden
        headers = {'User-Agent': 'AIRide-PoC-StudentProject/1.0'}
        response = requests.get(API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        records = []
        timestamp = datetime.now()
        
        # JSON parsen (Lands -> Rides)
        for land in data.get('lands', []):
            for ride in land.get('rides', []):
                # Wir sammeln ALLES. Filtern können wir später beim Training.
                records.append({
                    "timestamp": timestamp,
                    "ride_id": ride['id'],
                    "ride_name": ride['name'],
                    "is_open": ride['is_open'],
                    "wait_time": ride['wait_time'],
                    "last_updated": ride['last_updated']
                })
        
        print(f"✅ {timestamp.strftime('%H:%M:%S')}: {len(records)} Attraktionen abgerufen.")
        return records
    except Exception as e:
        print(f"⚠️ Fehler beim Abruf: {e}")
        return []

def save_to_csv(new_records):
    if not new_records: return
    
    df_new = pd.DataFrame(new_records)
    
    # Datei anlegen oder anhängen
    if not os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, index=False)
        print(f"🆕 Datei '{CSV_FILE}' neu erstellt.")
    else:
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)

def main():
    print("🚜 AIRide Harvester gestartet (Real Data Mode)...")
    print(f"📂 Speicherort: {os.path.abspath(CSV_FILE)}")
    print("Drücke STRG+C zum Beenden.\n")
    
    while True:
        records = fetch_live_data()
        save_to_csv(records)
        time.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    main()