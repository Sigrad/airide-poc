import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

class DataLoader:
    def __init__(self):
        # Cache etwas verkürzen (10 min), damit wir frische Daten kriegen
        cache_session = requests_cache.CachedSession('.cache', expire_after=600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_history(self, start_date, end_date):
        print(f"Fetching Weather (Forecast API): {start_date} -> {end_date}")
        
        # WICHTIG: Wechsel auf Forecast API für aktuelle Daten!
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": 48.26, 
            "longitude": 7.72,
            "start_date": start_date, 
            "end_date": end_date,
            "hourly": ["temperature_2m", "rain", "wind_speed_10m", "cloud_cover"],
            "timezone": "Europe/Berlin"  # Wichtig für korrekte Zuordnung
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            
            if not responses:
                print("No weather data returned.")
                return pd.DataFrame()

            response = responses[0]
            hourly = response.Hourly()
            
            # Zeitachse aufbauen
            data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            df = pd.DataFrame(data)
            df['temp'] = hourly.Variables(0).ValuesAsNumpy()
            df['rain'] = hourly.Variables(1).ValuesAsNumpy()
            df['wind'] = hourly.Variables(2).ValuesAsNumpy()
            df['cloud_cover'] = hourly.Variables(3).ValuesAsNumpy()
            
            # Formatierung
            df = df.rename(columns={'date': 'datetime'})
            # Zeitzone entfernen für sauberen Merge mit CSV
            df['datetime'] = df['datetime'].dt.tz_convert(None) 
            
            return df
            
        except Exception as e:
            print(f"Weather API Error: {e}")
            return pd.DataFrame()