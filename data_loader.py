import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

class DataLoader:
    def __init__(self):
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_history(self, start_date, end_date):
        print(f"Fetching Weather: {start_date} -> {end_date}")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 48.26, "longitude": 7.72,
            "start_date": start_date, "end_date": end_date,
            "hourly": ["temperature_2m", "rain", "wind_speed_10m", "cloud_cover"]
        }
        try:
            r = self.openmeteo.weather_api(url, params=params)[0]
            hourly = r.Hourly()
            
            data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
            )}
            df = pd.DataFrame(data)
            df['temp'] = hourly.Variables(0).ValuesAsNumpy()
            df['rain'] = hourly.Variables(1).ValuesAsNumpy()
            df['wind'] = hourly.Variables(2).ValuesAsNumpy()
            df['cloud_cover'] = hourly.Variables(3).ValuesAsNumpy()
            
            df = df.rename(columns={'date': 'datetime'})
            df['datetime'] = df['datetime'].dt.tz_localize(None)
            return df
        except Exception as e:
            print(f"Weather API Error: {e}")
            return pd.DataFrame()