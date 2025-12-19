import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

class WeatherService:
    """
    Interface for the Open-Meteo API to retrieve historical and forecasted 
    meteorological variables.
    """
    def __init__(self):
        # Cache responses for 10 minutes to prevent API rate limiting
        self.session = requests_cache.CachedSession('.cache', expire_after=600)
        self.retry_session = retry(self.session, retries=5, backoff_factor=0.2)
        self.api_client = openmeteo_requests.Client(session=self.retry_session)
        self.endpoint = "https://api.open-meteo.com/v1/forecast"

    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Queries weather features: temperature, rain, wind, and cloud cover.
        """
        query_params = {
            "latitude": 48.26, 
            "longitude": 7.72,
            "start_date": start_date, 
            "end_date": end_date,
            "hourly": ["temperature_2m", "rain", "wind_speed_10m", "cloud_cover"],
            "timezone": "Europe/Berlin"
        }
        
        try:
            api_responses = self.api_client.weather_api(self.endpoint, params=query_params)
            if not api_responses:
                return pd.DataFrame()

            result = api_responses[0]
            hourly_data = result.Hourly()
            
            time_range = pd.date_range(
                start=pd.to_datetime(hourly_data.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly_data.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly_data.Interval()),
                inclusive="left"
            )
            
            df_weather = pd.DataFrame({"datetime": time_range})
            df_weather['temp'] = hourly_data.Variables(0).ValuesAsNumpy()
            df_weather['rain'] = hourly_data.Variables(1).ValuesAsNumpy()
            df_weather['wind'] = hourly_data.Variables(2).ValuesAsNumpy()
            df_weather['cloud_cover'] = hourly_data.Variables(3).ValuesAsNumpy()
            
            df_weather['datetime'] = df_weather['datetime'].dt.tz_convert(None) 
            return df_weather
            
        except Exception as e:
            print(f"Weather API Error: {str(e)}")
            return pd.DataFrame()