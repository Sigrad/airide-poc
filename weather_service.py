"""
Meteorological Data Integration Service for AIRide PoC.

This module interfaces with the Open-Meteo API to retrieve historical and
forecasted weather data. It handles API authentication, caching, and
data normalization for integration into the primary prediction pipeline.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

class WeatherService:
    """
    Handles interaction with the Open-Meteo API for historical and forecast data.
    """
    def __init__(self):
        # Cache responses for 10 minutes to respect API limits
        cache_session = requests_cache.CachedSession('.cache', expire_after=600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data(self, start_date, end_date):
        """
        Fetches hourly weather variables: temp, rain, wind, cloud cover.
        Uses the Forecast API to ensure availability of recent data.
        """
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": 48.26, 
            "longitude": 7.72,
            "start_date": start_date, 
            "end_date": end_date,
            "hourly": ["temperature_2m", "rain", "wind_speed_10m", "cloud_cover"],
            "timezone": "Europe/Berlin"
        }
        
        try:
            responses = self.client.weather_api(url, params=params)
            
            if not responses:
                return pd.DataFrame()

            response = responses[0]
            hourly = response.Hourly()
            
            # Construct DataFrame from hourly data arrays
            data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            df = pd.DataFrame(data)
            df['temp'] = hourly.Variables(0).ValuesAsNumpy()
            df['rain'] = hourly.Variables(1).ValuesAsNumpy()
            df['wind'] = hourly.Variables(2).ValuesAsNumpy()
            df['cloud_cover'] = hourly.Variables(3).ValuesAsNumpy()
            
            # Formatting: remove timezone for easier merging
            df = df.rename(columns={'date': 'datetime'})
            df['datetime'] = df['datetime'].dt.tz_convert(None) 
            
            return df
            
        except Exception as e:
            print(f"Weather API Error: {e}")
            return pd.DataFrame()