"""
Meteorological Data Integration Module for AIRide PoC.

This module provides high-resolution weather data retrieval via the 
Open-Meteo API. It implements robust caching and retry mechanisms to 
ensure stable integration with the crowd flow prediction pipeline.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from typing import Optional

class WeatherService:
    """
    Handles communication with meteorological APIs for historical and forecast data.
    
    This class wraps the Open-Meteo API to provide specific variables:
    - Temperature (2m above ground)
    - Precipitation (Rain)
    - Wind speed (10m above ground)
    - Cloud coverage
    """

    # Geographic Configuration: Europa-Park, Rust (DE)
    LATITUDE = 48.26
    LONGITUDE = 7.72
    API_URL = "https://api.open-meteo.com/v1/forecast"
    TIMEZONE = "Europe/Berlin"

    def __init__(self):
        """
        Initialize the API client with caching and retry logic.
        
        Caching: Expires after 10 minutes to respect API constraints.
        Retries: Up to 5 attempts with exponential backoff.
        """
        cache_session = requests_cache.CachedSession('.cache', expire_after=600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)

    def fetch_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve hourly meteorological records for a specific time window.
        
        Args:
            start_date (str): Window start (ISO format YYYY-MM-DD).
            end_date (str): Window end (ISO format YYYY-MM-DD).
        Returns:
            pd.DataFrame: Hourly weather features.
        """
        params = {
            "latitude": self.LATITUDE, 
            "longitude": self.LONGITUDE,
            "start_date": start_date, 
            "end_date": end_date,
            "hourly": ["temperature_2m", "rain", "wind_speed_10m", "cloud_cover"],
            "timezone": self.TIMEZONE
        }
        
        try:
            # API Request via openmeteo-requests client
            responses = self.client.weather_api(self.API_URL, params=params)
            
            if not responses:
                return pd.DataFrame()

            response = responses[0]
            hourly = response.Hourly()
            
            # Construct time-series dataframe from API response structure
            df = pd.DataFrame({
                "datetime": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            })
            
            # Extract variables from nested response variable list
            df['temp'] = hourly.Variables(0).ValuesAsNumpy()
            df['rain'] = hourly.Variables(1).ValuesAsNumpy()
            df['wind'] = hourly.Variables(2).ValuesAsNumpy()
            df['cloud_cover'] = hourly.Variables(3).ValuesAsNumpy()
            
            # Formatting: Convert to naive datetime for seamless dataframe alignment
            df['datetime'] = df['datetime'].dt.tz_convert(None) 
            
            return df
            
        except Exception as e:
            print(f"[WeatherService Error] {e}")
            return pd.DataFrame()