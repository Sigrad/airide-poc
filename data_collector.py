# In data_collector.py - fetch_historical_data Methode anpassen:
def fetch_historical_data(self, days_back: int = 60) -> pd.DataFrame:
    if not os.path.exists(self.storage_path):
        return pd.DataFrame()

    try:
        df_historical = pd.read_csv(self.storage_path)
        if df_historical.empty:
            return pd.DataFrame()
            
        df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
        
        # Wetter-Synchronisation
        start_date = df_historical['timestamp'].min().strftime('%Y-%m-%d')
        end_date = df_historical['timestamp'].max().strftime('%Y-%m-%d')
        
        df_weather = self.weather_service.fetch_weather_data(start_date, end_date)
        
        if not df_weather.empty:
            # Round to hour for matching
            df_historical['merge_key'] = df_historical['timestamp'].dt.round('H')
            df_weather['merge_key'] = pd.to_datetime(df_weather['datetime']).dt.round('H')
            
            df_merged = pd.merge(df_historical, df_weather, on='merge_key', how='left')
            # Clean up temporary columns
            return df_merged.drop(columns=['merge_key', 'datetime'])
        
        return df_historical
    except Exception as e:
        print(f"Data Retrieval Error: {e}")
        return pd.DataFrame()