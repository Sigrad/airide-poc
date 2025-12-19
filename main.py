"""
Main Entry Point for AIRide PoC Pipeline Testing.

This script provides a command-line interface for testing the complete data
processing and model training pipeline in a local development environment.
It orchestrates the workflow from raw data acquisition to model benchmarking.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

from data_harvester import DataHarvester
from feature_engine import FeatureEngineer
from model_trainer import ModelTrainer

if __name__ == "__main__":
    print("=== Testing Pipeline Locally ===")
    h = DataHarvester()
    df = h.fetch_historical_data()
    
    if not df.empty:
        fe = FeatureEngineer()
        df_final = fe.enrich_data(df)
        print(f"Data ready: {len(df_final)} rows")
        
        mt = ModelTrainer()
        mt.train_and_evaluate(df_final)
    else:
        print("No data.")