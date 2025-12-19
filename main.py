"""
Local Pipeline Execution Runner for AIRide PoC.

This script acts as the primary entry point for testing the end-to-end 
AIRide data pipeline in a local environment, bypassing the Streamlit UI.

Author: Sakir Doenmez
Project: AIRide PoC
Academic Context: PA HS 25
Institution: ZHAW
"""

from data_collector import DataCollector
from feature_engineering import FeatureEngineering
from prediction_model import PredictionModel

def execute_local_pipeline():
    """
    Orchestrate a complete local test run of the prediction system.
    
    Workflow:
    1. Historical Data Harvesting (Queue API + Weather)
    2. Feature Enrichment Pipeline
    3. Multi-Model Benchmark Training
    """
    print("=== [AIRide] Starting Local Pipeline Benchmark ===")
    
    # 1. Initialization and Data Retrieval
    h = DataHarvester() # Note: Mapping DataCollector as Harvester for legacy support
    df_raw = h.fetch_historical_data()
    
    if not df_raw.empty:
        # 2. Data Transformation
        fe = FeatureEngineer()
        df_processed = fe.enrich_data(df_raw)
        print(f"[Pipeline] Preprocessing complete. Feature matrix: {len(df_processed)} rows")
        
        # 3. Model Orchestration
        mt = ModelTrainer()
        results = mt.train_and_evaluate(df_processed)
        print("[Pipeline] Benchmarking complete. Ready for evaluation.")
    else:
        print("[Pipeline Warning] No data retrieved. Aborting execution.")

if __name__ == "__main__":
    execute_local_pipeline()