import unittest
import pandas as pd
import numpy as np
from prediction_model import PredictionModel

class TestAIRideLogic(unittest.TestCase):
    """
    Unit tests for the core prediction and data logic.
    """
    
    def setUp(self):
        self.predictor = PredictionModel()
        # Create a minimal valid mock dataset
        self.mock_df = pd.DataFrame({
            'hour': np.random.randint(0, 24, 100),
            'weekday': np.random.randint(0, 7, 100),
            'month': np.random.randint(1, 13, 100),
            'is_weekend': np.random.randint(0, 2, 100),
            'holiday_de_bw': np.zeros(100),
            'holiday_fr_zone_b': np.zeros(100),
            'holiday_ch_bs': np.zeros(100),
            'temp': np.random.normal(20, 5, 100),
            'rain': np.zeros(100),
            'HCI_Urban': np.random.randint(0, 100, 100),
            'wait_time_lag_1': np.random.randint(0, 60, 100),
            'wait_time_lag_6': np.random.randint(0, 60, 100),
            'ride_id': np.random.randint(1, 10, 100),
            'wait_time': np.random.randint(0, 60, 100)
        })

    def test_benchmark_execution(self):
        """Verifies that the benchmark runs and produces results for RF and GB."""
        results = self.predictor.run_benchmark(self.mock_df)
        self.assertIn('Random Forest', results)
        self.assertIn('Gradient Boosting', results)
        self.assertGreater(results['Random Forest']['r2'], -1.0)

    def test_ensemble_prediction_structure(self):
        """Ensures the ensemble returns the correct model keys."""
        self.predictor.run_benchmark(self.mock_df)
        single_row = self.mock_df[self.predictor.feature_columns].iloc[[0]]
        preds = self.predictor.predict_ensemble(single_row)
        self.assertTrue(len(preds) >= 2)

if __name__ == '__main__':
    unittest.main()