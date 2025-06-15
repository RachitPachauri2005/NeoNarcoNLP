import unittest
import pandas as pd
from pathlib import Path

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.features_path = Path(__file__).parent.parent / 'neonarconlp' / 'data' / 'processed' / 'features.csv'

    def test_features_file_exists(self):
        self.assertTrue(self.features_path.exists())
        df = pd.read_csv(self.features_path)
        self.assertIn('sentiment', df.columns)
        self.assertIn('sentence_length', df.columns)
        self.assertTrue(len(df) > 0)

if __name__ == '__main__':
    unittest.main() 