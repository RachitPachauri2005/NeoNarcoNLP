import unittest
import pandas as pd
from pathlib import Path
import json

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        self.raw_path = Path(__file__).parent.parent / 'neonarconlp' / 'data' / 'raw' / 'synthetic_chat_logs.json'
        self.clean_path = Path(__file__).parent.parent / 'neonarconlp' / 'data' / 'processed' / 'clean_chat_logs.csv'

    def test_synthetic_data_exists(self):
        self.assertTrue(self.raw_path.exists())
        with open(self.raw_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertTrue(len(data) >= 1000)
        self.assertIn('user_id', data[0])

    def test_clean_data_exists(self):
        self.assertTrue(self.clean_path.exists())
        df = pd.read_csv(self.clean_path)
        self.assertFalse(df.isnull().values.any())
        self.assertIn('message', df.columns)

if __name__ == '__main__':
    unittest.main() 