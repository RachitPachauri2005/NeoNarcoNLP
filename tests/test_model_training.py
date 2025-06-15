import unittest
from pathlib import Path

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.model_path = Path(__file__).parent.parent / 'neonarconlp' / 'outputs' / 'rf_model.pkl'
        self.report_path = Path(__file__).parent.parent / 'neonarconlp' / 'outputs' / 'evaluation_report.txt'

    def test_model_file_exists(self):
        self.assertTrue(self.model_path.exists())

    def test_report_file_exists(self):
        self.assertTrue(self.report_path.exists())
        with open(self.report_path, 'r') as f:
            content = f.read()
        self.assertIn('precision', content)
        self.assertIn('recall', content)

if __name__ == '__main__':
    unittest.main() 