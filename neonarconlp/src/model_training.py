import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
FEATURES_PATH = DATA_DIR / 'features.csv'
MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'rf_model.pkl'
REPORT_PATH = Path(__file__).parent.parent / 'outputs' / 'evaluation_report.txt'

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NON_NUMERIC = ['user_id', 'role', 'ngram_1', 'ngram_2', 'emotion', 'powerdyno', 'riskprofile', 'cogbiasscan']

def main():
    try:
        df = pd.read_csv(FEATURES_PATH)
        # Drop non-numeric columns
        X = df.drop([col for col in NON_NUMERIC if col in df.columns], axis=1)
        y = df['role']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        with open(REPORT_PATH, 'w') as f:
            f.write(report)
        joblib.dump(clf, MODEL_PATH)
        print("Model training and evaluation completed.")
    except Exception as e:
        print(f"Error in model training: {e}")

if __name__ == "__main__":
    main() 