import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
FEATURES_PATH = DATA_DIR / 'features.csv'
MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'rf_model.pkl'
VIS_DIR = Path(__file__).parent.parent / 'outputs' / 'visualizations'

VIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_sentiment_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sentiment'], bins=20, kde=True)
    plt.title('Sentiment Compound Score Distribution')
    plt.xlabel('Sentiment Compound Score')
    plt.ylabel('Frequency')
    plt.savefig(VIS_DIR / 'sentiment_compound_distribution.png')
    plt.close()


def plot_role_probabilities(df, model):
    # Drop non-numeric columns to match model input
    X = df.drop(['user_id', 'role', 'ngram_1', 'ngram_2', 'emotion'], axis=1)
    proba = model.predict_proba(X)
    roles = model.classes_
    avg_proba = np.mean(proba, axis=0)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=roles, y=avg_proba)
    plt.title('Average Role Prediction Probabilities')
    plt.xlabel('Role')
    plt.ylabel('Probability')
    plt.savefig(VIS_DIR / 'role_probabilities.png')
    plt.close()


def main():
    try:
        df = pd.read_csv(FEATURES_PATH)
        model = joblib.load(MODEL_PATH)
        plot_sentiment_distribution(df)
        plot_role_probabilities(df, model)
        print("Visualizations generated.")
    except Exception as e:
        print(f"Error in visualization: {e}")

if __name__ == "__main__":
    main() 