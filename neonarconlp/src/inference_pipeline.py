import pandas as pd
import joblib
from pathlib import Path
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)

DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'rf_model.pkl'
PRED_PATH = Path(__file__).parent.parent / 'outputs' / 'predictions.csv'

analyzer = SentimentIntensityAnalyzer()

NON_NUMERIC = ['user_id', 'ngram_1', 'ngram_2', 'emotion', 'powerdyno', 'riskprofile', 'cogbiasscan']

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(df):
    features = []
    for _, row in df.iterrows():
        message = clean_text(row['message'])
        tokens = nltk.word_tokenize(message)
        sentiment = analyzer.polarity_scores(message)['compound']
        ngram_1 = ' '.join(tokens[:1]) if tokens else ''
        ngram_2 = ' '.join(tokens[:2]) if len(tokens) > 1 else ''
        sentence_length = len(tokens)
        punctuation_count = sum(1 for c in row['message'] if c in '.,;:!?')
        vocab_richness = len(set(tokens)) / (len(tokens) + 1e-5)
        # Advanced features (simulate for inference)
        psychelingo = int('nervous' in message or 'stress' in message or 'afraid' in message)
        deceptdetect = 0
        emovolatility = 0.5
        intenttrace = 0
        powerdyno = 'neutral'
        trustmeter = 0
        riskprofile = 'neutral'
        cogbiasscan = 'none'
        stresssignal = 0.0
        features.append({
            'user_id': row['user_id'],
            'sentiment': sentiment,
            'emotion': '',
            'ngram_1': ngram_1,
            'ngram_2': ngram_2,
            'sentence_length': sentence_length,
            'punctuation_count': punctuation_count,
            'vocab_richness': vocab_richness,
            'psychelingo': psychelingo,
            'deceptdetect': deceptdetect,
            'emovolatility': emovolatility,
            'intenttrace': intenttrace,
            'powerdyno': powerdyno,
            'trustmeter': trustmeter,
            'riskprofile': riskprofile,
            'cogbiasscan': cogbiasscan,
            'stresssignal': stresssignal
        })
    return pd.DataFrame(features)

def main():
    try:
        df = pd.read_csv(DATA_DIR / 'clean_chat_logs.csv')
        features_df = extract_features(df)
        model = joblib.load(MODEL_PATH)
        # Drop non-numeric columns to match model input
        X = features_df.drop([col for col in NON_NUMERIC if col in features_df.columns], axis=1)
        preds = model.predict(X)
        proba = model.predict_proba(X)
        results = features_df[['user_id']].copy()
        results['predicted_role'] = preds
        for i, role in enumerate(model.classes_):
            results[f'prob_{role}'] = proba[:, i]
        results.to_csv(PRED_PATH, index=False)
        print("Inference completed. Predictions saved.")
    except Exception as e:
        print(f"Error in inference pipeline: {e}")

if __name__ == "__main__":
    main() 