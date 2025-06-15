import joblib
import pandas as pd
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

nltk.download('punkt', quiet=True)

MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'rf_model.pkl'

analyzer = SentimentIntensityAnalyzer()

NON_NUMERIC = ['ngram_1', 'ngram_2', 'emotion', 'powerdyno', 'riskprofile', 'cogbiasscan']

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(messages):
    features = []
    for i, message in enumerate(messages):
        cleaned = clean_text(message)
        tokens = nltk.word_tokenize(cleaned)
        sentiment = analyzer.polarity_scores(cleaned)['compound']
        ngram_1 = ' '.join(tokens[:1]) if tokens else ''
        ngram_2 = ' '.join(tokens[:2]) if len(tokens) > 1 else ''
        sentence_length = len(tokens)
        punctuation_count = sum(1 for c in message if c in '.,;:!?')
        vocab_richness = len(set(tokens)) / (len(tokens) + 1e-5)
        # Advanced features (simulate for CLI)
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

def predict_roles(messages):
    model = joblib.load(MODEL_PATH)
    features_df = extract_features(messages)
    X = features_df.drop([col for col in NON_NUMERIC if col in features_df.columns], axis=1)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    for i, msg in enumerate(messages):
        print(f"\nMessage: {msg}")
        print(f"Predicted Role: {preds[i]}")
        print("Probabilities:")
        for j, role in enumerate(model.classes_):
            print(f"  {role}: {proba[i, j]:.4f}")

def main():
    print("NeoNarcoNLP Interactive Inference")
    print("Choose input mode:")
    print("1. Type a message")
    print("2. Provide a CSV file of messages (column: message)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == '1':
        msg = input("Type your message: ").strip()
        if msg:
            predict_roles([msg])
        else:
            print("No message entered.")
    elif choice == '2':
        path = input("Enter path to CSV file: ").strip()
        try:
            df = pd.read_csv(path)
            if 'message' not in df.columns:
                print("CSV must have a 'message' column.")
                return
            predict_roles(df['message'].tolist())
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 