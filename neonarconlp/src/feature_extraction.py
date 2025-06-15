import pandas as pd
import re
import spacy
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import numpy as np

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt', quiet=True)

DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
FEATURES_PATH = DATA_DIR / 'features.csv'
CLEAN_LOGS_PATH = DATA_DIR / 'clean_chat_logs.csv'

analyzer = SentimentIntensityAnalyzer()

# Helper functions for advanced features

def detect_deception(text):
    # Simple cues: hedging, over-specificity, negations
    hedges = ["maybe", "perhaps", "possibly", "I think", "I guess", "sort of", "kind of"]
    overspecific = ["exactly", "precisely", "specifically", "absolutely"]
    count = sum(1 for h in hedges if h in text) + sum(1 for o in overspecific if o in text)
    return int(count > 0)

def emotional_volatility(sentiments):
    # Use rolling window to detect rapid changes (simulate for single message)
    return np.random.uniform(0, 1)

def detect_intent(text):
    # Look for planning, negotiation, threat verbs
    intent_verbs = ["plan", "arrange", "organize", "threaten", "negotiate", "meet", "deliver", "move"]
    return int(any(verb in text for verb in intent_verbs))

def power_dynamics(text):
    # Imperative verbs, politeness markers
    doc = nlp(text)
    imperatives = sum(1 for token in doc if token.tag_ == "VB" and token.dep_ == "ROOT")
    polite = sum(1 for word in ["please", "thank you", "could you"] if word in text)
    if imperatives > polite:
        return "dominant"
    elif polite > imperatives:
        return "submissive"
    else:
        return "neutral"

def trust_meter(text):
    # Frequent reassurances, suspicion terms
    trust_words = ["trust", "promise", "sure", "guarantee"]
    distrust_words = ["doubt", "suspicious", "not sure", "uncertain"]
    trust = sum(1 for w in trust_words if w in text)
    distrust = sum(1 for w in distrust_words if w in text)
    return trust - distrust

def risk_profile(text):
    # Bold vs. cautious phrasing
    bold = ["no risk", "all in", "let's go", "move fast"]
    cautious = ["careful", "watch out", "be sure", "double check"]
    if any(b in text for b in bold):
        return "bold"
    elif any(c in text for c in cautious):
        return "cautious"
    else:
        return "neutral"

def cognitive_bias(text):
    # Overconfidence, confirmation bias
    overconf = ["definitely", "certainly", "no doubt"]
    confirm = ["I knew it", "just as expected", "as always"]
    if any(o in text for o in overconf):
        return "overconfidence"
    elif any(c in text for c in confirm):
        return "confirmation"
    else:
        return "none"

def stress_signal(text):
    # Fragmented sentences, high adverb use
    adverbs = [w for w in text.split() if w.endswith("ly")]
    fragments = text.count("...") + text.count("--")
    return min(1.0, 0.2 * len(adverbs) + 0.5 * fragments)

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
        # Advanced features
        psychelingo = int('nervous' in message or 'stress' in message or 'afraid' in message)
        deceptdetect = detect_deception(message)
        emovolatility = emotional_volatility([sentiment])
        intenttrace = detect_intent(message)
        powerdyno = power_dynamics(message)
        trustmeter = trust_meter(message)
        riskprofile = risk_profile(message)
        cogbiasscan = cognitive_bias(message)
        stresssignal = stress_signal(message)
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
            'stresssignal': stresssignal,
            'role': row['role']
        })
    return pd.DataFrame(features)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def main():
    try:
        df = pd.read_csv(CLEAN_LOGS_PATH)
        features_df = extract_features(df)
        features_df.to_csv(FEATURES_PATH, index=False)
        print("Feature extraction completed.")
    except Exception as e:
        print(f"Error in feature extraction: {e}")

if __name__ == "__main__":
    main() 