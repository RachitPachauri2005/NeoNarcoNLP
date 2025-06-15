import json
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st

ROLES = ['supplier', 'smuggler', 'middleman']
CHAT_GROUPS = [f'group_{i}' for i in range(1, 11)]
USER_IDS = [f'user_{i:04d}' for i in range(1, 101)]

SYNTHETIC_MESSAGES = [
    "Meet at the usual spot.",
    "Package is ready for pickup.",
    "Let me know when you arrive.",
    "The shipment is delayed.",
    "Payment received.",
    "Change of plans, update soon.",
    "Keep it quiet.",
    "Use the new route.",
    "Contact the middleman first.",
    "Everything is set.",
    # Add some with psychological markers
    "Honestly, I don't know if this will work...",
    "I swear, you can trust me.",
    "If you get caught, say nothing.",
    "I promise, this is the last time.",
    "We need to move fast, no time to think.",
    "Are you sure this is safe?",
    "Don't tell anyone else, okay?",
    "Just follow my lead, don't ask questions.",
    "I feel nervous about this run.",
    "Everything is under control, trust me."
]

# Psychological marker simulation functions
PSYCH_MARKERS = {
    'psychelingo': lambda: random.choice([0, 1]),  # 1: marker present
    'deceptdetect': lambda: random.choice([0, 1]),
    'emovolatility': lambda: random.uniform(0, 1),
    'intenttrace': lambda: random.choice([0, 1]),
    'powerdyno': lambda: random.choice(['dominant', 'submissive', 'neutral']),
    'trustmeter': lambda: random.uniform(-1, 1),  # -1: distrust, 1: trust
    'riskprofile': lambda: random.choice(['bold', 'cautious']),
    'cogbiasscan': lambda: random.choice(['overconfidence', 'confirmation', 'none']),
    'stresssignal': lambda: random.uniform(0, 1)
}

DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_chat_logs(num_entries=1000, seed=42):
    """Generate a list of synthetic chat log entries with psychological markers."""
    random.seed(seed)
    start_time = datetime.now() - timedelta(days=30)
    logs = []
    for i in range(num_entries):
        user_id = random.choice(USER_IDS)
        message = random.choice(SYNTHETIC_MESSAGES)
        timestamp = (start_time + timedelta(minutes=random.randint(0, 43200))).isoformat()
        chat_group = random.choice(CHAT_GROUPS)
        role = random.choice(ROLES)
        entry = {
            "user_id": user_id,
            "message": message,
            "timestamp": timestamp,
            "chat_group": chat_group,
            "role": role
        }
        # Add psychological markers
        for marker, func in PSYCH_MARKERS.items():
            entry[marker] = func()
        logs.append(entry)
    return logs


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def preprocess_logs(logs):
    """Remove nulls and normalize text."""
    df = pd.DataFrame(logs)
    df = df.dropna()
    df['message'] = df['message'].str.lower().str.strip()
    return df


def main():
    try:
        logs = generate_synthetic_chat_logs()
        save_json(logs, RAW_DIR / 'synthetic_chat_logs.json')
        df_clean = preprocess_logs(logs)
        df_clean.to_csv(PROCESSED_DIR / 'clean_chat_logs.csv', index=False)
        print("Synthetic data generated and preprocessed successfully.")
    except Exception as e:
        print(f"Error in data preparation: {e}")


if __name__ == "__main__":
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    new_message = st.text_input("Type your message and press Enter", key="chat_input")
    if st.button("Send"):
        if new_message.strip():
            st.session_state['chat_history'].append(new_message.strip())
            st.session_state['predicted'] = False  # Reset prediction state

    if st.session_state['chat_history']:
        preds, proba, classes, features_df, mindset_pct = predict_roles(st.session_state['chat_history'])
        # Display chat bubbles for each message in history
        for i, msg in enumerate(st.session_state['chat_history']):
            # ... render chat bubble for msg, preds[i], mindset_pct[i], etc.

    main() 