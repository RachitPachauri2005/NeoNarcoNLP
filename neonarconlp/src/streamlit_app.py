import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import io
import base64
import datetime
import pathlib
import zipfile
import tempfile
import os
import json
import numpy as np

nltk.download('punkt', quiet=True)

MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'rf_model.pkl'
analyzer = SentimentIntensityAnalyzer()

NON_NUMERIC = ['user_id', 'ngram_1', 'ngram_2', 'emotion', 'powerdyno', 'riskprofile', 'cogbiasscan']

NUMERIC_FEATURES = [
    'sentiment', 'sentence_length', 'punctuation_count', 'vocab_richness',
    'psychelingo', 'deceptdetect', 'emovolatility', 'intenttrace', 'trustmeter', 'stresssignal'
]
CATEGORICAL_FEATURES = [
    'powerdyno', 'riskprofile', 'cogbiasscan'
]
TEXT_FEATURES = [
    'ngram_1', 'ngram_2', 'emotion'
]

# --- Logo path ---
logo_path = 'neonarconlp/assets/logo.png'

# --- Sidebar with logo, info, and links ---
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;'>
""", unsafe_allow_html=True)
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown("<span style='font-size:3em;'>🕵️‍♂️</span>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style='color:#1976d2; margin-bottom:0;'>NeoNarcoNLP</h2>
        <span style='color:#555;'>Criminal Role Profiler</span>
    </div>
    <hr style='margin:18px 0;'>
    <b>🔍 About:</b><br>
    <span style='font-size:0.98em;'>Analyze encrypted chat messages for criminal role and mindset using advanced psycholinguistics and stylometry.</span>
    <hr style='margin:18px 0;'>
    <b>⚡ Quick Links:</b><br>
    <a href='https://github.com/RachitPachauri2005' target='_blank'>GitHub Repo</a><br>
    <a href='https://en.wikipedia.org/wiki/Psycholinguistics' target='_blank'>Psycholinguistics</a>
    <hr style='margin:18px 0;'>
    <span style='font-size:0.9em; color:#888;'>© 2025 NeoNarcoNLP Team</span>
    """, unsafe_allow_html=True)
    show_docs = st.button('📄 View Full Documentation')

# After sidebar and before main app logic
if 'show_docs' not in st.session_state:
    st.session_state['show_docs'] = False
if 'show_docs' in locals() and show_docs:
    st.session_state['show_docs'] = not st.session_state['show_docs']
if st.session_state['show_docs']:
    readme_path = pathlib.Path('README.md')
    with st.expander('📄 Full Project Documentation', expanded=True):
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                st.markdown(f.read())
        else:
            st.warning('Documentation file not found.')

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(messages, user_ids=None):
    features = []
    # Expanded keyword lists with more context-specific terms
    psychelingo_words = [
        "nervous", "stress", "afraid", "anxious", "worried", "scared", "panic", "fear", "shaky", "tense", "uneasy", "apprehensive", "restless", "distressed", "concerned", "overwhelmed", "pressured", "edgy", "fidgety", "dread", "unsettled",
        "paranoid", "suspicious", "careful", "watch", "check", "verify", "confirm", "double check", "make sure", "be careful", "be safe", "be quiet", "keep low", "stay hidden", "avoid detection"
    ]
    deceptdetect_words = [
        "maybe", "perhaps", "possibly", "i think", "i guess", "sort of", "kind of", "exactly", "precisely", "specifically", "absolutely", "honestly", "swear", "promise", "trust me", "believe me", "to be honest", "truth is", "as far as i know", "i swear", "i assure", "believe me",
        "no problem", "everything's fine", "nothing to worry", "don't worry", "it's safe", "trust me", "i know what i'm doing", "i've done this before", "it's routine", "standard procedure"
    ]
    intenttrace_words = [
        "plan", "arrange", "organize", "threaten", "negotiate", "meet", "deliver", "move", "prepare", "ready", "target", "goal", "objective", "deal", "exchange", "threat", "warn", "intend", "aim", "purpose", "schedule", "set up", "coordinate", "plot", "scheme", "mission", "assignment",
        "shipment", "package", "delivery", "pickup", "drop", "location", "address", "route", "transport", "transfer", "handover", "receive", "collect", "supply", "stock", "inventory", "quantity", "amount", "price", "cost", "payment", "money", "cash", "transfer", "bank", "account"
    ]
    submissive_words = ["please", "could you", "would you", "can you", "if you don't mind", "may i", "would it be possible", "kindly", "help", "assist", "support", "guide", "advise", "suggest", "recommend"]
    dominant_words = ["do this", "now", "immediately", "must", "have to", "need to", "listen", "obey", "order", "right away", "at once", "no excuses", "you will", "i demand", "i insist", "follow", "comply", "execute", "carry out", "implement", "enforce"]
    trust_words = ["trust", "promise", "sure", "guarantee", "believe", "confident", "certain", "assure", "reliable", "dependable", "faith", "loyal", "honest", "truthful", "genuine", "authentic", "verified", "confirmed", "tested", "proven"]
    distrust_words = ["doubt", "suspicious", "not sure", "uncertain", "liar", "unreliable", "untrustworthy", "skeptical", "questionable", "fishy", "shady", "dodgy", "risky", "dangerous", "warning", "caution", "beware", "careful", "watch out"]
    risk_bold = ["no risk", "all in", "let's go", "move fast", "go for it", "take the risk", "high stakes", "nothing to lose", "risk it", "full send", "no fear", "confident", "sure", "guaranteed", "safe", "secure", "protected", "covered"]
    risk_cautious = ["careful", "watch out", "be sure", "double check", "cautious", "slow down", "think twice", "play safe", "risk averse", "safety first", "precaution", "prevent", "avoid", "protect", "secure", "safe", "guarded"]
    cogbias_overconfidence = ["definitely", "certainly", "no doubt", "guaranteed", "for sure", "without question", "undoubtedly", "absolutely", "no way it fails", "100%", "perfect", "flawless", "impossible to fail", "can't go wrong"]
    cogbias_confirmation = ["i knew it", "just as expected", "as always", "see?", "told you", "as predicted", "just like before", "as i thought", "proved right", "confirmed", "verified", "validated", "tested", "proven"]
    
    for i, message in enumerate(messages):
        cleaned = clean_text(message)
        tokens = cleaned.split()
        sentiment = analyzer.polarity_scores(cleaned)['compound']
        ngram_1 = ' '.join(tokens[:1]) if tokens else ''
        ngram_2 = ' '.join(tokens[:2]) if len(tokens) > 1 else ''
        sentence_length = len(tokens)
        punctuation_count = sum(1 for c in message if c in '.,;:!?')
        vocab_richness = len(set(tokens)) / (len(tokens) + 1e-5)
        
        # Enhanced feature detection with context
        message_lower = message.lower()
        
        # Psychelingo with context weighting
        psychelingo_score = sum(2 if word in message_lower else 0 for word in psychelingo_words)
        psychelingo = min(1, psychelingo_score / 5)  # Normalize to 0-1
        
        # DeceptDetect with context weighting
        decept_score = sum(2 if word in message_lower else 0 for word in deceptdetect_words)
        deceptdetect = min(1, decept_score / 5)  # Normalize to 0-1
        
        # EmoVolatility with enhanced calculation
        emovolatility = float(abs(sentiment) * (1 + psychelingo))  # Amplify by psychelingo presence
        
        # IntentTrace with context weighting
        intent_score = sum(2 if word in message_lower else 0 for word in intenttrace_words)
        intenttrace = min(1, intent_score / 5)  # Normalize to 0-1
        
        # Power dynamics with context
        power_score = 0
        if any(phrase in message_lower for phrase in submissive_words):
            power_score = -1
        elif any(phrase in message_lower for phrase in dominant_words):
            power_score = 1
        powerdyno = 'submissive' if power_score < 0 else 'dominant' if power_score > 0 else 'neutral'
        
        # Trust meter with enhanced scoring
        trust_score = sum(2 if w in message_lower else 0 for w in trust_words) - sum(2 if w in message_lower else 0 for w in distrust_words)
        trustmeter = max(-1, min(1, trust_score / 5))  # Normalize to -1 to 1
        
        # Risk profile with context
        risk_score = 0
        if any(b in message_lower for b in risk_bold):
            risk_score = 1
        elif any(c in message_lower for c in risk_cautious):
            risk_score = -1
        riskprofile = 'bold' if risk_score > 0 else 'cautious' if risk_score < 0 else 'neutral'
        
        # Cognitive bias with context
        bias_score = 0
        if any(o in message_lower for o in cogbias_overconfidence):
            bias_score = 1
        elif any(c in message_lower for c in cogbias_confirmation):
            bias_score = -1
        cogbiasscan = 'overconfidence' if bias_score > 0 else 'confirmation' if bias_score < 0 else 'none'
        
        # Stress signal with enhanced detection
        adverbs = [w for w in message.split() if w.endswith("ly")]
        fragments = message.count("...") + message.count("--")
        exclamations = message.count("!")
        questions = message.count("?")
        stresssignal = float(min(1.0, 0.2 * len(adverbs) + 0.3 * fragments + 0.2 * exclamations + 0.1 * questions))
        
        # Emotion detection with context
        emotion = ''
        if sentiment > 0.5:
            emotion = 'positive'
        elif sentiment < -0.5:
            emotion = 'negative'
        else:
            if stresssignal > 0.5:
                emotion = 'stressed'
            elif intenttrace > 0.5:
                emotion = 'focused'
            else:
                emotion = 'neutral'
        
        features.append({
            'user_id': str(user_ids[i] if user_ids is not None else f'user_{i+1:04d}'),
            'sentiment': float(sentiment),
            'emotion': str(emotion),
            'ngram_1': str(ngram_1),
            'ngram_2': str(ngram_2),
            'sentence_length': int(sentence_length),
            'punctuation_count': int(punctuation_count),
            'vocab_richness': float(vocab_richness),
            'psychelingo': float(psychelingo),
            'deceptdetect': float(deceptdetect),
            'emovolatility': float(emovolatility),
            'intenttrace': float(intenttrace),
            'powerdyno': str(powerdyno),
            'trustmeter': float(trustmeter),
            'riskprofile': str(riskprofile),
            'cogbiasscan': str(cogbiasscan),
            'stresssignal': float(stresssignal)
        })
    
    # Create DataFrame with explicit dtypes
    df = pd.DataFrame(features)
    
    # Convert numeric columns to appropriate types
    numeric_cols = ['sentiment', 'sentence_length', 'punctuation_count', 'vocab_richness', 
                   'psychelingo', 'deceptdetect', 'emovolatility', 'intenttrace', 
                   'trustmeter', 'stresssignal']
    
    # Ensure numeric columns are properly converted
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure string columns are properly formatted
    string_cols = ['user_id', 'emotion', 'ngram_1', 'ngram_2', 'powerdyno', 'riskprofile', 'cogbiasscan']
    for col in string_cols:
        df[col] = df[col].astype(str).fillna('')
    
    return df

def predict_roles(messages, user_ids=None):
    try:
        model = joblib.load(MODEL_PATH)
        features_df = extract_features(messages, user_ids)
        # Drop non-numeric columns for prediction
        numeric_cols = [col for col in features_df.columns if col not in NON_NUMERIC]
        X = features_df[numeric_cols]
        
        # Make predictions
        preds = model.predict(X)
        proba = model.predict_proba(X)
        
        # Ensure we have the correct classes
        if not hasattr(model, 'classes_'):
            model.classes_ = ['supplier', 'smuggler', 'middleman']
        
        # Calculate mindset percentage
        mindset_pct = proba.max(axis=1) * 100
        
        # Print debug information
        print(f"Model classes: {model.classes_}")
        print(f"Predictions: {preds}")
        print(f"Probabilities shape: {proba.shape}")
        
        return preds, proba, model.classes_, features_df, mindset_pct
    except Exception as e:
        print(f"Error in predict_roles: {str(e)}")
        # Return default values in case of error
        default_preds = ['unknown'] * len(messages)
        default_proba = np.zeros((len(messages), 3))
        default_classes = ['supplier', 'smuggler', 'middleman']
        return default_preds, default_proba, default_classes, features_df, np.zeros(len(messages))

st.title("🕵️‍♂️ NeoNarcoNLP: Criminal Role Prediction")
st.write("Type a chat message, upload a CSV (user_id, message), or upload a PDF to predict criminal mindset.")

mode = st.radio("Choose input mode:", ["Single Message", "Batch (CSV Upload)", "PDF Upload"], horizontal=True)

if mode == "Single Message":
    # --- New Chat button and persistent chat state ---
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []  # Each entry: {'msg': str, 'time': datetime, 'role': str, 'mindset': float}
    if st.button('🆕 New Chat'):
        st.session_state['chat_history'] = []
        st.session_state['predicted'] = False
        st.session_state['messages'] = []
        st.session_state['preds'] = []
        st.session_state['proba'] = []
        st.session_state['classes'] = []
        st.session_state['features_df'] = None
        st.session_state['mindset_pct'] = []
        st.session_state['current_idx'] = 0
    # --- Chat input and send button ---
    new_message = st.text_input("Type your message and press Enter", key="chat_input")
    if st.button("Send"):
        if new_message.strip():
            now = datetime.datetime.now().strftime('%H:%M:%S')
            st.session_state['chat_history'].append({'msg': new_message.strip(), 'time': now})
            st.session_state['predicted'] = False
            st.session_state['current_idx'] = len(st.session_state['chat_history']) - 1
    # --- Delete message buttons ---
    if st.session_state['chat_history']:
        for i in range(len(st.session_state['chat_history'])):
            if st.button(f"❌ Delete Message {i+1}"):
                st.session_state['chat_history'].pop(i)
                st.rerun()
                break
    # --- Previous/Next navigation for chat messages ---
    if st.session_state['chat_history']:
        idx = st.session_state.get('current_idx', len(st.session_state['chat_history']) - 1)
        col_prev, col_next = st.columns([1,1])
        with col_prev:
            st.button('⏮️ Previous', key='prev_msg', disabled=(idx == 0), help='Go to previous message')
            if st.session_state['prev_msg'] and idx > 0:
                st.session_state['current_idx'] = idx - 1
                st.rerun()
        with col_next:
            st.button('⏭️ Next', key='next_msg', disabled=(idx == len(st.session_state['chat_history']) - 1), help='Go to next message')
            if st.session_state['next_msg'] and idx < len(st.session_state['chat_history']) - 1:
                st.session_state['current_idx'] = idx + 1
                st.rerun()
    # --- Show only the selected message bubble ---
    if st.session_state['chat_history']:
        idx = st.session_state.get('current_idx', len(st.session_state['chat_history']) - 1)
        entry = st.session_state['chat_history'][idx]
        role_color = {'supplier': '#e0f7fa', 'smuggler': '#ffe0b2', 'middleman': '#c8e6c9'}
        role_icon = {'supplier': '🚚', 'smuggler': '🕶️', 'middleman': '🤝'}
        role = entry.get('role', '?')
        color = role_color.get(role, '#f1f0f0')
        icon = role_icon.get(role, '❓')
        user_avatar = '🧑'
        ai_avatar = icon
        mindset = entry.get('mindset', None)
        mindset_str = f"{mindset:.2f}%" if mindset is not None else "—"
        st.markdown(f"""
        <div style='display:flex; align-items:flex-end; margin-bottom:10px;'>
            <div style='margin-right:10px;'>{user_avatar}</div>
            <div class='fade-in-bubble' style='background-color:{color}; border-radius:12px; padding:18px; min-width:120px; max-width:70%; box-shadow:0 2px 8px #ddd;'>
                <span style='font-size:1.1em; color:#616161;'><b>💬 Message {idx+1} ({entry['time']}):</b></span><br>
                <span style='font-size:1.2em; color:#212121;'>{entry['msg']}</span><br><br>
            </div>
            <div style='margin-left:10px;'>{ai_avatar}</div>
            <div style='font-size:1.1em; color:#1976d2; margin-left:8px;'><b>{role}</b></div>
            <div style='font-size:1.1em; color:#d32f2f; margin-left:8px;'><b>{mindset_str}</b></div>
        </div>
        """, unsafe_allow_html=True)
    # --- Context-aware prediction ---
    N = 3  # Number of previous messages as context
    chat_texts = []
    for i, entry in enumerate(st.session_state['chat_history']):
        context = ' '.join([e['msg'] for e in st.session_state['chat_history'][max(0, i-N):i+1]])
        chat_texts.append(context)
    if chat_texts:
        preds, proba, classes, features_df, mindset_pct = predict_roles(chat_texts)
        # Update chat history with predictions
        for i, entry in enumerate(st.session_state['chat_history']):
            entry['role'] = preds[i]
            entry['mindset'] = mindset_pct[i]
        st.session_state['preds'] = preds
        st.session_state['proba'] = proba
        st.session_state['classes'] = classes
        st.session_state['features_df'] = features_df
        st.session_state['mindset_pct'] = mindset_pct
        # --- Chat export as CSV ---
        if st.button('⬇️ Export Chat as CSV'):
            df_export = pd.DataFrame(st.session_state['chat_history'])
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', data=csv, file_name='chat_history.csv', mime='text/csv')
        # --- Summary panel ---
        st.markdown('---')
        st.subheader('Chat Summary')
        roles = [entry['role'] for entry in st.session_state['chat_history'] if 'role' in entry]
        if roles:
            most_common_role = max(set(roles), key=roles.count)
            avg_mindset = sum(entry['mindset'] for entry in st.session_state['chat_history']) / len(st.session_state['chat_history'])
            st.write(f"Most common role: **{most_common_role}**")
            st.write(f"Average mindset: **{avg_mindset:.2f}%**")
            st.write(f"Total messages: **{len(st.session_state['chat_history'])}**")
        # --- Scroll to latest (auto-scroll) ---
        st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)
        st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)
        # --- Features and probabilities for last message ---
        st.markdown("<b>Probabilities (last message):</b>", unsafe_allow_html=True)
        prob_dict = {role: float(prob) for role, prob in zip(classes, proba[-1])}
        st.bar_chart(prob_dict)
        st.markdown("<b>All Features (last message):</b>", unsafe_allow_html=True)
        st.dataframe(features_df.iloc[[-1]].T, use_container_width=True)
        st.markdown("<hr style='margin:30px 0;'>", unsafe_allow_html=True)
        # Display all features for the last message
        st.markdown("<b>Detailed Analysis of Last Message:</b>", unsafe_allow_html=True)
        last_features = features_df.iloc[-1]
        st.write("### Message Features")
        st.dataframe(last_features.to_frame().T, use_container_width=True)
        
        # Display feature explanations
        st.markdown("### Feature Explanations")
        st.write("""
        - **Sentiment**: Overall emotional tone (-1 to 1)
        - **Emotion**: Detected emotional state
        - **Sentence Length**: Number of words
        - **Vocab Richness**: Unique words ratio
        - **Psychelingo**: Psychological language markers
        - **DeceptDetect**: Deception indicators
        - **EmoVolatility**: Emotional intensity
        - **IntentTrace**: Action/intent indicators
        - **PowerDyno**: Power dynamics in message
        - **TrustMeter**: Trust/distrust indicators
        - **RiskProfile**: Risk-taking behavior
        - **CogBiasScan**: Cognitive bias detection
        - **StressSignal**: Stress indicators
        """)
elif mode == "Batch (CSV Upload)":
    st.write("Upload a CSV, Excel, TSV, TXT, JSON, ODS, or ZIP file with columns: user_id, message")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "tsv", "txt", "json", "ods", "zip"])
    dfs = []
    def read_table(file, name):
        if name.endswith('.csv'):
            return pd.read_csv(file)
        elif name.endswith('.tsv'):
            return pd.read_csv(file, sep='\t')
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        elif name.endswith('.ods'):
            import pyexcel_ods
            data = pyexcel_ods.get_data(file)
            sheet = next(iter(data.values()))
            df = pd.DataFrame(sheet[1:], columns=sheet[0])
            return df
        elif name.endswith('.txt'):
            lines = file.read().decode('utf-8').splitlines()
            return pd.DataFrame({'user_id': [f'user_{i+1:04d}' for i in range(len(lines))], 'message': lines})
        elif name.endswith('.json'):
            data = json.load(file)
            return pd.DataFrame(data)
        else:
            return None
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(uploaded_file, 'r') as z:
                    z.extractall(tmpdir)
                    for fname in z.namelist():
                        fpath = os.path.join(tmpdir, fname)
                        try:
                            with open(fpath, 'rb') as f:
                                df = read_table(f, fname)
                                if df is not None:
                                    dfs.append(df)
                        except Exception as e:
                            st.warning(f"Could not read {fname}: {e}")
        else:
            df = read_table(uploaded_file, uploaded_file.name)
            if df is not None:
                dfs.append(df)
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            # --- Column mapping ---
            st.write("### Data Preview (first 10 rows)")
            st.dataframe(df.head(10))
            columns = df.columns.tolist()
            user_col = st.selectbox("Select user_id column", columns, index=columns.index('user_id') if 'user_id' in columns else 0)
            msg_col = st.selectbox("Select message column", columns, index=columns.index('message') if 'message' in columns else 1)
            if st.button("Confirm and Predict"):
                if user_col not in df.columns or msg_col not in df.columns:
                    st.error("Selected columns not found in file.")
                else:
                    preds, proba, classes, features_df, mindset_pct = predict_roles(df[msg_col].astype(str).tolist(), df[user_col].astype(str).tolist())
                    # Results table
                    results = df.copy()
                    results['Predicted Role'] = preds
                    for i, role in enumerate(classes):
                        results[f'Prob_{role}'] = proba[:, i]
                    results['Criminal Mindset %'] = mindset_pct
                    st.write("### Prediction Results")
                    st.dataframe(results)
                    # Show all features for all messages in a table
                    st.write("### All Features Table (Batch)")
                    st.dataframe(features_df)
                    # Per-user aggregation
                    st.write("### Per-User Criminal Mindset Summary")
                    user_summary = results.groupby(user_col)['Criminal Mindset %'].agg(['mean', 'max', 'count']).reset_index()
                    user_summary.columns = [user_col, 'Avg Mindset %', 'Max Mindset %', 'Message Count']
                    st.dataframe(user_summary)
                    # Per-user chat-style visualization
                    st.write("### Per-User Chat History with Roles")
                    user_list = results[user_col].unique().tolist()
                    selected_user = st.selectbox("Select a user to view chat history:", user_list)
                    user_msgs = results[results[user_col] == selected_user].reset_index(drop=True)
                    for idx, row in user_msgs.iterrows():
                        st.markdown(f"<div style='background-color:#f1f0f0; border-radius:10px; padding:10px; margin-bottom:5px;'><b>Message:</b> {row[msg_col]}<br><b>Role:</b> {row['Predicted Role']}<br><b>Mindset %:</b> {row['Criminal Mindset %']:.2f}</div>", unsafe_allow_html=True)
                    # Downloadable report for selected user
                    st.write("#### Download User Report")
                    user_report = user_msgs.copy()
                    csv = user_report.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV for this user",
                        data=csv,
                        file_name=f'user_{selected_user}_report.csv',
                        mime='text/csv'
                    )
        else:
            st.error("No valid data found in uploaded file(s). Supported formats: CSV, Excel, TSV, TXT, JSON, ODS, ZIP.")
else:
    st.write("Upload a PDF file. Each page will be treated as a separate message.")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_pdf is not None:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_pdf.read()))
        messages = []
        user_ids = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            # Split by paragraphs for more granularity
            for j, para in enumerate(text.split('\n')):
                para = para.strip()
                if para:
                    messages.append(para)
                    user_ids.append(f"pdf_page{i+1}_para{j+1}")
        if messages:
            preds, proba, classes, features_df, mindset_pct = predict_roles(messages, user_ids)
            results = pd.DataFrame({
                'user_id': user_ids,
                'message': messages,
                'Predicted Role': preds,
                'Criminal Mindset %': mindset_pct
            })
            for i, role in enumerate(classes):
                results[f'Prob_{role}'] = proba[:, i]
            st.write("### PDF Prediction Results")
            st.dataframe(results)
            # Show all features for all PDF messages in a table
            st.write("### All Features Table (PDF)")
            st.dataframe(features_df)
            user_summary = results.groupby('user_id')['Criminal Mindset %'].agg(['mean', 'max', 'count']).reset_index()
            user_summary.columns = ['user_id', 'Avg Mindset %', 'Max Mindset %', 'Message Count']
            st.write("### Per-User (Page/Para) Criminal Mindset Summary")
            st.dataframe(user_summary)
        else:
            st.warning("No text found in PDF.")

# --- Footer with credits, ethical guidelines, and NO developer names ---
st.markdown("""
<hr style='margin:40px 0 10px 0;'>
<div style='text-align:center; color:#888; font-size:0.95em;'>
    Made with <span style='color:#d32f2f;'>❤</span> by the NeoNarcoNLP Team<br>
</div>
""", unsafe_allow_html=True)
with st.expander('📜 Full Ethical Guidelines', expanded=False):
    st.markdown('''
**Ethical Guidelines**

This tool is intended strictly for research and responsible use in criminal psychology and law enforcement. Misuse for surveillance, discrimination, or violation of privacy is strictly prohibited. All users must ensure compliance with local laws and ethical standards. Data processed should be anonymized and handled with care to protect individual privacy and rights.
''') 