# NeoNarcoNLP

A Natural Language Processing (NLP) system designed to analyze and predict criminal roles in communication patterns.

## Features

- Sentiment Analysis
- Emotion Detection
- Role Prediction
- Behavioral Pattern Analysis
- Trust and Risk Assessment
- Cognitive Bias Detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NeoNarcoNLP.git
cd NeoNarcoNLP
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run neonarconlp/src/streamlit_app.py
```

## Project Structure

```
neonarconlp/
├── src/
│   ├── streamlit_app.py
│   ├── feature_extraction.py
│   ├── inference_pipeline.py
│   └── interactive_inference.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 