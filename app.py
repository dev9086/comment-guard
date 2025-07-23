import os
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer


# NLTK Setup 
nltk_data_path = '/usr/local/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path) 

if not os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
    import nltk_downloader

# Load ML models with relative paths
model = joblib.load("models/spam_toxic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
# Initialize variables
english_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
sentiment_analyzer = SentimentIntensityAnalyzer()
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           "]+", flags=re.UNICODE)

# Load rule-based data with relative paths
def load_rule_data():
    # Load CSV files
    hindi_top = pd.read_csv("rule_data/hindi_top_150_swear_words.csv", header=None)[0].tolist()
    hindi_slurs = pd.read_csv("rule_data/hindi_slurs.csv", header=None)[0].tolist()
    
    # Load text file with regex patterns
    regex_patterns = []
    enhanced_swears = []
    with open("rule_data/enhanced_swears.txt", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("regex:"):
                regex_patterns.append(line.replace("regex:", "").strip())
            else:
                enhanced_swears.append(line)
    
    all_slurs = set(hindi_top + hindi_slurs + enhanced_swears)
    return list(all_slurs), regex_patterns

slurs_list, regex_patterns = load_rule_data()

# Text processing functions
def preprocess_text(text):
    text = emoji_pattern.sub(r'', text)
    return re.sub(r'[^a-zA-Z\s]', '', text).strip().lower()

def normalize_text(text):
    text = text.lower()
    replacements = {
        '@': 'a', '4': 'a', '1': 'i', '!': 'i', '|': 'i',
        '3': 'e', '€': 'e', '0': 'o', '$': 's', '5': 's',
        '7': 't', '+': 't', '&': ' and '
    }
    for pattern, replacement in replacements.items():
        text = text.replace(pattern, replacement)
    
    special_chars = ['*', '.', '_', '-', ',', '`', "'", '"', '?', '%', '(', ')']
    for char in special_chars:
        text = text.replace(char, '')
    
    return re.sub(' +', ' ', text).strip()

def clean_comment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    cleaned_words = [stemmer.stem(x) for x in words if x not in english_stopwords]
    return " ".join(cleaned_words)

# fuzzy matching
def rule_based_filter(text):
    normalized = normalize_text(text)
    compressed = normalized.replace(" ", "")
    
    # 1. Check regex patterns
    for pattern in regex_patterns:
        if re.search(pattern, normalized): 
            return True, f"Toxic pattern: {pattern[:20]}..."
    
    # 2. Check for direct matches
    for slur in slurs_list:
        if slur in normalized or slur.replace(" ", "") in compressed: 
            return True, f"Direct slur match: {slur}"
    
    # 3. Check for repeated slurs
    for word in slurs_list:
        if normalized.count(word) >= 3: 
            return True, "Repeated slur"
    
    # 4. N-gram matches
    tokens = normalized.split()
    ngrams = [' '.join(tokens[i:i+n]) for n in [2,3] for i in range(len(tokens)-n+1)]
    for phrase in ngrams:
        if phrase in slurs_list: 
            return True, f"N-gram slur: {phrase}"
    
    # 5. Mocking laughter
    if re.search(r"(ha){3,}", normalized):
        return True, "Mocking laughter"
    
    # 6. Fuzzy match
    for slur in slurs_list:
        if fuzz.partial_ratio(slur, normalized) > 85: 
            return True, f"Fuzzy match: {slur[:10]}..."
        if fuzz.partial_ratio(slur, compressed) > 90: 
            return True, f"Compressed fuzzy: {slur[:10]}..."
    
    # 7. Excessive caps
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2:
        return True, "Excessive caps"
    
    return False, None

def ml_predict(text):
    cleaned = clean_comment(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector.toarray())[0]
    return {"spam": bool(prediction[0]), "toxic": bool(prediction[1])}

def analyze_comment(text):
    # 1. Rule-based filter
    violation, reason = rule_based_filter(text)
    if violation:
        return {"decision": "DELETE", "reason": reason}

    # 2. ML model detection
    ml_result = ml_predict(text)
    if ml_result["spam"] or ml_result["toxic"]:
        return {"decision": "FLAG", "reason": "ML model detection"}

    # 3. Sentiment analysis 
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound <= -0.4:
        return {"decision": "DELETE", "reason": "Very negative sentiment"}
    elif compound <= -0.2:
        return {"decision": "FLAG", "reason": "Moderately negative sentiment"}

    # 4. If clean
    return {"decision": "APPROVE", "reason": "Clean comment"}


# Streamlit UI
st.title("Comment Guard")
st.write("Analyze comments for toxicity, spam, and negative sentiment")

comment = st.text_area("Enter a comment:")
if st.button("ANALYZE"):
    if not comment.strip():
        st.warning("Please enter a comment to analyze")
    else:
        result = analyze_comment(comment)
        
        st.subheader("Analysis Result:")
        
        if result['decision'] == "DELETE":
            st.error(f"Decision: {result['decision']} ❌")
        elif result['decision'] == "FLAG":
            st.warning(f"Decision: {result['decision']} ⚠️")
        else:
            st.success(f"Decision: {result['decision']} ✅")
        
        
        # Details 
        with st.expander("Technical Details"):
            st.write("Cleaned text:", clean_comment(comment))
            ml_res = ml_predict(comment)
            st.write(f"ML Prediction - Spam: {ml_res['spam']}, Toxic: {ml_res['toxic']}")
            
            sentiment = sentiment_analyzer.polarity_scores(comment)
            st.write(f"Sentiment: Compound={sentiment['compound']:.2f}")
            st.write(f"Negative={sentiment['neg']:.2f}, Neutral={sentiment['neu']:.2f}, Positive={sentiment['pos']:.2f}")
