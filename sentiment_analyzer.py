
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import nltk
import os
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure NLTK data is available
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
if os.path.exists(nltk_data_path):
    nltk.data.path.append(nltk_data_path)

# Initialize Hugging Face sentiment pipeline
# Using a pre-trained model for sentiment analysis
# This will download the model the first time it's run
try:
    hf_sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Could not load Hugging Face pipeline: {e}")
    hf_sentiment_pipeline = None

def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    confidence = abs(polarity)

    return sentiment, confidence, polarity, subjectivity

def analyze_sentiment_huggingface(text):
    if hf_sentiment_pipeline is None:
        return 'Neutral', 0.0, 0.0, 0.0 # Fallback if pipeline not loaded

    result = hf_sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    # Map Hugging Face labels to our multi-class system
    if label == 'POSITIVE':
        sentiment = 'Positive'
        polarity = score # Use score as polarity for consistency
    elif label == 'NEGATIVE':
        sentiment = 'Negative'
        polarity = -score # Use negative score for negative sentiment
    else:
        sentiment = 'Neutral'
        polarity = 0.0
    
    confidence = score
    subjectivity = 0.0 # Hugging Face pipeline doesn't directly provide subjectivity like TextBlob

    return sentiment, confidence, polarity, subjectivity

def extract_keywords(text, num_keywords=5):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'])
    
    try:
        word_tokens = word_tokenize(text.lower())
    except LookupError:
        word_tokens = text.lower().split()
    
    filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    
    word_counts = Counter(filtered_words)
    
    most_common = word_counts.most_common(num_keywords)
    
    return [word for word, count in most_common]

def batch_analyze_sentiment(texts, analyzer_type='TextBlob'):
    results = []
    for text_item in texts:
        text_content = text_item['text']
        if analyzer_type == 'HuggingFace':
            sentiment, confidence, polarity, subjectivity = analyze_sentiment_huggingface(text_content)
        else:
            sentiment, confidence, polarity, subjectivity = analyze_sentiment_textblob(text_content)
        
        keywords = extract_keywords(text_content)
        results.append({
            'text': text_content,
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'keywords': keywords,
            'source': text_item.get('source', 'N/A'),
            'date': text_item.get('date', 'N/A')
        })
    return pd.DataFrame(results)

def get_sentiment_explanation(text, sentiment, polarity, analyzer_type='TextBlob'):
    if analyzer_type == 'HuggingFace':
        model_name = "Hugging Face (distilbert-base-uncased-finetuned-sst-2-english)"
    else:
        model_name = "TextBlob"

    if sentiment == 'Positive':
        return f"The text '{text}' is classified as Positive by {model_name} due to strong positive indicators. The polarity score is {polarity:.2f}."
    elif sentiment == 'Negative':
        return f"The text '{text}' is classified as Negative by {model_name} due to strong negative indicators. The polarity score is {polarity:.2f}."
    else:
        return f"The text '{text}' is classified as Neutral by {model_name}. The polarity score is {polarity:.2f}. This could be due to a lack of strong emotional language or a balance of positive and negative terms."

def generate_accuracy_report(df_labeled_data, analyzer_type='TextBlob'):
    if df_labeled_data.empty or 'text' not in df_labeled_data.columns or 'true_sentiment' not in df_labeled_data.columns:
        return None, "Labeled data must contain 'text' and 'true_sentiment' columns."

    # Perform sentiment analysis on the labeled data
    predicted_sentiments = []
    for _, row in df_labeled_data.iterrows():
        text_content = row['text']
        if analyzer_type == 'HuggingFace':
            sentiment, _, _, _ = analyze_sentiment_huggingface(text_content)
        else:
            sentiment, _, _, _ = analyze_sentiment_textblob(text_content)
        predicted_sentiments.append(sentiment)
    
    df_labeled_data['predicted_sentiment'] = predicted_sentiments

    # Generate classification report
    true_labels = df_labeled_data['true_sentiment'].tolist()
    predicted_labels = df_labeled_data['predicted_sentiment'].tolist()

    # Ensure all labels are in the set of possible labels (Positive, Negative, Neutral)
    all_labels = sorted(list(set(true_labels + predicted_labels)))

    report = classification_report(true_labels, predicted_labels, labels=all_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)

    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(), # Convert numpy array to list for easier handling
        'accuracy': accuracy,
        'labels': all_labels
    }, None


