from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
import logging
from prediction import predict_review_score_and_sentiment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and vectorizer
try:
    helpfulness_model = joblib.load('models/helpfulness_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
except FileNotFoundError as e:
    logging.error(f"Error loading models: {e}")
    exit()

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text.
    Args:
        text (str): Raw text input.
    Returns:
        str: Cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|[^\w\s]|\d', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_helpfulness(review_text):
    """
    Predict helpfulness of a review.
    Args:
        review_text (str): Raw review text.
    Returns:
        str: Helpfulness label.
    """
    cleaned_review = clean_text(review_text)
    review_tfidf = vectorizer.transform([cleaned_review])
    predicted_helpfulness = helpfulness_model.predict(review_tfidf)
    return "Helpful" if predicted_helpfulness[0] == 1 else "Not Helpful"

def get_top_features(review_text):
    """
    Get top 3 influential words with non-zero weights.
    Args:
        review_text (str): Raw review text.
    Returns:
        list: Top features and their weights.
    """
    cleaned_review = clean_text(review_text)
    review_tfidf = vectorizer.transform([cleaned_review]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    top_indices = review_tfidf.argsort()[-3:][::-1]
    top_features = [(feature_names[i], round(review_tfidf[i], 3)) for i in top_indices if review_tfidf[i] > 0]
    return top_features if top_features else [("No significant features", 0.0)]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    helpfulness = None
    sentiment = None
    top_features = None
    review_text = ""
    feedback_message = None
    if request.method == 'POST':
        review_text = request.form.get('review')
        feedback = request.form.get('feedback')
        if feedback and feedback.strip():
            logging.info(f"User feedback: {feedback}")
            feedback_message = "Feedback submitted! Thank you!"
        if review_text and review_text.strip():
            try:
                # Predict review score and sentiment
                prediction, sentiment = predict_review_score_and_sentiment(review_text)
                # Predict helpfulness
                helpfulness = predict_helpfulness(review_text)
                # Get top influential words
                top_features = get_top_features(review_text)
                logging.info(f"Prediction: {prediction}, Helpfulness: {helpfulness}, Sentiment: {sentiment}")
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                prediction = "Error predicting score. Please try again."
        else:
            prediction = "Please enter a valid review."
    return render_template('index.html', prediction=prediction, helpfulness=helpfulness,
                          sentiment=sentiment, top_features=top_features,
                          review_text=review_text, feedback_message=feedback_message)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)