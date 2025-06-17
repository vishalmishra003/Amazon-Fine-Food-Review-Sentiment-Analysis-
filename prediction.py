import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """
    Clean and preprocess review text.
    Args:
        text (str): Raw review text.
    Returns:
        str: Cleaned and lemmatized text.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|[^\w\s]|\d', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    cleaned = ' '.join(tokens)
    return cleaned if cleaned.strip() else ''

def get_sentiment_label(compound_score):
    """
    Convert VADER compound score to sentiment label.
    Args:
        compound_score (float): VADER compound score.
    Returns:
        str: Sentiment label.
    """
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def predict_review_score_and_sentiment(review_text):
    """
    Predict review score and sentiment for a given text, with VADER adjustment for the score.
    Args:
        review_text (str): Raw review text.
    Returns:
        tuple: (score, sentiment) where score is the predicted star rating (1-5) and sentiment is the label ("Positive", "Negative", "Neutral").
    """
    if not review_text or not review_text.strip():
        raise ValueError("Review text cannot be empty.")
    # Load pre-trained model and vectorizer
    model = joblib.load('models/review_score_model.pkl')  # Loads the Voting Classifier Ensemble model
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    # Clean and transform the input text
    cleaned_review = clean_text(review_text)
    review_tfidf = vectorizer.transform([cleaned_review])
    # Predict score using the model
    predicted_score = model.predict(review_tfidf)
    predicted_score = label_encoder.inverse_transform(predicted_score)[0]
    predicted_score = int(predicted_score) + 1  # Convert from 0-4 to 1-5
    # Adjust prediction using VADER sentiment
    sentiment_scores = sia.polarity_scores(review_text)
    compound = sentiment_scores['compound']
    sentiment = get_sentiment_label(compound)
    if compound >= 0.5:
        vader_score = 4
    elif compound >= 0.2:
        vader_score = 3
    elif compound > -0.2:
        vader_score = 2
    elif compound > -0.5:
        vader_score = 1
    else:
        vader_score = 0
    # Adjust prediction if VADER score differs significantly
    if abs(predicted_score - vader_score) > 1:
        predicted_score = (predicted_score + vader_score) // 2
    return predicted_score, sentiment

if __name__ == "__main__":
    print("Welcome to the Amazon Review Sentiment Predictor!")
    print("Enter a review to predict its score and sentiment.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            # Prompt user for input
            review = input("Enter your review: ").strip()
            
            # Check if user wants to exit
            if review.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break
            
            # Predict score and sentiment
            score, sentiment = predict_review_score_and_sentiment(review)
            
            # Display results
            print(f"\nPredicted Score: {score} stars")
            print(f"Sentiment: {sentiment}\n")
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please enter a valid review.\n")
        except Exception as e:
            print(f"Error: {e}")
            print("An unexpected error occurred. Please try again.\n")
        except KeyboardInterrupt:
            print("\nExiting the program. Goodbye!")
            break