import logging
import pandas as pd
import argparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import joblib
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def prepare_data(df):
    """
    Prepare data for model training.
    Args:
        df (pd.DataFrame): DataFrame with cleaned_text and Score columns.
    Returns:
        Tuple: Training and testing splits, vectorizer, and label encoder.
    """
    label_encoder = LabelEncoder()
    df_sentiment = df[['cleaned_text', 'Score', 'review_length']].copy()
    df_sentiment['Score'] = label_encoder.fit_transform(df_sentiment['Score'])
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    X = df_sentiment['cleaned_text']
    y = df_sentiment['Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    logging.info(f"Split: {len(X_train)} training, {len(X_test)} testing reviews.")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    logging.info(f"TF-IDF: {X_train_tfidf.shape[1]} features.")
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    smote = SMOTE(random_state=43)
    X_train_tfidf_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
    logging.info("SMOTE: Balanced classes for training.")
    return X_train_tfidf_balanced, X_test_tfidf, y_train_balanced, y_test, vectorizer, label_encoder, X_train, X_test

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis on Amazon Fine Food Reviews')
    parser.add_argument('--data', default='Reviews.csv', help='Path to dataset')
    parser.add_argument('--sample', type=int, default=50000, help='Number of reviews to sample')
    args = parser.parse_args()
    path = args.data
    sample_size = args.sample

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # 4.1 Data Acquisition
    logging.info("Loading dataset...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logging.error(f"Cannot find {path}. Check the file path.")
        exit()

    required_columns = ['Text', 'Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Dataset missing required columns: {required_columns}")
        exit()

    if sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=43)
    logging.info(f"Sampled {len(df)} reviews")

    # 4.2 Text Preprocessing
    df = df.dropna(subset=['Text', 'Score']).drop_duplicates()
    logging.info("Data shape after cleaning: %s", df.shape)
    df['cleaned_text'] = df['Text'].apply(clean_text)
    df['review_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df_helpfulness = df[df['HelpfulnessDenominator'] > 0][['cleaned_text', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'review_length']].copy()
    df_helpfulness['Helpfulness'] = df_helpfulness['HelpfulnessNumerator'] / df_helpfulness['HelpfulnessDenominator']
    df_helpfulness['Helpfulness'] = df_helpfulness['Helpfulness'].apply(lambda x: 1 if x > 0.5 else 0)

    # 4.3 Data Exploration
    df['sentiment'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    logging.info("Generating Review Score Distribution plot...")
    fig = px.histogram(df, x='Score', color='Score', title='Review Score Distribution')
    fig.update_layout(xaxis_title='Score', yaxis_title='Count', showlegend=False)
    fig.write_image('outputs/score_distribution.png')

    logging.info("Generating Word Cloud...")
    all_reviews = ' '.join(df['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
    wordcloud.to_file('outputs/wordcloud.png')

    logging.info("Generating Review Length Distribution plot...")
    fig = px.histogram(df, x='review_length', nbins=50, title='Review Length Distribution')
    fig.update_layout(xaxis_title='Word Count', yaxis_title='Frequency', bargap=0.2, bargroupgap=0.1,
                      xaxis=dict(tickmode='array', tickvals=[i for i in range(0, 475, 25)], ticktext=[str(i) for i in range(0, 475, 25)]))
    fig.write_image('outputs/lengths.png')

    logging.info("Generating Sentiment-Score Correlation plot...")
    plt.figure(figsize=(8, 6))
    sns.stripplot(x='sentiment', y='Score', data=df, jitter=True)
    plt.title('Correlation between Review Score and Sentiment')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Review Score')
    plt.savefig('outputs/sentiment.png')
    plt.close()

    logging.info("Generating Sentiment Distribution per Class plot...")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Score', y='sentiment', data=df)
    plt.title('Sentiment Distribution by Review Score')
    plt.xlabel('Review Score')
    plt.ylabel('Sentiment Score')
    plt.savefig('outputs/sentiment_distribution.png')
    plt.close()

    # 4.4 Model Development
    # Store X_test for misclassification analysis
    X_train_tfidf_balanced, X_test_tfidf, y_train_balanced, y_test, vectorizer, label_encoder, X_train, X_test = prepare_data(df)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=43, n_jobs=-1, class_weight='balanced')
    }
    ensemble = VotingClassifier(estimators=[('lr', models['Logistic Regression']), ('nb', models['Naive Bayes']), ('rf', models['Random Forest'])], voting='soft')
    models['Ensemble'] = ensemble

    # 4.5 Model Assessment
    results = {}
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train_tfidf_balanced, y_train_balanced)
        y_pred = model.predict(X_test_tfidf)
        accuracy = model.score(X_test_tfidf, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        logging.info(f"{name} Report:\n{classification_report(y_test, y_pred)}")
        logging.info(f"{name} Accuracy: {accuracy * 100:.2f}%")
        results[name] = {'accuracy': accuracy, 'report': report}
        macro_f1 = report['macro avg']['f1-score']
        logging.info(f"{name} Macro F1-Score: {macro_f1:.2f}")

    with open('outputs/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Focus on Ensemble model for visualizations and deployment
    ensemble_model = models['Ensemble']
    y_pred = ensemble_model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.title('Confusion Matrix - Voting Classifier Ensemble')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('outputs/confusion_matrix_ensemble.png')
    plt.close()

    y_test_bin = label_binarize(y_test, classes=range(5))
    y_score = ensemble_model.predict_proba(X_test_tfidf)
    plt.figure(figsize=(8, 6))
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic Curve - Voting Classifier Ensemble')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig('outputs/roc_curves_ensemble.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(5):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.title('Multiclass Precision-Recall Curve - Voting Classifier Ensemble')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.savefig('outputs/precision_recall_curves_ensemble.png')
    plt.close()

    # Log sample misclassified reviews using X_test, y_test, and y_pred
    misclassified_indices = y_test != y_pred
    misclassified_reviews = X_test[misclassified_indices]
    misclassified_actual = label_encoder.inverse_transform(y_test[misclassified_indices])
    misclassified_pred = label_encoder.inverse_transform(y_pred[misclassified_indices])
    logging.info("Sample misclassified reviews:")
    for review, actual, pred in list(zip(misclassified_reviews, misclassified_actual, misclassified_pred))[:3]:
        logging.info(f"Review: {review[:100]}... Actual: {actual}, Predicted: {pred}")

    # Save the Ensemble model
    joblib.dump(ensemble_model, 'models/review_score_model.pkl')

    # 4.6 Helpfulness Analysis
    X_help = df_helpfulness['cleaned_text']
    y_help = df_helpfulness['Helpfulness']
    X_train_help, X_test_help, y_train_help, y_test_help = train_test_split(X_help, y_help, test_size=0.2, random_state=43)
    X_train_help_tfidf = vectorizer.transform(X_train_help)
    X_test_help_tfidf = vectorizer.transform(X_test_help)
    scale_pos_weight = sum(y_train_help == 0) / sum(y_train_help == 1)
    help_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=43, scale_pos_weight=scale_pos_weight)
    help_model.fit(X_train_help_tfidf, y_train_help)
    help_accuracy = help_model.score(X_test_help_tfidf, y_test_help)
    y_pred_help = help_model.predict(X_test_help_tfidf)
    logging.info(f"Helpfulness Accuracy: {help_accuracy * 100:.2f}%")
    logging.info("Helpfulness Report:\n%s", classification_report(y_test_help, y_pred_help))
    y_score_help = help_model.predict_proba(X_test_help_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_help, y_score_help)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Helpfulness (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Helpfulness Prediction')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig('outputs/helpfulness_roc.png')
    plt.close()
    joblib.dump(help_model, 'models/helpfulness_model.pkl')

if __name__ == "__main__":
    main()