Predicting Review Scores Using Sentiment Analysis of Amazon Fine Food Reviews
Overview
This project develops a machine learning system to predict review scores (1-5 stars) and helpfulness of Amazon Fine Food Reviews using sentiment analysis. A 50,000-review subset of the dataset is preprocessed, balanced using SMOTE, and used to train a Voting Classifier Ensemble model (combining Logistic Regression, Naive Bayes, and Random Forest) for score prediction, achieving 68.46% accuracy. An XGBoost model predicts helpfulness with 77.03% accuracy. The system is deployed as a Flask-based web application for real-time predictions, displaying scores, helpfulness, sentiment, and influential words, with a feedback mechanism. Ethical AI practices ensure fairness and transparency.
This project was developed as part of a B.Tech (Computer Science & Engineering, AI Specialization) program at Babu Banarasi Das University, Lucknow, by Shushil Suyel, Vaishnavi Kashyap, Vishal Mishra, and Yash Pathak under the supervision of Ms. Ritu Dwivedi.
Features

Data Preprocessing: Cleans text by removing noise (URLs, special characters), tokenizing, removing stopwords, and lemmatizing.
Exploratory Data Analysis (EDA): Visualizes review score distribution, frequent words, review lengths, and sentiment correlations.
Model Development: Trains a Voting Classifier Ensemble (Logistic Regression, Naive Bayes, Random Forest) for review score prediction and an XGBoost model for helpfulness prediction.
Evaluation: Uses advanced metrics like macro F1-score (0.50 for Ensemble), ROC curves, and confusion matrices.
Web Application: Deploys a Flask app for real-time predictions, showing scores, helpfulness, sentiment, and top features.
Ethical AI: Mitigates bias using SMOTE and class weighting, ensuring transparency.

Dataset
The project uses a 50,000-review subset of the Amazon Fine Food Reviews dataset. Key columns include:

Text: Review text
Score: Rating (1-5 stars)
HelpfulnessNumerator: Number of helpful votes
HelpfulnessDenominator: Total votes

Note: The dataset (Reviews.csv) is not included in this repository due to its size. Download it from the link above and place it in the project root directory.
Requirements

Python: 3.8+
Libraries:
pandas, numpy: Data processing
nltk, vaderSentiment: NLP and sentiment analysis
scikit-learn, imblearn, xgboost: Machine learning
plotly, seaborn, matplotlib, wordcloud: Visualization
flask: Web application
joblib: Model saving



Install dependencies using:
pip install -r requirements.txt

Download NLTK data:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

Project Structure
├── Reviews.csv               # Dataset (not included, download from Kaggle)
├── main.py                   # Main script for data preprocessing, EDA, and model training
├── app.py                    # Flask application for web interface
├── requirements.txt          # List of dependencies
├── templates/
│   └── index.html            # HTML template for Flask app
├── models/
│   ├── review_score_model.pkl    # Trained Voting Classifier Ensemble model
│   ├── helpfulness_model.pkl     # Trained XGBoost model
│   ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│   └── label_encoder.pkl         # Label encoder for scores
├── outputs/
│   ├── score_distribution.png    # Review score histogram
│   ├── wordcloud.png             # Frequent words visualization
│   ├── lengths.png               # Review length distribution
│   ├── sentiment.png             # Sentiment-score correlation
│   ├── sentiment_distribution.png # Sentiment distribution by score
│   ├── confusion_matrix_ensemble.png # Ensemble confusion matrix
│   ├── roc_curves_ensemble.png   # ROC curves for Ensemble
│   ├── precision_recall_curves_ensemble.png # Precision-recall curves
│   ├── helpfulness_roc.png       # ROC curve for helpfulness prediction
│   └── model_results.json        # Model performance metrics
├── README.md                 # This file
└── Project_Report_Final.pdf  # Full project report (optional)

Setup Instructions

Clone the Repository : git clone https://github.com/Shusitsuyel/AmazoonFineFoodReviewSentimentAnalysis.git
cd amazon-fine-food-reviews-sentiment-analysis


Download the Dataset:
Kaggle Link for Dataset : https://www.kaggle.com/code/laowingkin/amazon-fine-food-review-sentiment-analysis?select=Reviews.csv
Download Reviews.csv from the Kaggle link above.
Place it in the project root directory.


Install Dependencies:
pip install -r requirements.txt


Download NLTK Data:Run the NLTK download commands mentioned in the Requirements section.

Prepare Models and Outputs:

If you want to retrain the models, run main.py (see Usage section).
Pre-trained models and outputs are provided in models/ and outputs/ directories for immediate use.



Usage
1. Preprocess Data, Perform EDA, and Train Models
Run the main script to preprocess the dataset, perform EDA, train the Voting Classifier Ensemble and XGBoost models, and generate visualizations:
python main.py --data Reviews.csv --sample 50000


--data: Path to the dataset (default: Reviews.csv)
--sample: Number of reviews to sample (default: 50,000)

This will:

Preprocess the data and save cleaned text.
Generate EDA plots in outputs/ (e.g., score_distribution.png, wordcloud.png).
Train models and save them in models/ (e.g., review_score_model.pkl).
Save performance metrics in outputs/model_results.json.

2. Run the Web Application
Launch the Flask app to use the web interface for real-time predictions:
python app.py


Open your browser and go to http://0.0.0.0:5000.
Enter a review (e.g., "It's alright, nothing special").
View the predicted score, helpfulness, sentiment, and top influential words.

Sample Output:
Amazon Review Sentiment Predictor
Enter Review: It's alright, nothing special.
Predicted Score: 3 stars
Helpfulness: Helpful
Sentiment: Neutral
Top Influential Words:
- special (Weight: 0.753)
- nothing (Weight: 0.657)

3. Example Code for Prediction
You can also use the prediction function directly:
from prediction import predict_review_score

review = "I have bought several of the Vitality canned dog food products and have found them all to be of good quality."
score = predict_review_score(review)
print(f"Predicted Score: {score}")  # Output: Predicted Score: 4

Results

Review Score Prediction (Voting Classifier Ensemble):
Accuracy: 68.46%
Macro F1-Score: 0.50
Strong performance on 5-star reviews (recall: 0.82), but challenges with minority classes (e.g., 2-star recall: 0.31).


Helpfulness Prediction (XGBoost):
Accuracy: 77.03%
Recall for "Not Helpful" class is low (0.06), indicating areas for improvement.


Visualizations:
Review score distribution shows 64.14% 5-star reviews.
Word cloud highlights frequent terms like "great," "taste," "product."
ROC and precision-recall curves provide detailed model insights.



Ethical Considerations

Bias Mitigation: Used SMOTE and class weighting to address class imbalance (64.14% 5-star reviews).
Transparency: Documented limitations (e.g., minority class performance, computational constraints).
Data Privacy: The dataset is anonymized, ensuring no personally identifiable information is used.

Challenges and Limitations

Class Imbalance: Addressed with SMOTE, but minority class performance could be improved.
Text Noise: Handled through preprocessing, though some nuances (e.g., sarcasm) may be missed.
Computational Constraints: Limited to 50,000 reviews and machine learning models due to hardware limitations.

Future Work

Explore deep learning models like BERT for better accuracy.
Use focal loss or GAN-based augmentation to improve minority class performance.
Incorporate reviewer credibility features for helpfulness prediction.
Deploy on AWS with Gunicorn for scalability.
Expand to other domains (e.g., electronics reviews).

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a pull request with a description of your changes.


Shushil Suyel (1210439072)
Vishal Mishra (1210439082)
Vaishnavi Kashyap (1210439078)
Yash Pathak (1210439083)

Supervisor: Ms. Ritu Dwivedi, Assistant Professor, Babu Banarasi Das University, Lucknow
Acknowledgments

Thanks to Ms. Ritu Dwivedi for her guidance and support.
The Amazon Fine Food Reviews dataset is credited to Kaggle and the original authors.
Libraries like Scikit-learn, NLTK, and Flask made this project possible.



