from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask_cors import CORS

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

download_nltk_resources()

app = Flask(__name__)
CORS(app)

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not text or not isinstance(text, str):
            return ""  # Return empty string for invalid or empty text
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, numbers, and punctuation
        text = re.sub('[^a-zA-Z]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        words = nltk.word_tokenize(text)
        print(f"Original words: {words}")  # Debug print
        
        # Stem words, skipping invalid or very short words
        stemmed_words = []
        for word in words:
            if word not in self.stop_words:
                try:
                    if len(word) > 1:  # Only stem words longer than 1 character
                        stemmed_word = self.stemmer.stem(word)
                    else:
                        stemmed_word = word
                    stemmed_words.append(stemmed_word)
                except Exception as e:
                    print(f"Error stemming word '{word}': {e}")  # Debug print
                    stemmed_words.append(word)  # Keep the original word if stemming fails
        
        print(f"Stemmed words: {stemmed_words}")  # Debug print
        
        return ' '.join(stemmed_words)
    
    def load_data(self, true_news_path, fake_news_path):
        """Load and prepare the dataset"""
        # Load datasets
        true_news = pd.read_csv(true_news_path)
        fake_news = pd.read_csv(fake_news_path)
        
        # Add labels
        true_news['label'] = 1  # 1 for true news
        fake_news['label'] = 0  # 0 for fake news
        
        # Combine datasets
        data = pd.concat([true_news, fake_news], ignore_index=True)
        
        # Check for empty or invalid rows
        data = data.dropna(subset=['text'])  # Drop rows with empty text
        
        # Preprocess text
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        return data
    
    def train(self, data, test_size=0.2, random_state=42):
        """Train the fake news detection model"""
        # Split data
        X = data['processed_text']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Vectorize text data
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vectorized, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        vectorized_text = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(vectorized_text)[0]
        probability = self.model.predict_proba(vectorized_text)[0]
        
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': probability[1] if prediction == 1 else probability[0]
        }
        
        return result

# Initialize detector
detector = FakeNewsDetector()

# Load and prepare data
data = detector.load_data('True.csv', 'Fake.csv')

# Train model
detector.train(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news text from the form
    news_text = request.form['news']

    # Predict using the detector
    result = detector.predict(news_text)

    # Return the result as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5500)

