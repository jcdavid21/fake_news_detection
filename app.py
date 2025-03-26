# app.py - Flask application
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
import joblib
import os
import traceback

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters, numbers, and punctuation
        text = re.sub('[^a-zA-Z]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        words = nltk.word_tokenize(text)
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        print(f"Stemmed words after removing stopwords: {words}")
        
        return ' '.join(words)
    
    def load_data(self, true_news_path, fake_news_path):
        """Load and prepare the dataset"""
        try:
            # Load datasets
            true_news = pd.read_csv(true_news_path)
            fake_news = pd.read_csv(fake_news_path)
            
            # Ensure 'text' column exists in both datasets
            missing_columns = []
            if 'text' not in true_news.columns:
                missing_columns.append('true_news')
                print("Columns in true_news:", true_news.columns)  # Debug: Print columns
            if 'text' not in fake_news.columns:
                missing_columns.append('fake_news')
                print("Columns in fake_news:", fake_news.columns)  # Debug: Print columns
            
            if missing_columns:
                raise ValueError(f"'text' column missing in: {', '.join(missing_columns)}")
            
            # Add labels
            true_news['label'] = 1  # 1 for true news
            fake_news['label'] = 0  # 0 for fake news
            
            # Combine datasets
            data = pd.concat([true_news, fake_news], ignore_index=True)
            
            # Handle NaN values in the 'text' column
            if data['text'].isnull().any():
                print("Warning: NaN values found in 'text' column. Replaced with empty strings.")  # Debug: Print NaN warning
                data['text'] = data['text'].fillna('')
            
            # Check for empty strings
            empty_text_count = data[data['text'].str.strip() == ''].shape[0]
            if empty_text_count > 0:
                print(f"Warning: {empty_text_count} empty strings found in 'text' column.")  # Debug: Print empty string warning
            
            # Preprocess text
            data['processed_text'] = data['text'].apply(self.preprocess_text)
            print("Processed text sample:", data['processed_text'].head())  # Debug: Print sample of processed text
            
            # Check if the dataset is empty after preprocessing
            if data.empty:
                raise ValueError("Dataset is empty after preprocessing.")
            
            return data
        
        except FileNotFoundError as e:
            raise ValueError(f"File not found: {str(e)}")
        
        except pd.errors.EmptyDataError:
            raise ValueError("One or both CSV files are empty.")
        
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
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
        
        # Save model and vectorizer
        self.save_model('model.pkl', 'vectorizer.pkl')
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(self, model_path, vectorizer_path):
        """Save the trained model and vectorizer"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
    def load_model(self, model_path, vectorizer_path):
        """Load a trained model and vectorizer"""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            return True
        return False
    
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
            'confidence': float(probability[1] if prediction == 1 else probability[0])
        }
        
        return result

# Initialize detector
detector = FakeNewsDetector()

# Try to load pre-trained model
model_loaded = detector.load_model('model.pkl', 'vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if not model_loaded and not hasattr(detector.model, 'coef_'):
            return jsonify({'error': 'Model not trained yet'}), 400
        
        result = detector.predict(text)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        files = request.files
        
        if 'true_news' not in files or 'fake_news' not in files:
            return jsonify({'error': 'Both true news and fake news files are required'}), 400
        
        true_news_file = files['true_news']
        fake_news_file = files['fake_news']
        
        # Save files temporarily
        true_news_path = 'true_news_temp.csv'
        fake_news_path = 'fake_news_temp.csv'
        
        true_news_file.save(true_news_path)
        fake_news_file.save(fake_news_path)
        
        
        # Load and train
        data = detector.load_data(true_news_path, fake_news_path)


        results = detector.train(data)
        
        # Clean up temp files
        if os.path.exists(true_news_path):
            os.remove(true_news_path)
        if os.path.exists(fake_news_path):
            os.remove(fake_news_path)
        
        global model_loaded
        model_loaded = True
        
        return jsonify({
            'success': True,
            'accuracy': float(results['accuracy']),
            'message': 'Model trained successfully'
        })
    
    except Exception as e:
        app.logger.error(f"Error in train: {str(e)}\n{traceback.format_exc()}")
        # Clean up temp files if they exist
        if os.path.exists('true_news_temp.csv'):
            os.remove('true_news_temp.csv')
        if os.path.exists('fake_news_temp.csv'):
            os.remove('fake_news_temp.csv')
        return jsonify({'error': str(e)}), 500

@app.route('/check-file', methods=['GET'])
def check_file():
    """Check if a file exists in the current directory"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename parameter is required'}), 400
    
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return jsonify({'exists': True}), 200
    else:
        return jsonify({'exists': False}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)