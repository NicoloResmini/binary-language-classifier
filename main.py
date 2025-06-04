# Language Classifier - Binary Italian Detection
# iGenius Take Home Assessment

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, request, jsonify
import pickle
import os

# ============================================
# 1. DATA PREPROCESSING AND MODEL TRAINING
# ============================================

class ItalianLanguageClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents='unicode'
        )
        self.model = MultinomialNB(alpha=0.1)
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep accented chars for language detection
        text = re.sub(r'[^\w\sàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', ' ', text)
        return text.strip()
    
    def load_and_prepare_data(self, csv_path):
        """Load dataset and prepare binary classification data"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Languages in dataset: {df['Language'].unique()}")
        print(f"Italian samples: {sum(df['Language'] == 'Italian')}")
        
        # Preprocess text
        df['text_clean'] = df['Text'].apply(self.preprocess_text)
        
        # Create binary labels: 1 for Italian, 0 for non-Italian
        df['is_italian'] = (df['Language'] == 'Italian').astype(int)
        
        # Balance the dataset - take equal samples of Italian and non-Italian
        italian_samples = df[df['is_italian'] == 1]
        non_italian_samples = df[df['is_italian'] == 0]
        
        # Sample to balance classes
        min_samples = min(len(italian_samples), len(non_italian_samples))
        balanced_df = pd.concat([
            italian_samples.sample(n=min_samples, random_state=42),
            non_italian_samples.sample(n=min_samples, random_state=42)
        ])
        
        print(f"Balanced dataset shape: {balanced_df.shape}")
        print(f"Class distribution: {balanced_df['is_italian'].value_counts()}")
        
        return balanced_df
    
    def train(self, csv_path):
        """Train the classifier"""
        # Load and prepare data
        df = self.load_and_prepare_data(csv_path)
        
        X = df['text_clean']
        y = df['is_italian']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training TF-IDF vectorizer...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print("Training classifier...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Italian', 'Italian']))
        
        self.is_trained = True
        return accuracy
    
    def predict(self, text):
        """Predict if text is Italian (1) or not (0)"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        return {
            'prediction': int(prediction),
            'confidence': float(max(probability)),
            'probabilities': {
                'non_italian': float(probability[0]),
                'italian': float(probability[1])
            }
        }
    
    def save_model(self, model_path='italian_classifier.pkl'):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='italian_classifier.pkl'):
        """Load trained model and vectorizer"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {model_path}")

# ============================================
# 2. FLASK API SERVER
# ============================================

# Global classifier instance
classifier = ItalianLanguageClassifier()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'Text' not in data:
            return jsonify({
                'error': 'Missing Text field in request body'
            }), 400
        
        text = data['Text']
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text must be a non-empty string'
            }), 400
        
        # Make prediction
        result = classifier.predict(text)
        
        return jsonify({
            'text': text,
            'prediction': result['prediction'],
            'is_italian': bool(result['prediction']),
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': classifier.is_trained
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage information"""
    return jsonify({
        'message': 'Italian Language Classifier API',
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'body': {'Text': 'Your text here'},
            'response': {
                'prediction': '1 for Italian, 0 for non-Italian',
                'is_italian': 'boolean',
                'confidence': 'prediction confidence',
                'probabilities': 'class probabilities'
            }
        }
    })

# ============================================
# 3. MAIN EXECUTION
# ============================================

def main():
    """Main function to train model and start API server"""
    
    # Configuration
    CSV_PATH = 'dataset.csv'  # Path to your dataset
    MODEL_PATH = 'italian_classifier.pkl'
    
    # Try to load existing model, otherwise train new one
    try:
        print("Attempting to load existing model...")
        classifier.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        
        # Check if dataset exists
        if not os.path.exists(CSV_PATH):
            print(f"Error: Dataset file '{CSV_PATH}' not found!")
            print("Please download the dataset from Kaggle and place it in the same directory.")
            return
        
        # Train model
        try:
            accuracy = classifier.train(CSV_PATH)
            print(f"Model trained with accuracy: {accuracy:.4f}")
            
            # Save model
            classifier.save_model(MODEL_PATH)
            
        except Exception as e:
            print(f"Error training model: {e}")
            return
    
    # Test the classifier
    print("\n" + "="*50)
    print("TESTING THE CLASSIFIER")
    print("="*50)
    
    test_sentences = [
        "Questa è una frase in italiano!",
        "This is an English sentence.",
        "Ciao, come stai oggi?",
        "Hello, how are you today?",
        "Mi piace molto la pizza italiana.",
        "I love Italian pizza very much.",
        "Buongiorno, vorrei un caffè per favore.",
        "Good morning, I would like a coffee please."
    ]
    
    for text in test_sentences:
        result = classifier.predict(text)
        print(f"Text: '{text}'")
        print(f"Prediction: {result['prediction']} ({'Italian' if result['prediction'] else 'Non-Italian'})")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 50)
    
    # Start Flask server
    print("\n" + "="*50)
    print("STARTING API SERVER")
    print("="*50)
    print("Server will be available at: http://localhost:5000")
    print("Prediction endpoint: http://localhost:5000/predict")
    print("Health check: http://localhost:5000/health")
    print("\nExample usage:")
    print("curl -X POST http://localhost:5000/predict \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"Text\": \"questa è una frase in italiano!\"}'")
    print("\nPress Ctrl+C to stop the server")
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()

# ============================================
# 4. ALTERNATIVE SIMPLE IMPLEMENTATION
# ============================================

"""
ALTERNATIVE IMPLEMENTATION WITHOUT SKLEARN (simpler approach):
This uses basic character n-gram analysis and frequency counting.
Less accurate but demonstrates understanding of the underlying concepts.

class SimpleItalianClassifier:
    def __init__(self):
        self.italian_patterns = {}
        self.non_italian_patterns = {}
        self.is_trained = False
    
    def extract_features(self, text):
        # Convert to lowercase and extract character n-grams
        text = text.lower()
        features = {}
        
        # Character trigrams
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            features[trigram] = features.get(trigram, 0) + 1
        
        # Common Italian patterns
        italian_indicators = ['gli', 'lla', 'tto', 'nno', 'zio', 'chi', 'ghi']
        for indicator in italian_indicators:
            features[f'has_{indicator}'] = indicator in text
        
        return features
    
    def train_simple(self, texts, labels):
        italian_texts = [t for t, l in zip(texts, labels) if l == 1]
        non_italian_texts = [t for t, l in zip(texts, labels) if l == 0]
        
        # Build pattern dictionaries
        for text in italian_texts:
            features = self.extract_features(text)
            for feature, count in features.items():
                self.italian_patterns[feature] = self.italian_patterns.get(feature, 0) + count
        
        for text in non_italian_texts:
            features = self.extract_features(text)
            for feature, count in features.items():
                self.non_italian_patterns[feature] = self.non_italian_patterns.get(feature, 0) + count
        
        self.is_trained = True
    
    def predict_simple(self, text):
        if not self.is_trained:
            return 0
        
        features = self.extract_features(text)
        italian_score = 0
        non_italian_score = 0
        
        for feature, count in features.items():
            italian_score += self.italian_patterns.get(feature, 0) * count
            non_italian_score += self.non_italian_patterns.get(feature, 0) * count
        
        return 1 if italian_score > non_italian_score else 0
"""