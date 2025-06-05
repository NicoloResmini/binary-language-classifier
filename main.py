import pandas as pd
import re
import os
import json
from collections import Counter
import string

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from flask import Flask, request, jsonify


# Impostazioni per riproducibilità (opzionale, ma buona pratica)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================
# 0. PYTORCH MLP MODEL DEFINITION
# ============================================
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Custom Dataset per PyTorch
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer_fn):
        self.texts = texts
        self.labels = labels
        self.vectorizer_fn = vectorizer_fn

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_vector = self.vectorizer_fn(self.texts[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return text_vector, label

# ============================================
# 1. BLC CLASS WITH PYTORCH INTEGRATION
# ============================================

class BLC:
    def __init__(self, ngram_range=(2, 4), max_features=10000, hidden_size=128, learning_rate=0.001, epochs=10):
        # N-gram parameters
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab = {}
        self.idx_to_ngram = {}

        # PyTorch model parameters
        self.model = None # Sarà istanziato in train o load
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.is_trained = False


    def _generate_char_ngrams(self, text):
        """Genera N-grammi di caratteri da un testo."""
        ngrams = set()
        # Aggiungiamo spazi per catturare n-grammi all'inizio/fine delle parole se si desidera
        # text = " " + text + " " # Opzionale, dipende dalla strategia
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.add(text[i:i+n])
        return list(ngrams)


    def _build_vocabulary(self, texts):
        """Costruisce il vocabolario dagli N-grammi dei testi di training."""
        print("Building vocabulary...")
        all_ngrams = Counter()
        for text in texts:
            ngrams = self._generate_char_ngrams(text)
            all_ngrams.update(ngrams)

        most_common_ngrams = all_ngrams.most_common(self.max_features)
        self.vocab = {ngram: i for i, (ngram, _) in enumerate(most_common_ngrams)}
        self.idx_to_ngram = {i: ngram for ngram, i in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")


    def _vectorize_text(self, text):
        """Converte un testo in un vettore binario basato sul vocabolario."""
        ngrams = self._generate_char_ngrams(text)
        vector = torch.zeros(len(self.vocab), dtype=torch.float32)
        for ngram in ngrams:
            if ngram in self.vocab:
                vector[self.vocab[ngram]] = 1 # Presenza binaria
        return vector.unsqueeze(0) # Aggiunge dimensione batch (1, vocab_size)


    def _vectorize_text_for_dataset(self, text):
        """Versione per DataLoader che non aggiunge dimensione batch."""
        ngrams = self._generate_char_ngrams(text)
        vector = torch.zeros(len(self.vocab), dtype=torch.float32)
        for ngram in ngrams:
            if ngram in self.vocab:
                vector[self.vocab[ngram]] = 1
        return vector


    def preprocess_text(self, text):
        """Pulisce e pre-processa il testo."""
        # Trasforma in minuscolo
        text = text.lower()
        # Rimuovi punteggiatura standard ma mantieni caratteri Unicode (es. lettere accentate)
        text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)
        # Rimuovi numeri
        text = re.sub(r'\d+', '', text)
        # Trasforma spazi multipli consecutivi in un singolo spazio
        text = re.sub(r'\s+', ' ', text)
        # Rimuovi eventuali spazi all'inizio e alla fine del testo
        text = text.strip()
        return text


    def load_and_prepare_data(self, csv_path):
        """Carica il dataset e prepara i dati per la classificazione binaria."""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Languages in dataset: {df['Language'].unique()}")
        print(f"Italian samples: {sum(df['Language'] == 'Italian')}")

        df['text_clean'] = df['Text'].apply(self.preprocess_text)
        df['is_italian'] = (df['Language'] == 'Italian').astype(int)

        # Bilanciamento del dataset come nella tua implementazione originale
        italian_samples = df[df['is_italian'] == 1]
        non_italian_samples = df[df['is_italian'] == 0]

        # Under-sampling the non_italian samples to balance the dataset (since original ratio is 1:14)
        min_samples = len(italian_samples)
        balanced_df = pd.concat([
            italian_samples.sample(n=min_samples, random_state=42),
            non_italian_samples.sample(n=min_samples*2, random_state=42) # ratio 1:2 
        ])
        
        if balanced_df.empty:
            raise ValueError("Dataset is empty after balancing. Check data and balancing logic.")

        print(f"Balanced dataset shape: {balanced_df.shape}")
        print(f"Class distribution: {balanced_df['is_italian'].value_counts()}")

        return balanced_df


    def train(self, csv_path):
        """Addestra il classificatore PyTorch."""
        df = self.load_and_prepare_data(csv_path)

        X = df['text_clean'].tolist()
        y = df['is_italian'].tolist()

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self._build_vocabulary(X_train_texts) # Costruisce il vocabolario solo sui dati di training

        # Crea istanza del modello
        input_size = len(self.vocab)
        if input_size == 0:
            raise ValueError("Vocabulary is empty. Check text preprocessing and data.")
        self.model = SimpleMLP(input_size, self.hidden_size, 1).to(self.device)

        # Loss e optimizer
        criterion = nn.BCELoss() # Binary Cross Entropy
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # DataLoader
        train_dataset = TextDataset(X_train_texts, y_train, self._vectorize_text_for_dataset)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        print("Training classifier...")
        self.model.train() # Mette il modello in modalità training
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_texts_vec, batch_labels in train_loader:
                batch_texts_vec = batch_texts_vec.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_texts_vec)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

        # Valutazione
        self.model.eval() # Mette il modello in modalità valutazione
        all_preds = []
        all_true = []
        with torch.no_grad(): # Disabilita calcolo gradienti per valutazione
            for text, label in zip(X_test_texts, y_test):
                text_vector = self._vectorize_text(text).to(self.device) # _vectorize_text aggiunge batch dim
                output = self.model(text_vector)
                predicted_class = (output.item() > 0.5)
                all_preds.append(int(predicted_class))
                all_true.append(label)

        accuracy = accuracy_score(all_true, all_preds)
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_true, all_preds, target_names=['Non-Italian', 'Italian'], zero_division=0))

        self.is_trained = True
        return accuracy


    def predict(self, text):
        """Predice se il testo è Italiano (1) o no (0)."""
        if not self.is_trained or not self.model:
            raise ValueError("Model must be trained or loaded first!")

        self.model.eval() # Modalità valutazione
        clean_text = self.preprocess_text(text)
        text_vector = self._vectorize_text(clean_text).to(self.device)

        with torch.no_grad():
            probability_italian = self.model(text_vector).item()

        prediction = 1 if probability_italian > 0.5 else 0
        confidence = probability_italian if prediction == 1 else 1 - probability_italian

        return {
            'prediction': prediction,
            'confidence': float(confidence)
        }


    def save_model(self, model_path_prefix='pytorch_model'):
        """Salva il modello PyTorch addestrato e il vocabolario."""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        model_state_path = f"{model_path_prefix}_state.pth"
        vocab_path = f"{model_path_prefix}_vocab.json"
        params_path = f"{model_path_prefix}_params.json"

        torch.save(self.model.state_dict(), model_state_path)

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

        model_params = {
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'hidden_size': self.hidden_size,
            'input_size': len(self.vocab) # Salva la dimensione di input usata
        }
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(model_params, f, indent=4)

        print(f"Model state saved to {model_state_path}")
        print(f"Vocabulary saved to {vocab_path}")
        print(f"Parameters saved to {params_path}")


    def load_model(self, model_path_prefix='pytorch_model'):
        """Carica un modello PyTorch e il vocabolario."""
        model_state_path = f"{model_path_prefix}_state.pth"
        vocab_path = f"{model_path_prefix}_vocab.json"
        params_path = f"{model_path_prefix}_params.json"

        if not (os.path.exists(model_state_path) and os.path.exists(vocab_path) and os.path.exists(params_path)):
            raise FileNotFoundError(f"Model files not found with prefix {model_path_prefix}. Ensure .pth, _vocab.json, and _params.json exist.")

        with open(params_path, 'r', encoding='utf-8') as f:
            model_params = json.load(f)
        
        self.ngram_range = tuple(model_params['ngram_range']) # json salva tuple come liste
        self.max_features = model_params['max_features']
        self.hidden_size = model_params['hidden_size']
        input_size = model_params['input_size']

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.idx_to_ngram = {i: ngram for ngram, i in self.vocab.items()} # Ricostruisci se necessario

        if input_size != len(self.vocab):
             print(f"Warning: Loaded vocabulary size ({len(self.vocab)}) differs from saved input size ({input_size}). Using loaded vocabulary size.")
             input_size = len(self.vocab)
        
        if input_size == 0 :
             raise ValueError("Loaded vocabulary is empty.")

        self.model = SimpleMLP(input_size, self.hidden_size, 1).to(self.device)
        self.model.load_state_dict(torch.load(model_state_path, map_location=self.device))
        self.model.eval() # Metti in modalità valutazione dopo il caricamento
        self.is_trained = True
        print(f"Model loaded from prefix {model_path_prefix}")


# ============================================
# 2. FLASK API SERVER (Identico alla tua versione)
# ============================================

# Global classifier instance
classifier = BLC() # Istanza con parametri di default per Ngram/MLP
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api(): # Rinominato per evitare conflitto con il metodo predict della classe
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        if not data or 'Text' not in data: # Mantenuto 'Text' per compatibilità con demo.py
            return jsonify({'error': 'Missing Text field in request body'}), 400
        
        text = data['Text']
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        result = classifier.predict(text) # Chiama il metodo predict della classe BLC
        
        return jsonify({
            'text': text,
            'prediction': result['prediction'],
            'is_italian': bool(result['prediction']),
            'confidence': result['confidence']
        })
        
    except Exception as e:
        print(f"Error in /predict endpoint: {e}") # Log dell'errore lato server
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
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
        'message': 'Binary Language Classifier API (PyTorch Version)',
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'body': {'Text': 'Your text here'}, # Mantenuto 'Text'
            'response': {
                'prediction': '1 for Italian, 0 for non-Italian',
                'is_italian': 'boolean',
                'confidence': 'prediction confidence'
            }
        }
    })

# ============================================
# 3. MAIN EXECUTION (Simile alla tua versione)
# ============================================

def main():
    """Main function to train model and start API server"""
    
    DATASET_PATH = 'dataset.csv' # Assicurati che questo file esista
    MODEL_PREFIX = 'pytorch_blc_model' # Prefisso per i file del modello PyTorch
    
    try:
        print("\nAttempting to load existing PyTorch model...")
        classifier.load_model(MODEL_PREFIX)
        print("PyTorch Model loaded successfully!")
        
    except FileNotFoundError:
        print(f"No existing model found with prefix '{MODEL_PREFIX}'. Training new model...")
        
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset file '{DATASET_PATH}' not found!")
            print("Please download the dataset e.g. from https://www.kaggle.com/datasets/basilb2s/language-detection and place it as 'dataset.csv'.")
            return
        
        try:
            accuracy = classifier.train(DATASET_PATH)
            print(f"Model trained with accuracy: {accuracy:.4f}")
            classifier.save_model(MODEL_PREFIX)
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc() # Stampa traceback completo per debug
            return
    
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        return

    if not classifier.is_trained:
        print("Classifier is not trained. Exiting.")
        return

    print("\nTESTING THE CLASSIFIER (after load/train):\n")
    test_sentences = [
        "Questa è una frase in italiano!",
        "This is an English sentence."
    ]
    
    for text in test_sentences:
        try:
            result = classifier.predict(text)
            print(f"Text: '{text}'")
            print(f"Prediction: {result['prediction']} ({'Italian' if result['prediction'] else 'Non-Italian'})")
            print(f"Confidence: {result['confidence'] * 100:.2f}%\n")
        except Exception as e:
            print(f"Error predicting for text '{text}': {e}")

    print("\nSTARTING API SERVER (PyTorch Version)...\n")
    print("Server is running at http://localhost:5000")
    print("Prediction endpoint: http://localhost:5000/predict (POST with JSON {'Text': 'your text'})")
    print("Health check endpoint: http://localhost:5000/health (GET)")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()