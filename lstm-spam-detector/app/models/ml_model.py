# app/models/ml_model.py
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from .database import db, ModelPerformance
import joblib

class EnhancedLSTMSpamDetector:
    def __init__(self, config):
        self.config = config
        self.max_features = config.MAX_FEATURES
        self.max_length = config.MAX_LENGTH
        self.embedding_dim = config.EMBEDDING_DIM
        self.model_path = config.MODEL_PATH
        
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing model
        self.load_model_if_exists()
    
    def load_model_if_exists(self):
        """Load pre-trained model if it exists"""
        model_file = os.path.join(self.model_path, 'lstm_spam_model.h5')
        tokenizer_file = os.path.join(self.model_path, 'tokenizer.pkl')
        encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
        
        try:
            if all(os.path.exists(f) for f in [model_file, tokenizer_file, encoder_file]):
                self.model = load_model(model_file)
                self.tokenizer = joblib.load(tokenizer_file)
                self.label_encoder = joblib.load(encoder_file)
                print("Pre-trained model loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        if self.model is not None:
            model_file = os.path.join(self.model_path, 'lstm_spam_model.h5')
            tokenizer_file = os.path.join(self.model_path, 'tokenizer.pkl')
            encoder_file = os.path.join(self.model_path, 'label_encoder.pkl')
            
            self.model.save(model_file)
            joblib.dump(self.tokenizer, tokenizer_file)
            joblib.dump(self.label_encoder, encoder_file)
            print("Model saved successfully!")
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            words = nltk.word_tokenize(text)
        except:
            words = text.split()
        
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def train_from_database(self):
        """Train model using data from database"""
        from .database import Email, Prediction
        
        # Query emails with known labels
        emails = db.session.query(Email).filter(Email.actual_label.isnot(None)).all()
        
        if len(emails) < 100:  # Minimum required for training
            return False, "Insufficient labeled data in database"
        
        # Prepare data
        texts = []
        labels = []
        
        for email in emails:
            content = f"{email.subject or ''} {email.content}"
            texts.append(content)
            labels.append(email.actual_label)
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        # Train the model
        results = self.train_model(df)
        
        # Save performance to database
        if results:
            performance = ModelPerformance(
                model_version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                accuracy=results['accuracy'],
                precision=results['precision'],
                recall=results['recall'],
                f1_score=results['f1_score'],
                training_data_size=len(df),
                notes="Trained from database data"
            )
            db.session.add(performance)
            db.session.commit()
        
        return True, "Model trained successfully"
    
    def train_model(self, df):
        """Train the LSTM model"""
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 0]
        
        # Prepare sequences
        X, y = self.prepare_sequences(df)
        
        # Apply SMOTE
        X, y = self.apply_smote(X, y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Build and train model
        self.build_model()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Save model
        self.save_model()
        
        return results
    
    def prepare_sequences(self, df):
        """Tokenize and pad sequences"""
        texts = df['processed_text'].values
        labels = df['label'].values
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Initialize and fit tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        return X, labels_encoded
    
    def apply_smote(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        # Reshape X for SMOTE
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        
        # Reshape back
        X_resampled = X_resampled.reshape(-1, self.max_length)
        
        return X_resampled, y_resampled
    
    def build_model(self):
        """Build the LSTM model architecture"""
        model = Sequential([
            Embedding(input_dim=self.max_features, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def predict(self, text):
        """Predict if a text is spam or ham"""
        if self.model is None or self.tokenizer is None:
            return None
        
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length, 
                                      padding='post', truncating='post')
        
        # Predict
        probability = self.model.predict(padded_sequence)[0][0]
        prediction = 'spam' if probability > 0.5 else 'ham'
        
        return {
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(max(probability, 1 - probability))
        }
    
    def predict_email(self, email_id):
        """Predict spam for an email in database and store result"""
        from .database import Email, Prediction
        
        email = Email.query.get(email_id)
        if not email:
            return None
        
        # Combine subject and content
        text = f"{email.subject or ''} {email.content}"
        
        # Make prediction
        result = self.predict(text)
        
        if result:
            # Store prediction in database
            prediction = Prediction(
                user_id=email.user_id,
                email_id=email.id,
                predicted_label=result['prediction'],
                confidence_score=result['confidence'],
                model_version='v1.0'
            )
            db.session.add(prediction)
            db.session.commit()
        
        return result
