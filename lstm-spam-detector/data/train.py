# Contents of /lstm-spam-detector/lstm-spam-detector/src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from model import create_model  # Assuming create_model is defined in model.py
from data_preprocessing import preprocess_data  # Assuming preprocess_data is defined in data_preprocessing.py

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def main():
    # Load and preprocess the data
    data = load_data('../data/spam_dataset.csv')
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pad sequences to ensure uniform input size
    X_train = pad_sequences(X_train, maxlen=100)
    X_test = pad_sequences(X_test, maxlen=100)
    
    # Convert labels to categorical format
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Create and train the LSTM model
    model = create_model()
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save('lstm_spam_detector.h5')

if __name__ == "__main__":
    main()