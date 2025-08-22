# Contents of /lstm-spam-detector/lstm-spam-detector/src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def clean_text(self, text):
        text = re.sub(r'\W', ' ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self):
        self.data['cleaned_text'] = self.data['text'].apply(self.clean_text)
        label_encoder = LabelEncoder()
        self.data['label'] = label_encoder.fit_transform(self.data['label'])
        return self.data

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data['cleaned_text']
        y = self.data['label']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)