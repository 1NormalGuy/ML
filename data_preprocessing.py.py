import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data = data[['label', 'text']]
    data.columns = ['label', 'text']

    def clean_text(text):
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^\s+|\s+?$', '', text)
        text = text.lower()
        return text

    data['text'] = data['text'].apply(clean_text)
    return data

def preprocess_data(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer

if __name__ == "__main__":
    data = load_and_clean_data('trec06p/spam_data.csv')
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)
    np.save('X_train.npy', X_train.toarray())
    np.save('X_test.npy', X_test.toarray())
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)