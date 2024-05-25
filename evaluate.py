import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data():
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    return X_test, y_test

def evaluate_model():
    X_test, y_test = load_data()
    model = tf.keras.models.load_model('spam_classifier_model.h5')
    X_test = np.expand_dims(X_test, axis=2)  # 扩展维度以适应Conv1D输入
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

if __name__ == "__main__":
    evaluate_model()