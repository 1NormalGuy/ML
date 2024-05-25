import numpy as np
import pickle
from model import build_model, compile_model
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    return X_train, y_train

def train_model():
    X_train, y_train = load_data()
    input_shape = (X_train.shape[1], 1)
    X_train = np.expand_dims(X_train, axis=2)  # 扩展维度以适应Conv1D输入
    model = build_model(input_shape)
    model = compile_model(model)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
    model.save('spam_classifier_model.h5')
    return history

if __name__ == "__main__":
    train_model()