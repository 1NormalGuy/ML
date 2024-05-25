import os
from data_preprocessing import load_and_clean_data, preprocess_data
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    # 数据预处理
    data = load_and_clean_data('trec06p/spam_data.csv')
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)
    np.save('X_train.npy', X_train.toarray())
    np.save('X_test.npy', X_test.toarray())
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # 训练模型
    train_model()
    
    # 评估模型
    evaluate_model()