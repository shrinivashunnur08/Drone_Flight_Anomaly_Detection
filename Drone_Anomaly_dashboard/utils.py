import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path, timesteps=10):
    df = pd.read_csv(csv_path)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = MinMaxScaler()
    features = df.drop(columns=['timestamp']).values
    scaled = scaler.fit_transform(features)
    
    # Prepare sequences for LSTM
    X = []
    for i in range(len(scaled) - timesteps):
        X.append(scaled[i:i+timesteps])
    X = np.array(X)
    return X, df, scaler

def detect_anomalies(model, X, threshold=None):
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))
    if threshold is None:
        threshold = np.mean(mse) + 2*np.std(mse)
    anomalies = mse > threshold
    return anomalies, mse, threshold
