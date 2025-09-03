import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def create_bilstm_autoencoder(timesteps, n_features, latent_dim=64):
    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(latent_dim, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(latent_dim//2, activation='relu', return_sequences=False)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim//2, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(latent_dim, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
