from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Reshape, Dense, ELU,
                                     Conv1D, MaxPooling1D, UpSampling1D,
                                     BatchNormalization, Dropout)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh
import utils.constants as const
import numpy as np

# Just disables the warning, doesn't enable AVX
__import__("os").environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def MSE(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Computes the mean squared errors (MSE) between labels and predictions

    :param gt: ground truth / labels
    :param pred: predictions
    :return: MSE
    """
    return ((gt - pred) ** 2).mean(axis=1)


class NeuralNetwork:
    def __init__(self,
                 mode=const.NN.AUTOENCODER,
                 n_features: int = const.N_FEATURES,
                 optimizer='Adam',
                 loss='logcosh'):
        """
        Initialization of neural network architecture

        :param mode: AUTOENCODER or CNN
        :param n_features: number of features
        :param optimizer: optimizer
        :param loss: loss function
        """
        self.mode = mode
        if self.mode == const.NN.AUTOENCODER:
            self.model = Sequential(layers=[
                Input(shape=(n_features,), name="input"),
                # Encoder
                Dense(units=16, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_encoder_1"),
                Dropout(rate=0.2, name="dropout_1"),
                Dense(units=14, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_encoder_2"),
                Dropout(rate=0.2, name="dropout_2"),
                Dense(units=12, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_encoder_3"),
                Dropout(rate=0.2, name="dropout_3"),
                # Middle layer
                Dense(units=10, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_middle_4"),
                # Decoder
                Dense(units=12, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_decoder_5"),
                Dense(units=14, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_decoder_6"),
                Dense(units=16, activation=ELU(), kernel_regularizer=l2(1e-3), name="dense_decoder_7"),
                Dense(units=n_features, name="output")
            ], name='ED')
        elif self.mode == const.NN.CNN:
            self.model = Sequential(layers=[
                Input(shape=(n_features,), name="input"),
                Reshape(target_shape=(n_features, 1)),
                # Encoder
                Conv1D(filters=4, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=4, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=8, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=8, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                MaxPooling1D(pool_size=2),
                # Decoder
                UpSampling1D(size=2),
                Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                UpSampling1D(size=2),
                Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=16, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                UpSampling1D(size=2),
                Conv1D(filters=8, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=8, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                UpSampling1D(size=2),
                Conv1D(filters=4, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=4, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                # Predictor
                Conv1D(filters=2, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Conv1D(filters=1, kernel_size=3, padding='same', kernel_regularizer=l2(0.001)),
                Reshape(target_shape=(n_features,))
            ], name='CNN')
        else:
            raise ValueError(f"[Warning] invalid model type: {mode}")

        self.model.compile(optimizer=optimizer, loss=loss)

    def summary(self) -> None:
        """
        Prints a string summary of the network.

        :return: None
        """
        self.model.summary()

    def fit(self, X: np.ndarray,
            X_valid: np.ndarray = None,
            batch_size: int = 64,
            epochs: int = 100,
            verbose: int = 0):
        """
        Train the model

        :param X: train data
        :param X_valid: validation data
        :param batch_size: batch size
        :param epochs: number of epochs
        :param verbose: verbose output to stdout,
                        0 -- silence, [1, 2, 3] -- more verbose
        :return: self
        """
        validation_data = (X_valid, X_valid) if X_valid is not None else None
        self.model.fit(X, X,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=True)
        return self

    def decision_function(self, y: np.ndarray) -> np.ndarray:
        """
        Distance closer to zero for the inlier

        :param y: feature vectors
        :return: -MSE(y, prediction)
        """
        y_pred = self.model.predict(y)
        return -MSE(y, y_pred)
