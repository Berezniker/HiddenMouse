from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import LogCosh
from sklearn.preprocessing import StandardScaler
import pandas as pd


def customize_for_Google_Colab():
    import tensorflow as tf
    tf.test.gpu_device_name()
    # подключить GPU: Runtime -> Change runtime type -> Hardware accelerator: GPU
    # вывод: '/device:GPU:0'

    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    # монтируем гугл диск

def get_model(input_shape):
    i = Input(shape=input_shape, name="input")
    m = Dense(units=64, activation=ELU(), kernel_regularizer="l2", name="dense_e1")(i)
    m = Dense(units=32, activation=ELU(), kernel_regularizer="l2", name="dense_e2")(m)
    m = Dense(units=16, activation=ELU(), kernel_regularizer="l2", name="dense_m3")(m)
    m = Dense(units=32, activation=ELU(), kernel_regularizer="l2", name="dense_d4")(m)
    m = Dense(units=64, activation=ELU(), kernel_regularizer="l2", name="dense_d5")(m)
    o = Dense(units=input_shape[0], name="output")(m)

    encoder_decoder = Model(inputs=i, outputs=o, name="ED")
    return encoder_decoder


if __name__ == '__main__':
    customize_for_Google_Colab()
    # ------------------------------ #
    dataset = "BALABIT"
    user = "user35"
    path = f"{dataset}/all_train_features/{user}/session_all.csv"
    X = pd.read_csv(path, header=None)
    X = X.drop(columns=[0, 2, 4]).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # ------------------------------ #
    X_valid = pd.read_csv(path.replace("train", "test"), header=None)
    X_valid = X_valid.drop(columns=[0, 2, 4]).values
    X_valid = scaler.transform(X_valid)
    # ------------------------------ #
    model = get_model(X.shape[1:])
    model.summary()
    # ------------------------------ #
    model.compile(optimizer=Adam(lr=0.001),
                  loss=LogCosh())
    # ------------------------------ #
    model.fit(X, X,
              batch_size=64,
              epochs=100,
              verbose=2,
              validation_data=(X_valid, X_valid),
              shuffle=False)
    # ------------------------------ #
    y_pred = model.predict(X)
    MSE = lambda y, y_pred: ((y - y_pred) ** 2).mean(axis=1)
    MSE(X, y_pred)
    # ------------------------------ #
