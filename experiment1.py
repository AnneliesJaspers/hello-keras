import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt


def build_model(number_neurons=512):
    inp = keras.Input(shape=(1,))
    hidden1 = keras.layers.Dense(number_neurons, activation='relu')(inp)
    #hidden2 = keras.layers.Dense(512, activation='relu')(hidden1)
    output = keras.layers.Dense(1, activation='linear')(hidden1)
    model = keras.models.Model(inputs=inp, outputs=output)
    return model


if __name__ == '__main__':
    dataset_size = 100
    std_x = 1
    std_noise = std_x

    x = np.random.normal(scale=std_x, size=dataset_size)
    epsilon = np.random.normal(scale=std_noise, size=dataset_size)
    y = x * x + epsilon

    x_test = np.linspace(-5, 5, 500)
    y_test = x_test * x_test

    plt.figure()
    plt.scatter(x, y, alpha=0.2)
    plt.show()

    df = pd.DataFrame({'x': x, 'y': y})

    model = build_model()

    adam=keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mse')

    result = model.fit(df['x'], df['y'], epochs=100, batch_size=32, validation_split=0.9)
    plt.plot(result.history['val_loss'])
    plt.plot(result.history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.axhline(1, color='red')
    plt.show()

    y_hat = model.predict(x_test)
    plt.plot(x_test, y_hat)
    plt.plot(x_test, y_test, color='red')
    plt.scatter(x, y, alpha=0.1)
    plt.legend(['prediction', 'ground_truth'])
    plt.show()

    loss = []
    val_loss = []
    log_two = np.arange(13)
    powers_of_two = [2 ** power for power in log_two]

    for number_neurons in powers_of_two:
        model = build_model()
        model.compile(optimizer=adam, loss='mse')
        result = model.fit(df['x'], df['y'], epochs=100, batch_size=32, validation_split=0.5)
        loss.append(result.history['loss'][-1])
        val_loss.append(result.history['val_loss'][-1])
    plt.plot(log_two, loss)
    plt.plot(log_two, val_loss)
    plt.xticks(log_two, labels=powers_of_two)
    plt.legend(['loss', 'val_loss'])
    plt.show()




