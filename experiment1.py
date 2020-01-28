import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def build_model(number_neurons=512, number_layers=2):
    inp = keras.Input(shape=(1,))
    hidden = inp
    for layer in range(number_layers):
        hidden = keras.layers.Dense(number_neurons, activation='relu')(hidden)
    output = keras.layers.Dense(1, activation='linear')(hidden)
    model = keras.models.Model(inputs=inp, outputs=output)
    return model


if __name__ == '__main__':
    train_size = 100
    validation_size = 1000
    test_size = 500
    max_x = 1
    std_noise = 0.2

    # Sample x uniformly
    # y is the sine of x + some noise
    x = np.random.uniform(low=-max_x, high=max_x, size=train_size)
    epsilon = np.random.normal(scale=std_noise, size=train_size)
    y = np.sin(x * 10) + epsilon

    x_val = np.random.uniform(low=-max_x, high=max_x, size=validation_size)
    y_val = np.sin(x_val * 10)

    # In order to study the effect of extrapolation, we will sample x_test from a wider range
    x_test = np.linspace(-max_x * 1.5, max_x * 1.5, test_size)
    y_test = np.sin(x_test * 10)

    # Plot of the training data
    plt.figure()
    plt.scatter(x, y, alpha=0.2)
    plt.show()

    # Build, train and fit a model.
    model = build_model()
    adam = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=adam, loss='mse')

    result = model.fit(x, y, epochs=500, batch_size=32, validation_data=(x_val, y_val))
    plt.plot(result.history['val_loss'])
    plt.plot(result.history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Training and validation error')
    plt.xlabel('Epochs')
    plt.savefig('one.png')
    # plt.show()
    plt.close()

    # Predict on the test set and plot the model
    y_hat = model.predict(x_test)
    plt.plot(x_test, y_hat)
    plt.plot(x_test, y_test, color='red')
    plt.scatter(x, y, alpha=0.1)
    plt.legend(['prediction', 'ground truth'])
    plt.axvline(-max_x, color='black')
    plt.axvline(max_x, color='black')
    plt.show()


    ##### EXPERIMENT 1: Influence of number of neurons in 2 layers #####
    loss = []
    val_loss = []
    log_two = np.arange(13)
    powers_of_two = [2 ** power for power in log_two]

    for number_neurons in powers_of_two:
        model = build_model(number_neurons)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        result = model.fit(x, y, epochs=100, batch_size=32, validation_data=(x_val, y_val))
        l = np.mean(result.history['loss'][-3:])
        vl = np.mean(result.history['val_loss'][-3:])
        loss.append(l)
        val_loss.append(vl)
    plt.plot(log_two, loss)
    plt.plot(log_two, val_loss)
    plt.xticks(log_two, labels=powers_of_two)
    plt.legend(['loss', 'val_loss'])
    plt.title('Training and validation error by number of neurons')
    plt.xlabel('Number of neurons')
    plt.show()

    ##### EXPERIMENT 2: Influence of number of layers#####
    loss = []
    val_loss = []
    layers = np.arange(1, 10)

    for number_layers in layers:
        model = build_model(number_layers=number_layers)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        result = model.fit(x, y, epochs=100, batch_size=32, validation_data=(x_val, y_val))
        l = np.mean(result.history['loss'][-3:])
        vl = np.mean(result.history['val_loss'][-3:])
        loss.append(l)
        val_loss.append(vl)
    plt.plot(layers, loss)
    plt.plot(layers, val_loss)
    plt.xticks(layers)
    plt.legend(['loss', 'val_loss'])
    plt.title('Training and validation error by number of layers')
    plt.xlabel('Number of layers')
    plt.show()
