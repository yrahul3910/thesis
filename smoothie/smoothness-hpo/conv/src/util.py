import random
from typing import List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization

from src.data import Dataset
from src.config import Config

BATCH_SIZE=256

def run_experiment(data: Dataset, config: Config, n_class: int = 10) -> float:
    print('[run_experiment] Getting model')
    model = get_model(data, config, n_class)

    print('[run_experiment] Got model')
    model.fit(data.x_train, data.y_train, epochs=50, verbose=1, batch_size=BATCH_SIZE)
    print('[run_experiment] Fit model')

    y_pred = np.argmax(model.predict(data.x_test), axis=-1)
    
    if len(data.y_test.shape) > 1:
        data.y_test = np.argmax(data.y_test, axis=1)

    return accuracy_score(data.y_test, y_pred)


def get_smoothness(data: Dataset, config: Config, n_class: int = 10) -> float:
    model = get_model(data, config, n_class)

    if n_class > 2 and len(data.y_train.shape) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)

    # Fit for one epoch before computing smoothness
    model.fit(data.x_train, data.y_train, batch_size=BATCH_SIZE, epochs=1),

    Ka_func = K.function([model.layers[0].input], [model.layers[-2].output])

    batch_size = BATCH_SIZE
    best_mu = np.inf
    for i in range((len(data.x_train) - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = data.x_train[start_i:end_i]

        mu = np.linalg.norm(Ka_func([xb])) / np.linalg.norm(model.layers[-1].weights[0])
        if mu < best_mu and mu != np.inf:
            best_mu = mu

    return best_mu


def get_random_hyperparams(options: dict) -> Config:
    """
    Get hyperparameters from options.
    """
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            hyperparams[key] = random.randint(value[0], value[1])
    return Config(**hyperparams)


def get_many_random_hyperparams(options: dict, n: int) -> list:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams


def get_model(data: Dataset, config: Config, n_class: int = 10) -> Sequential:
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    :param {int} n_class - The number of classes in the dataset.
    """
    learner = Sequential()

    for i in range(config.n_blocks):
        learner.add(Conv2D(config.n_filters, config.kernel_size, padding=config.padding, kernel_initializer='he_uniform', activation='relu'))
        learner.add(MaxPooling2D(pool_size=(2, 2)))

    learner.add(Flatten())
    learner.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    learner.add(Dense(n_class, activation='softmax'))
    
    learner.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return learner
