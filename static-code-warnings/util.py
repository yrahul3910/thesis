import pyximport
import random
import numpy as np
from typing import List
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from raise_utils.transforms import Transform
from raise_utils.learners import Autoencoder
from raise_utils.data import Data
from config import Config

pyximport.install()

from remove_labels import remove_labels


options = {
    'n_units': (2, 6),
    'n_layers': (2, 5),
    'transform': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}

def get_random_hyperparams(options: dict) -> Config:
    """
    Get hyperparameters from options.
    """
    hyperparams = Config()
    for key, value in options.items():
        if isinstance(value, list):
            setattr(hyperparams, key, random.choice(value))
        elif isinstance(value, tuple):
            setattr(hyperparams, key, random.randint(value[0], value[1]))
    return hyperparams


def get_many_random_hyperparams(n: int) -> List[Config]:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams


def get_model(data: Data, config: Config) -> tuple:
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    """
    transform = Transform(config.transform)
    transform.apply(data)

    if config.smooth:
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)

        print('[get_model] Running smooth')
        data.x_train, data.y_train = remove_labels(data.x_train, data.y_train)
        print('[get_model] Finished running smooth')

    if config.ultrasample:
        # Apply WFO
        print('[get_model] Running ultrasample:wfo')
        transform = Transform('wfo')
        transform.apply(data)
        print('[get_model] Finished running ultrasample:wfo')

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5, verbose=0)
        ae.set_data(*data)

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    learner = Sequential()
    data.y_train = data.y_train.astype('float32')

    data.x_train, data.x_test, data.y_train, data.y_test = \
        map(np.array, (data.x_train, data.x_test, data.y_train, data.y_test))
    data.y_train = data.y_train.squeeze()
    data.y_test = data.y_test.squeeze()

    if config.wfo:
        print('[get_model] Running wfo')
        transform = Transform('wfo')
        transform.apply(data)
        print('[get_model] Finished running wfo')
    
    if config.smote:
        print('[get_model] Running smote')
        try:
            transform = Transform('smote')
            transform.apply(data)
        except:
            print('[get_model] Failed running smote')

        print('[get_model] Finished running smote')
    
    for _ in range(config.n_layers):
        learner.add(Dense(config.n_units, activation='relu'))
    
    learner.add(Dense(1, activation='sigmoid'))
    learner.compile(loss='binary_crossentropy', optimizer='sgd')

    return learner, data


def get_smoothness(data: Data, config: Config) -> float:
    print('[get_smoothness] Getting model')
    model, data = get_model(data, config)
    print('[get_smoothness] Got model')

    # Fit for one epoch before computing smoothness
    model.fit(data.x_train, data.y_train, epochs=1, verbose=1, batch_size=128)
    print('[get_smoothness] Fit model')

    func = K.function([model.layers[0].input], [model.layers[-2].output])
    batch_size = 128
    Kz = 0.
    Kw = 0.
    for i in range((len(data.x_train) - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = data.x_train[start_i:end_i]

        activ = np.linalg.norm(func([xb]))
        if activ > Kz:
            Kz = activ
        
        assert len(model.layers[-1].weights[0].shape) == 2
        W = np.linalg.norm(model.layers[-1].weights[0])
        if W > Kw:
            Kw = W

    if Kw == 0:
        return 0.

    return Kz / Kw


def get_best_results(results: list) -> list:
    """
    Get the best results from a list of results.
    """
    return sorted(results, key=lambda x: (x[0] - x[1], x[-1]), reverse=True)[0]