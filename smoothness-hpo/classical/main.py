from typing import Tuple

import numpy as np
from raise_utils.transforms.wfo import fuzz_data
from raise_utils.learners import Autoencoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from smoothness.configs import config_space, config_space_hebo, learner_configs, learner_configs_hebo
from smoothness.hpo import SmoothnessHPO, RandomHPO, HeboHPO
from smoothness.data import load_issue_lifetime_prediction_data, remove_labels_legacy, remove_labels
from smoothness.hpo.util import get_learner


def data_fn(config: dict) -> Tuple[np.array, np.array, np.array, np.array]:
    n_class = 2
    x_train, x_test, y_train, y_test = load_issue_lifetime_prediction_data('eclipse', n_class)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if config['smooth']:
        print('[get_model] Running smooth')
        if n_class == 2:
            x_train, y_train = remove_labels(x_train, y_train)
        else:
            x_train, y_train = remove_labels_legacy(x_train, y_train)
        print('[get_model] Finished running smooth')

    if config['ultrasample']:
        # Apply WFO
        print('[get_model] Running ultrasample:wfo')
        x_train, y_train = fuzz_data(x_train, y_train)
        print('[get_model] Finished running ultrasample:wfo')

        # Reverse labels
        y_train = 1. - y_train
        y_test = 1. - y_test

        # Autoencode the inputs
        loss = 1e4
        while loss > 1e3:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
            ae.set_data(x_train, y_train, x_test, y_test)
            print('[get_model] Fitting autoencoder')
            ae.fit()
            print('[get_model] Fit autoencoder')

            loss = ae.model.history.history['loss'][-1]

        x_train = np.array(ae.encode(x_train))
        x_test = np.array(ae.encode(x_test))

    if config['wfo']:
        print('[get_model] Running wfo')
        x_train, y_train = fuzz_data(x_train, y_train)
        print('[get_model] Finished running wfo')

    if config['smote']:
        if n_class > 2 and len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)

        smote = SMOTE()
        x_train, y_train = smote.fit_resample(x_train, y_train)

        if n_class > 2 and len(y_train.shape) == 1:
            y_train = to_categorical(y_train, n_class)

    return x_train, x_test, y_train, y_test


def query_fn(config: dict):
    x_train, x_test, y_train, y_test = data_fn(config)
    learner = get_learner('nb', config)
    learner.fit(x_train, y_train)
    preds = learner.predict(x_train)

    return accuracy_score(y_train, preds)


def run_hpo(name: str, learner: str):
    if name == 'smoothness':
        hpo_space = config_space | learner_configs[learner]

        hpo = SmoothnessHPO(hpo_space, learner, query_fn, data_fn)
        scores, time = hpo.run(1, 30)

        print(f'Accuracy: {np.median(scores)}\nTime: {time}')
    elif name == 'random':
        hpo_space = config_space | learner_configs[learner]

        hpo = RandomHPO(hpo_space, learner, query_fn)
        scores, time = hpo.run(1, 5)

        print(f'Accuracy: {np.median(scores)}\nTime: {time}')
    elif name == 'hebo':
        hpo_space = config_space_hebo | learner_configs_hebo[learner]

        hpo = HeboHPO(hpo_space, learner, query_fn)
        scores, time = hpo.run(1, 5)

        print(f'Accuracy: {np.median(scores)}\nTime: {time}')
    else:
        raise ValueError(f'Unknown HPO method: {name}')


if __name__ == '__main__':
    learner = 'nb'
    hpo_method = 'smoothness'

    run_hpo(hpo_method, learner)
