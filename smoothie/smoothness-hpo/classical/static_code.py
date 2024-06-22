from typing import Tuple
import os

import numpy as np
import pandas as pd
from raise_utils.data import Data
from raise_utils.transforms.wfo import fuzz_data
from raise_utils.transforms import Transform
from raise_utils.learners import Autoencoder
from raise_utils.metrics import ClassificationMetrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from smoothness.configs import learner_configs
from smoothness.hpo.smoothness import SmoothnessHPO
from smoothness.data import remove_labels
from smoothness.hpo.util import get_learner

import ses


datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr', 'maven', 'tomcat']

config_space = {
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}
learner_name = 'nb'
base_path = './data/static_code/'


def load_data(dataset: str, config: dict) -> Tuple[np.array, np.array, np.array, np.array]:
    train_file = base_path + 'train/' + dataset + '_B_features.csv'
    test_file = base_path + 'test/' + dataset + '_C_features.csv'

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat((train_df, test_df), join='inner')

    X = df.drop('category', axis=1)
    y = df['category']

    y[y == 'close'] = 1
    y[y == 'open'] = 0

    y = np.array(y, dtype=np.float32)

    X = X.select_dtypes(
        exclude=['object']).astype(np.float32)

    if dataset == 'maven':
        data = Data(*train_test_split(X, y, test_size=.5, shuffle=False))
    else:
        data = Data(*train_test_split(X, y, test_size=.2, shuffle=False))

    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    transform = Transform(config['preprocessor'])
    transform.apply(data)

    x_train, x_test, y_train, y_test = data.x_train, data.x_test, data.y_train, data.y_test

    if config['smooth']:
        print('[get_model] Running smooth')
        x_train, y_train = remove_labels(x_train, y_train)
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
        ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
        ae.set_data(x_train, y_train, x_test, y_test)
        ae.fit()

        x_train = np.array(ae.encode(x_train))
        x_test = np.array(ae.encode(x_test))

    if config['wfo']:
        print('[get_model] Running wfo')
        x_train, y_train = fuzz_data(x_train, y_train)
        print('[get_model] Finished running wfo')

    if config['smote']:
        smote = SMOTE()
        x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test


def query_fn(dataset: str, config: dict, seed: int = 42, budget: int = 100):
    x_train, x_test, y_train, y_test = load_data(dataset, config)

    # Comment the below if statements for MulticlassDL.
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    learner = get_learner(learner_name, config)
    learner.fit(x_train, y_train)
    preds = learner.predict(x_test)

    metrics = ClassificationMetrics(y_test, preds)
    metrics.add_metrics(['pd', 'pf', 'prec', 'auc'])

    return metrics.get_metrics()


n_jobs = 20
if __name__ == '__main__':
    hpo_space = {**config_space, **learner_configs[learner_name]}

    for dataset in datasets:
        q_fn = lambda config, seed, budget: query_fn(dataset, config, seed, budget)
        d_fn = lambda config: load_data(dataset, config)
        hpo = SmoothnessHPO(hpo_space, learner_name, q_fn, d_fn)

        try:
            scores, time = hpo.run(20, 10)
            ses.send_email('gcloud Success Notification', f'Run completed.')

            print(f'Accuracy: {np.median(scores)}\nTime: {time}')
        except:
            ses.send_email('ARC Failure Notification', f'Run failed.')
            n_jobs -= 1
