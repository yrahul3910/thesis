import pyximport
import numpy as np
import pandas as pd
import os
import random
from raise_utils.learners import FeedforwardDL, Autoencoder
from raise_utils.data import Data
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from sklearn.model_selection import train_test_split
from copy import deepcopy
from config import Config
from pprint import pprint
from util import get_smoothness, get_many_random_hyperparams, get_best_results

pyximport.install()

from remove_labels import remove_labels


datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr', 'maven', 'tomcat']
base_path = './data/'


def load_data(dataset: str) -> Data:
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

    return data


def run(data: Data, config: Config):
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {Config} config - The config to use. Must be one in the format used in `process_configs`.
    """
    if config.smooth:
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)
        data.x_train, data.y_train = remove_labels(data.x_train, data.y_train)

    if config.ultrasample:
        # Apply WFO
        transform = Transform('wfo')
        transform.apply(data)

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
        ae.set_data(*data)
        ae.fit()

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    learner = FeedforwardDL(n_layers=config.n_layers, n_units=config.n_units, 
                            weighted=config.weighted, wfo=config.wfo,
                            smote=config.smote, n_epochs=100)

    learner.set_data(*data)
    learner.fit()

    # Get the results.
    preds = learner.predict(data.x_test)
    m = ClassificationMetrics(data.y_test, preds)
    m.add_metrics(['pd', 'pf', 'prec', 'auc'])
    results = m.get_metrics()
    return results


def run_experiment(dataset: str, config: Config):
    '''
    Runs a specific config on a certain file.

    :param {str} filename - The filename. Must not include a path.
    :param {Config} config - The name of the config. Must be in `process_configs`.
    '''
    name = f'{dataset}'
    name += f'-{config.wfo}-{config.ultrasample}-{config.smote}-{config.smooth}'
    name += f'-{os.environ["SLURM_JOB_ID"] or "local"}'

    data = load_data(dataset)
    return run(data, config)


def run_all_experiments():
    """
    Runs all experiments 10 times each.
    """
    # DODGE needs a directory called `./ghost-log/`, or `./ghost-log-wang/` depending
    # on the datasets used.
    if 'ghost-log' not in os.listdir('.'):
        os.mkdir('./ghost-log')

    file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
    file = open(f'runs-{file_number}.txt', 'a')

    keep_configs = 10
    num_configs = 50

    for dataset in datasets:
        print(f'{dataset}:', file=file)
        data_orig = load_data(dataset)

        # 20 repeats
        results = []
        for _ in range(20):
            best_betas = []
            best_configs = []
            configs = get_many_random_hyperparams(num_configs)
            repeat_results = []
            for config in configs:
                data = deepcopy(data_orig)
                print('[main] Computing smoothness for config:', config)
                smoothness = get_smoothness(data, config)
                print(f'Config: {config}\nSmoothness: {smoothness}', file=file)
                file.flush()

                if len(best_betas) < keep_configs or smoothness > min(best_betas):
                    best_betas.append(smoothness)
                    best_configs.append(config)

                    best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs), reverse=True, key=lambda x: x[0]))
                    best_betas = list(best_betas[:keep_configs])
                    best_configs = list(best_configs[:keep_configs])
            
            for beta, config in zip(best_betas, best_configs):
                data = deepcopy(data_orig)
                print(f'Config: {config}\nbeta: {beta}', file=file)

                cur_results = run(data, config)
                print('[main] Accuracy:', cur_results, file=file)
                file.flush()

                # Find the best of these
                repeat_results.append(cur_results)

            results.append(get_best_results(repeat_results))
            print('', file=file)
        
        print('Median results:', file=file)
        print(np.median(results, axis=0), file=file)
        print('', file=file)
        print('Best results:', file=file)
        pprint(results, stream=file)

    print('Done.')


if __name__ == '__main__':
    run_all_experiments()
