import numpy as np
import pandas as pd
import os
import random
from raise_utils.learners import FeedforwardDL, Autoencoder, MulticlassDL
from raise_utils.data import Data
from raise_utils.hyperparams import DODGE
from raise_utils.interpret import DODGEInterpreter
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from raise_utils.hooks import Hook
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
from scipy.stats import mode
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


top1 = []
top2 = []
file = None
def hook(model, x_test, y_test):
    m = ClassificationMetrics(y_test, model.predict(x_test))
    m.add_metric('accuracy')
    t1 = m.get_metrics()[0]
    print("Top-1 Accuracy =", t1, file=file)
    top1.append(t1)

    data = Data(None, x_test, None, y_test)
    t2 = get_top2(model, data)
    print("Top-2 Accuracy =", t2, file=file)
    top2.append(t2)


def split_data(filename: str, data: Data, n_classes: int):
    if n_classes == 2:
        if filename == 'firefox.csv':
            data.y_train = data.y_train < 4
            data.y_test = data.y_test < 4
        elif filename == 'chromium.csv':
            data.y_train = data.y_train < 5
            data.y_test = data.y_test < 5
        else:
            data.y_train = data.y_train < 6
            data.y_test = data.y_test < 6
    elif n_classes == 3:
        data.y_train = np.where(data.y_train < 2, 0,
                                np.where(data.y_train < 6, 1, 2))
        data.y_test = np.where(
            data.y_test < 2, 0, np.where(data.y_test < 6, 1, 2))
    elif n_classes == 5:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 3, 1, np.where(
            data.y_train < 6, 2, np.where(data.y_train < 21, 3, 4))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 3, 1, np.where(
            data.y_test < 6, 2, np.where(data.y_test < 21, 3, 4))))
    elif n_classes == 7:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(
            data.y_train < 6, 3, np.where(data.y_train < 11, 4, np.where(data.y_train < 21, 5, 6))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(
            data.y_test < 6, 3, np.where(data.y_test < 11, 4, np.where(data.y_test < 21, 5, 6))))))
    else:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(data.y_train < 4, 3, np.where(
            data.y_train < 6, 4, np.where(data.y_train < 8, 5, np.where(data.y_train < 11, 6, np.where(data.y_train < 21, 7, 8))))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(data.y_test < 4, 3, np.where(
            data.y_test < 6, 4, np.where(data.y_test < 8, 5, np.where(data.y_test < 11, 6, np.where(data.y_test < 21, 7, 8))))))))

    if n_classes > 2:
        data.y_train = to_categorical(data.y_train, num_classes=n_classes, dtype=int)
        data.y_test = to_categorical(data.y_test, num_classes=n_classes, dtype=int)

    return data


def load_data(filename: str, n_classes: int):
    df = pd.read_csv(f'./data/{filename}.csv')
    df.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)
    _df = df[['s1', 's2', 's3', 's4', 's5', 's6', 's8', 'y']]
    _df['s70'] = df['s7'].apply(lambda x: eval(x)[0])
    _df['s71'] = df['s7'].apply(lambda x: eval(x)[1])
    _df['s72'] = df['s7'].apply(lambda x: eval(x)[2])
    _df['s90'] = df['s9'].apply(lambda x: eval(x)[0])
    _df['s91'] = df['s9'].apply(lambda x: eval(x)[1])
    
    if filename == 'firefox':
        _df['s92'] = df['s9'].apply(lambda x: eval(x)[2])
    
    x = _df.drop('y', axis=1)
    y = _df['y']
    
    data = Data(*train_test_split(x, y))
    data = split_data(filename, data, n_classes)
    return data


def get_top2(model, data):
    y_test = np.array(data.y_test)
    preds = model.learner.model.predict(data.x_test)
    best_n = np.argsort(preds, axis=1)[:,-2:]
    correct = 0
    total = len(y_test)
    
    for i, pred in enumerate(best_n):
        if y_test[i] in pred:
            correct += 1
    return round(correct / total, 3)


def remove_labels(data):
    """
    Keep only sqrt(n) of the real labels. To find the others, use k=sqrt(sqrt(n))
    nearest neighbors from the labels we know, and use the mode.
    """
    # "Remove" labels
    lost_idx = np.random.choice(
        len(data.y_train), size=int(len(data.y_train) - np.sqrt(len(data.y_train))), replace=False)

    x_lost = data.x_train[lost_idx]
    x_rest = np.delete(data.x_train, lost_idx, axis=0)
    y_lost = data.y_train[lost_idx]
    y_rest = np.delete(data.y_train, lost_idx, axis=0)

    if len(x_lost.shape) == 1:
        x_lost = x_lost.reshape(1, -1)
    if len(x_rest.shape) == 1:
        x_rest = x_rest.reshape(1, -1)

    # Impute data
    for i in range(len(x_lost)):
        tree = KDTree(x_rest)
        d, idx = tree.query([x_lost[i]], k=int(np.sqrt(len(x_rest))), p=1)
        y_lost[i] = mode(y_rest[idx][0])[0][0]

    print('Ratio =', round(len(x_rest) / len(data.y_train), 2))
    print('Total =', len(x_lost) + len(x_rest))
    data.x_train = np.concatenate((x_lost, x_rest), axis=0)
    data.y_train = np.concatenate((y_lost, y_rest), axis=0)
    return data, 0.8 * len(x_rest) / (len(x_rest) + len(x_lost))


def run(data: Data, name: str, n_class: int, config: dict):
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    """
    if config.get('smooth', False):
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)
        data, ratio = remove_labels(data)

    if config.get('ultrasample', False):
        # Apply WFO
        transform = Transform('wfo')
        transform.apply(data)

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        loss = 1e4
        while loss > 1e3:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
            ae.set_data(*data)
            ae.fit()

            loss = ae.model.history.history['loss'][-1]

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    if config.get('dodge', False):
        # Tune the hyper-params
        dodge_config = {
            'n_runs': 1,
            'data': [data],
            'metrics': ['accuracy'],
            'learners': [],
            'log_path': './ghost-log/',
            'transforms': ['standardize', 'normalize', 'minmax', 'robust', 'maxabs'] * 30,
            'random': True,
            'name': name,
            'post_train_hooks': [Hook(name='top2', function=hook)]
        }

        for _ in range(30):
            wfo = config.get('wfo', True)
            smote = config.get('smote', True)
            weighted = config.get('weighted_loss', True)

            if n_class == 2:
                dodge_config['learners'].append(
                    FeedforwardDL(weighted=weighted, wfo=wfo, smote=smote,
                                random={'n_units': (
                                    2, 6), 'n_layers': (2, 5)},
                                n_epochs=100)
                )
            else:
                dodge_config['learners'].append(
                    MulticlassDL(wfo=wfo, n_classes=n_class, n_epochs=100,
                                random={'n_units': (2, 6), 'n_layers': (2, 5)})
                )

        dodge = DODGE(dodge_config)
        return dodge.optimize()[0]

    # Otherwise, it's one of the untuned approaches.
    elif config.get('wfo', False):
        if n_class == 2:
            learner = FeedforwardDL(weighted=True, wfo=True,
                                    smote=True, n_epochs=100)
        else:
            learner = MulticlassDL(wfo=True, n_classes=n_class, n_epochs=100)

        learner.set_data(*data)
        learner.fit()

    else:
        if n_class == 2:
            learner = FeedforwardDL(weighted=True, wfo=False,
                                    smote=False, n_epochs=100)
        else:
            learner = MulticlassDL(wfo=False, n_classes=n_class, n_epochs=100)

        learner.set_data(*data)
        learner.fit()

    # Get the results.
    preds = learner.predict(data.x_test)
    m = ClassificationMetrics(data.y_test, preds)
    m.add_metrics(['accuracy'])
    results = m.get_metrics()
    return results


def run_experiment(filename: str, n_class: int, config: dict, file):
    '''
    Runs a specific config on a certain file.

    :param {str} filename - The filename. Must not include a path.
    :param {dict} config_name - The name of the config. Must be in `process_configs`.
    '''
    global top1, top2

    name = f'{filename}-{n_class}'
    name += f'-{config["dodge"]}-{config["wfo"]}-{config["ultrasample"]}-{config["smote"]}-{config["smooth"]}'
    name += f'-{os.environ["SLURM_JOB_ID"] or "local"}'

    data = load_data(filename, n_class)
    top1 = []
    top2 = []

    return run(data, name, n_class, config)


def run_all_experiments():
    """
    Runs all experiments 10 times each.
    """
    global file

    # DODGE needs a directory called `./ghost-log/`, or `./ghost-log-wang/` depending
    # on the datasets used.
    if 'ghost-log' not in os.listdir('.'):
        os.mkdir('./ghost-log')

    total = 90
    pbar = tqdm(total=total)
    file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
    for filename in ['chromium', 'firefox', 'eclipse']:
        for n_class in [3, 5, 7, 9]:
            file = open(f'runs-{file_number}.txt', 'a')
            print(f'{filename}-{n_class}:', file=file)
            print('=' * len(f'{filename}-{n_class}:'), file=file)

            base_config = [True, True, True, True, True]
            for i in range(len(base_config) + 1):
                config = base_config[:]

                if i > 0:
                    config[i - 1] = False

                (dodge, wfo, smote, ultrasample, smooth) = config
                run_config = {
                    'dodge': dodge,
                    'wfo': wfo,
                    'smote': smote,
                    'ultrasample': ultrasample,
                    'smooth': smooth,
                    'weighted_loss': True
                }

                print('Running', config, file=file)

                result = run_experiment(filename, n_class, run_config, file)
                print(result, file=file)
                file.flush()
                pbar.update()

            print('', file=file)

    print('Done.')


if __name__ == '__main__':
    run_all_experiments()
