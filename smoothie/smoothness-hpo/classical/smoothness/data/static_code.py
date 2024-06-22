import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from raise_utils.data import Data


base_path = './data/static_code/'


def load_static_code_data(dataset: str) -> Data:
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