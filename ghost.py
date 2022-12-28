from raise_utils.learners import FeedforwardDL, Autoencoder
from raise_utils.hyperparams import DODGE
from raise_utils.transform import Transform
from raise_utils.data import DataLoader
import os
from keras import backend as K
import numpy as np

data = DataLoader.from_files(base_path='./raise/promise/', 
                                           files=['camel-1.4.csv', 'ant-1.6.csv'])
transform = Transform('wfo')
transform.apply(data)

data.y_train = 1.-data.y_train
data.y_test = 1.-data.y_test

autoencoder = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5, n_epochs=750)
autoencoder.set_data(*data)
autoencoder.fit()

data.x_train = autoencoder.encode(K.constant(np.array(data.x_train)))
data.x_test = autoencoder.encode(K.constant(np.array(data.x_test)))

config = {
            'n_runs': 10,
            'data': [data],
            'metrics': ['f1', 'pd', 'prec'],
            'learners': [],
            'log_path': './ghost-log-defect',
            'transforms': ['standardize', 'normalize', 'minmax'] * 30,
            'random': True,
            'name': 'camel-ant'
}

for i in range(30):
        config['learners'].append(
                        FeedforwardDL(weighted=True, wfo=True, smote=True, random={'n_units': (2, 6), 'n_layers': (2, 5)}, n_epochs=100)
                            )

dodge = DODGE(config)
dodge.optimize()
