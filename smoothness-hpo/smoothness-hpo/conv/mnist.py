import os
import random

from copy import deepcopy
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from src.util import get_many_random_hyperparams, get_smoothness, run_experiment, get_random_hyperparams
from src.data import Dataset

hpo_space = {
    'n_filters': (2, 6),
    'kernel_size': (2, 6),
    'padding': ['valid', 'same'],
    'n_blocks': (1, 3)
}

file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
data_orig = Dataset(x_train, y_train, x_test, y_test)

# Run actual experiment
best_betas = []
best_configs = []
keep_configs = 5
num_configs = 30

configs = get_many_random_hyperparams(hpo_space, num_configs)

for i, config in enumerate(configs):
    try:
        data = deepcopy(data_orig)
        print(f'[main] ({i}/{num_configs}) Computing smoothness for config:', config)
        smoothness = get_smoothness(data, config)
        print(f'Config: {config}\nSmoothness: {smoothness}')

        if len(best_betas) < keep_configs or smoothness > min(best_betas):
            best_betas.append(smoothness)
            best_configs.append(config)

            best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs), reverse=True, key=lambda x: x[0]))
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])
    except:
        print(f'Error, skipping config')
    
for beta, config in zip(best_betas, best_configs):
    data = deepcopy(data_orig)
    print(f'Config: {config}\nbeta: {beta}')
    print('[main] Accuracy:', run_experiment(data, config, 10))
