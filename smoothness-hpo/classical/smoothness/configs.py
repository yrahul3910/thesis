config_space = {
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}

config_space_hebo = [
    {'name': 'preprocessor', 'type': 'cat', 'categories': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust']},
    {'name': 'wfo', 'type': 'bool'},
    {'name': 'smote', 'type': 'bool'},
    {'name': 'ultrasample', 'type': 'bool'},
    {'name': 'smooth', 'type': 'bool'},
]

learner_configs = {
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2', None],
    },
    'logistic': {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.1, 1, 10],
    },
    'nb': {}
}

learner_configs_hebo = {
    'tree': [
        {'name': 'criterion', 'type': 'cat', 'categories': ['gini', 'entropy']},
        {'name': 'max_depth', 'type': 'cat', 'categories': [None, 5, 10]},
        {'name': 'min_samples_split', 'type': 'cat', 'categories': [2, 5, 10]},
        {'name': 'min_samples_leaf', 'type': 'cat', 'categories': [1, 5, 10]},
        {'name': 'max_features', 'type': 'cat', 'categories': ['sqrt', 'log2', None]},
    ],
    'logistic': [
        {'name': 'penalty', 'type': 'cat', 'categories': ['l1', 'l2', 'elasticnet']},
        {'name': 'C', 'type': 'cat', 'categories': [0.1, 1, 10]},
    ],
    'nb': []
}
