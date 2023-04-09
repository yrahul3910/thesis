import bayesmark.random_search as rs
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class SmoothnessOptimizer(AbstractOptimizer):
    primary_import = None
    N_EVALS = 30

    def __init__(self, api_config):
        """Build wrapper class to use smoothness optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

    def _get_smoothness(self, guess):
        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        model = Sequential()
        model.add(Dense(guess['hidden_layer_sizes'], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, batch_size=guess['batch_size'])

        func = K.function([model.layers[0].input], [model.layers[-2].output])
        batch_size = guess['batch_size']
        Kz = 0.
        Kw = 0.
        for i in range((len(X_train) - 1) // batch_size + 1):
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb = X_train[start_i:end_i]

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

    def suggest(self, n_suggestions=1):
        """Get suggestion.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        guesses = rs.suggest_dict([], [], self.api_config, n_suggestions=max(
            self.N_EVALS, n_suggestions * 3))
        betas = [self._get_smoothness(guess) for guess in guesses]

        return [guesses[i] for i in sorted(range(len(betas)), key=lambda x: betas[x])[:n_suggestions]]

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        pass


if __name__ == "__main__":
    experiment_main(SmoothnessOptimizer)