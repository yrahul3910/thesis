from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bohb import BOHB
import bohb.configspace as bohb_space


class BohbOptimizer(AbstractOptimizer):
    primary_import = None

    def __init__(self, api_config):
        """Build wrapper class to use smoothness optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.space = []

        for k, v in api_config.items():
            if v["type"] == "int":
                is_log = v["space"] in ["log", "logit"]
                self.space.append(bohb_space.IntegerUniformHyperparameter(k, *v["range"], log=is_log))
            elif v["type"] in ["cat", "ordinal"]:
                self.space.append(bohb_space.CategoricalHyperparameter(k, v["values"]))
            elif v["type"] == "bool":
                self.space.append(bohb_space.CategoricalHyperparameter(k, [True, False]))
            elif v["type"] == "real":
                is_log = v["space"] in ["log", "logit"]
                self.space.append(bohb_space.UniformHyperparameter(k, *v["range"], log=is_log))

        self.observations = []
        self.opt = BOHB(configspace=bohb_space.ConfigurationSpace(self.space), evaluate=self.dummy_f, min_budget=1, max_budget=1)
    
    def dummy_f(self, x):
        if x in self.observations:
            return self.observations[x]
        else:
            return 100.
    
    def suggest(self, n_suggestions=1):
        logs = self.opt.optimize()
        return [logs.best["hyperparameter"].to_dict()]

    def observe(self, X, y):
        self.observations.append((X, y))
        pass


if __name__ == "__main__":
    experiment_main(BohbOptimizer)
