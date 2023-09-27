## Files:

* `data/`: Datasets
* `results/`: Results
* `config.py`: Config dataclass, part of code
* `remove_labels.*`: Cython implementation of SMOOTHing operator (see TSE paper with David Lo on static code warnings)
* `Stats.ipynb`: Statistical tests. Not comprehensive, more of a playground notebook
* `steps_hpo.py`: HPO using random or BOHB. Used to compare to smoothness results.
* `Untitled.ipynb`: Notebook that tests the correlation between the smoothness and number of steps needed for convergence. This isn't very useful anymore, since I have now established that since max min f <= min max f, setting f = norm(hessian) means that max (strong convexity parameter) <= min smoothness, so that the smoothness is merely a loose upper bound on the strong convexity parameter.
* `util.py`: Some functions used in the code.