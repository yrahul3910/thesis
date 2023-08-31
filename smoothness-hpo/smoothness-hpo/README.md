## File descriptions

* `defect_prediction.py`: Defect prediction code for feedforward networks.
* `issue_lifetime_prediction.py`: Issue lifetime prediction code for feedforward networks.
* `Kruskal-Wallis test.ipynb`: Code for running the Kruskal-Wallis test and pairwise Mann-Whitney U-tests.
* `bbo_challenge_starter_kit/`: Feedforward network code for the NeurIPS 2020 BBO Challenge.
* `classical/`: Code for classical learners
* `data/`: Link to datasets.
* `results/`: Feedforward network results.
* `smoothness-approximation/`: Experiments to test if a subset of the data can be used to compute $\beta-
smoothness accurately (Spoiler: no).
* `src/`: Code for computing the $\beta-$smoothness.

## Compiling Cython code

Part of the code is written in Cython. To compile it,

```
cython remove_labels.pyx
gcc -O2 -Wall `python3.11-config --include` -o remove_labels.so remove_labels.c
```

Alternatively,

```
cythonize -i remove_labels.pyx
```

