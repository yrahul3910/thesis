# Black Box Optimization Challenge

This directory is a clone of the [official BBO Challenge starter kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit), which follows the Apache license.

The `experiment_analysis.py` file is adapted from the Bayesmark code to print out the scores from all runs (as opposed to just a mean). This makes statistical analysis easier. Run it as follows:

```
python3 experiment_analysis.py -dir results/[dataset] -b run_XXX
```

where `run_XXX` is the *latest* directory with the name `run_XXX` inside `results/[dataset]`.
