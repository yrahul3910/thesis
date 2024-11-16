# Thesis

This repo will act as the definitive source of code for my PhD thesis. It will be updated as the work progresses. This is NOT the only source of code, however. It is merely a part of it, and anything here should (in theory) be up-to-date.

## File descriptions

* `defect-prediction`: Defect prediction ablation study for GHOST-v2
* `email`: Wrapper around Amazon SES to send email notifications
* `issue_close_time`: Issue lifetime prediction ablation study for GHOST-v2
* `smoothness-hpo`: Experiments on using beta-smoothness as a heuristic for HPO.
* `static-code-warnings`: I believe this is the code for beta-smoothness HPO (and random/BOHB) on static code data.

## Recommended Links

* [GHOST Materials](https://github.com/yrahul3910/ghost-materials): Contains some materials relating to GHOST-v2. Specifically, it seems to contain some images from the paper.
* [TSE Paper Code](https://github.com/yrahul3910/ghost-dl): This is the code shared in the TSE paper. It's a little cleaned up version of the original code.
* [DL4SE](https://github.com/yrahul3910/dl4se): The DL4SE repo is a more general repo that tests GHOST-v2 on multiple tasks. It is known that GHOST-v2 is the SOTA for multiple tasks in this repo.
* [GHOST paper](https://arxiv.org/pdf/2008.03835.pdf): The original GHOST paper.
* [Progressive GHOST repo](https://github.com/yrahul3910/progressive-ghost): Seems to be the code for ablation studies.
* [HPO based on strong convexity](https://github.com/yrahul3910/strong-convexity): An extension of the smoothness-based HPO work here.
* [Other, one-off experiments](https://github.com/yrahul3910/yedida-gone-nuts/blob/master/Landscape%20analysis.ipynb): This is more of a prototyping repo, but it has some important results (like the [win rate vs smoothness figure in the SMOOTHIE paper](https://github.com/yrahul3910/yedida-gone-nuts/blob/master/Landscape%20analysis.ipynb)).
