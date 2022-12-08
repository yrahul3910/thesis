# Defect prediction

This code borrows from the sources listed in the main repo folder. Importantly, this repo does NOT achieve the results in the paper. For example, in Table 7 of the paper (Wang et al. datasets), the original paper optimized for recall. I believe that in that original paper, Table 5 was optimized to maximize pd - pf. This repo maximized for F1-score instead.

The important file here is `parse.pysh`, which does the parsing. Specifically, from each of the runs, it groups by dataset, extracts the best F1-score, and then shows all rows with the best performer.