import numpy as np
import pandas as pd

from numpy import array
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


def run_stats(data):
    print('----\nKruskal-Wallis\n----')
    # Perform Kruskal-Wallis test
    _, p_value = stats.kruskal(*data.values())
    print(f"Kruskal-Wallis test p-value: {p_value}")

    # Calculate medians for each group
    group_medians = {key: np.mean(val) for key, val in data.items()}
    print(f"Group means: {group_medians}")

    # Find the group with the largest median
    max_group = max(group_medians, key=group_medians.get)
    print(f"Group with the largest mean: {max_group}")

    # Perform pairwise Mann-Whitney U tests
    groups = list(data.keys())
    num_groups = len(groups)
    p_values = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        for j in range(i+1, num_groups):
            _, p = stats.mannwhitneyu(
                data[groups[i]], data[groups[j]], alternative='two-sided')
            p_values[i, j] = p
            p_values[j, i] = p

    print('Pairwise Mann-Whitney U tests')
    print(pd.DataFrame(p_values, index=groups, columns=groups))
    print()

    # Apply Bonferroni correction for multiple comparisons
    adjusted_p_values = multipletests(p_values.ravel(), method='fdr_tsbh')[
        1].reshape(p_values.shape)
    post_hoc = pd.DataFrame(
        adjusted_p_values, index=groups, columns=groups)

    print("Pairwise Mann-Whitney U tests with Benjamini/Hochberg correction:")
    print(post_hoc)
    print()

    # Check if the group with the largest median is significantly better than the others
    significantly_better = True
    for key in data.keys():
        if key != max_group and post_hoc.loc[max_group, key] >= 0.05:
            significantly_better = False
            break

    if significantly_better:
        print(
            f"The group '{max_group}' with the largest median IS significantly better than the others, ", end='')
    else:
        print(
            "The group with the largest median IS NOT significantly better than all the others, ", end='')

    print(
        f'and the highest p-value is {round(max(post_hoc[max_group]), 2)}')


data = {'hyperopt_0.2.7_a376313': array([86.97215521, 94.29631864, 96.24715772, 93.09259589, 90.78968848,
                                         84.02253948, 91.16800172, 92.06725999, 89.52826242, 89.62518519,
                                         92.46976582, 88.27523757, 94.6869087, 91.26896448, 93.5864723,
                                         83.16404552, 90.94390462, 94.46910496, 95.24634937, 90.51071532]), 'opentuner_0.8.8_a376313': array([87.98809536, 91.12169546, 84.40151032, 95.84119527, 96.29572988,
                                                                                                                                              83.55853422, 82.5941117, 90.00043616, 96.80582769, 87.29473341,
                                                                                                                                              84.79796036, 82.77485463, 92.29240111, 96.06342264, 90.3252549,
                                                                                                                                              80.66746515, 96.98926642, 86.92651125, 90.10480772, 79.4111455]), 'random_0.0.8_a376313': array([87.45613982, 87.64045654, 82.97701173, 85.51002193, 89.06690229,
                                                                                                                                                                                                                                               92.29217223, 82.15358353, 88.33014121, 84.05948234, 84.44127419,
                                                                                                                                                                                                                                               88.51047677, 84.70293701, 90.07691331, 85.76774256, 74.48980287,
                                                                                                                                                                                                                                               89.47368864, 96.54398594, 81.76314449, 84.51091525, 83.58468681]), 'skopt_0.9.0_a376313': array([88.17599035, 96.40597656, 88.00367731, 94.59119263, 89.64586735,
                                                                                                                                                                                                                                                                                                                                                92.73342882, 80.2848912, 96.35943832, 94.12395503, 83.16387649,
                                                                                                                                                                                                                                                                                                                                                90.87258968, 90.26979963, 88.7079897, 85.31428925, 82.70215164,
                                                                                                                                                                                                                                                                                                                                                85.66286833, 88.01744626, 89.67287881, 75.85971482, 81.87206081]), 'turbo_0.0.1_a376313': array([95.74484296, 95.75620197, 97.71479763, 94.7187575, 95.66357488,
                                                                                                                                                                                                                                                                                                                                                                                                                                                 96.77126127, 96.81486619, 98.09628087, 87.69646834, 83.13735138,
                                                                                                                                                                                                                                                                                                                                                                                                                                                 94.67190275, 87.79797811, 97.12023798, 97.3471074, 94.27459579,
                                                                                                                                                                                                                                                                                                                                                                                                                                                 93.94531442, 96.16855241, 89.7978168, 99.68951768, 92.1532862]),
        'smoothness_x.x.x_a376313': array([94.29033559,  96.5322596,  94.42158185,  99.92171909,
                                          91.4674189,  90.49164818,  99.30051347, 100.03160003,
                                          92.8068035,  79.72424956,  97.73751657,  90.71132963,
                                          99.46403341, 101.56742754,  96.23008724,  89.17337364,
                                          93.14830308,  75.57467461,  92.70223896,  91.46021962])}

run_stats(data)
