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


data = {'hebo_0.0.8_a376313': array([85.15640212, 76.39730733, 80.26210251, 90.96728588, 69.37354506,
                                     88.75614311, 77.27961615, 62.41869553, 74.4527658, 78.09455929,
                                     86.62357658, 77.50816573, 67.90558363, 64.1261061, 80.76586342,
                                     81.64359936, 82.8047499, 90.68465153, 69.75461382, 80.3726458]), 'hyperopt_0.2.7_a376313': array([72.78788862, 87.99597194, 88.52352965, 77.38984709, 76.79142525,
                                                                                                                                       79.25226805, 90.39505081, 81.92103435, 81.61613837, 84.00128576,
                                                                                                                                       70.04421442, 69.60099832, 84.37092901, 72.60252225, 80.77620232,
                                                                                                                                       81.60809914, 60.88830493, 83.88957442, 83.50012314, 88.68026458]), 'opentuner_0.8.8_a376313': array([85.88022458, 89.85540067, 84.98219293, 83.24802092, 86.19632436,
                                                                                                                                                                                                                                            80.90391663, 85.15364859, 83.87449831, 90.76414794, 84.523854,
                                                                                                                                                                                                                                            88.06572258, 81.2846965, 85.45940546, 83.11290897, 82.76547689,
                                                                                                                                                                                                                                            83.23302353, 82.84234552, 84.8304721, 88.7294765, 85.53876776]), 'random-search_0.0.8_a376313': array([86.48899665, 86.91839266, 82.90547272, 81.51526174, 91.77827884,
                                                                                                                                                                                                                                                                                                                                                   80.11257905, 84.73171943, 78.16773736, 80.1344087, 94.27981848,
                                                                                                                                                                                                                                                                                                                                                   87.27684665, 86.71497998, 50.4464371, 84.61819423, 84.88361934,
                                                                                                                                                                                                                                                                                                                                                   86.5354261, 84.78396192, 82.38035937, 82.42013851, 82.58972412]), 'skopt_0.9.0_a376313': array([76.14891631, 78.64738673, 80.7951811, 83.57500323, 83.37023189,
                                                                                                                                                                                                                                                                                                                                                                                                                                                   67.94223705, 77.96187314, 78.17951772, 73.59441544, 87.69555711,
                                                                                                                                                                                                                                                                                                                                                                                                                                                   88.77905371, 85.16499487, 88.09492386, 43.09829698, 82.63718845,
                                                                                                                                                                                                                                                                                                                                                                                                                                                   82.46294019, 82.14727316, 84.68087222, 78.36003304, 83.04849762]), 'smoothness_x.x.x_a376313': array([90.59065702, 86.24129188, 89.30774262, 88.91324693, 79.88460137,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         87.4278943, 89.60913376, 87.83423204, 86.26734203, 84.30085692,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         94.858568, 90.04147841, 86.7514705, 90.54419355, 88.21397075,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         75.3087184, 95.02053325, 83.02501076, 84.67152289, 94.07133543]), 'turbo_0.0.1_a376313': array([94.86787664, 84.59689859, 73.68948016, 89.97753607, 84.23669273,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         78.53972931, 85.48620218, 80.44861631, 86.04056561, 75.38072875,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         86.26801135, 83.15442782, 83.20546933, 84.38745363, 82.68257894,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         79.80286173, 82.8965522, 86.08468931, 86.99119115, 81.59833169]), 'stcvx_x.x.x_a376313': array([91.9459061, 91.97886225, 81.5491569, 87.17445036, 93.10909964,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         89.74396181, 91.81771952, 88.54139499, 92.16518842, 96.50414433,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         87.9950813, 93.77602647, 94.52389575, 96.07843137, 90.57384622,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         96.61097867, 92.99887755, 87.68087612, 89.24831833, 85.83592108])}
run_stats(data)
