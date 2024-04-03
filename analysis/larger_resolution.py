"""
This evaluation produces 4 line graphs comparing mega_mmap to alternative
implementations of KMeans, RandomForst, DBSCAN, and Gray Scott.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from jarvis_cd.basic.jarvis_manager import JarvisManager
from jarvis_cd.basic.pkg import Pipeline
import pandas as pd
import random
from matplotlib.patches import Patch

# Load data for spark + mega for kmeans
# kmeans_spark = Pipeline().load(
#     'mm_kmeans_spark', with_config=False)
# rf_spark = Pipeline().load(
#     'mm_rf_spark', with_config=False)
# gray_scott_mpi = Pipeline().load(
#     'mm_gray_scott_mpi', with_config=False)

def make_dataset(app_name, impl):
    df = []
    Ls = [2048, 2048+320, 2688, 2688+320, 3456]
    df_sizes = [(.0033 * L)**3 for L in Ls]
    mem_utils = [100 * min(df_size / 16 / 48*(1<<30), 1.0) for df_size in df_sizes]
    for L, df_size, mem_util in zip(Ls, df_sizes, mem_utils):
        for i in range(3):
            df.append({
                'nprocs': 16 * 48,
                'df_size': df_size,
                'mem_util': mem_util,
                'L': L,
                'algo': app_name,
                'impl': impl
            })
    return pd.DataFrame(df)

# Gray Scott CSVs
gray_scott_df = make_dataset('gray_scott', 'mega')
gray_scott_df.to_csv('csv/large_resolution_r/gray_scott_mega.csv')

# class Tiering:
#     def __init__(self):
#         plt.figure(figsize=(8, 2))
#         sns.set(style="whitegrid", color_codes=True)
#
#     def plot(self, df):
#         ax = plt.gca()
#         sns.barplot(data=df, x='L', y='df_size', hue='impl',
#                     errorbar='sd', hatch='//', err_kws={'color': 'darkred'})
#         ax.set_title('')
#         ax.set_xlabel('L (grid size)')
#         ax.set_ylabel('Data Size (GB)')
#         ax.legend([], [], frameon=False)
#         # ax2 = ax.twinx()
#         # sns.lineplot(data=df, x='L', y='mem_util', hue='impl', marker='o', ax=ax2)
#         # ax2.set_ylabel('y2', color='r')
#
#     def save(self):
#         plt.tight_layout()
#         plt.savefig('output/larger_resolution.pdf')
#
#
# fig = Tiering()
# fig.plot(gray_scott_df)
# fig.save()
