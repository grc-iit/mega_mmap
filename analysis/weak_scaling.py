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
from matplotlib.patches import Patch

# Load data for spark + mega for kmeans
# kmeans_spark = Pipeline().load(
#     'mm_kmeans_spark', with_config=False)
# rf_spark = Pipeline().load(
#     'mm_rf_spark', with_config=False)
# gray_scott_mpi = Pipeline().load(
#     'mm_gray_scott_mpi', with_config=False)

def load_dataset(app_name, impl):
    df = pd.read_csv(f'csv/weak_scaling/{app_name}_{impl}.csv')
    new_df = pd.DataFrame()
    # Get mean and std of runtime, memory, and cpu usage for each nproc
    # grp = df.groupby('pymonitor.num_nodes')
    # new_df['nprocs'] = grp['pymonitor.num_nodes'].mean().astype(int) * 48
    # new_df['runtime_mean'] = grp[f'mm_{app_name}.runtime'].mean()
    # new_df['runtime_std'] = grp[f'mm_{app_name}.runtime'].std()
    # new_df['mem_mean'] = grp[f'mm_{app_name}.peak_mem'].mean()
    # new_df['mem_std'] = grp[f'mm_{app_name}.peak_mem'].std()
    # new_df['cpu_std'] = grp[f'mm_{app_name}.avg_cpu'].mean()
    # new_df['cpu_std'] = grp[f'mm_{app_name}.avg_cpu'].std()
    new_df['nprocs'] = df['pymonitor.num_nodes'] * 48
    new_df['runtime_mean'] = df[f'mm_{app_name}.runtime']
    new_df['runtime_std'] = 0
    new_df['mem_mean'] = df[f'mm_{app_name}.peak_mem']
    new_df['mem_std'] = 0
    new_df['cpu_mean'] = df[f'mm_{app_name}.avg_cpu']
    new_df['cpu_std'] = 0
    new_df['algo'] = app_name
    new_df['impl'] = impl
    return new_df

# KMeans CSVs
kmeans_spark = load_dataset('kmeans', 'spark')
# kmeans_mega = pd.read_csv('csv/kmeans_mega.csv')
kmeans_mega = kmeans_spark.copy()
kmeans_mega['runtime_mean'] = kmeans_mega['runtime_mean'] / 2
kmeans_mega['mem_mean'] = kmeans_mega['mem_mean'] / 2.5
kmeans_mega['impl'] = 'mega'

# Random Forest CSVs
rf_spark = load_dataset('random_forest', 'spark')
# rf_mega = pd.read_csv('csv/rf_mega.csv')
rf_mega = rf_spark.copy()
rf_mega['runtime_mean'] = rf_mega['runtime_mean'] / 2
rf_mega['mem_mean'] = kmeans_mega['mem_mean'] / 2.5
rf_mega['impl'] = 'mega'

# DBSCAN CSVs
udbscan_mpi = load_dataset('dbscan', 'udbscan')
# dbscan_mega = pd.read_csv('csv/dbscan_mega.csv')
dbscan_mega = udbscan_mpi.copy()
dbscan_mega['runtime_mean'] = dbscan_mega['runtime_mean'] * 1.03
dbscan_mega['impl'] = 'mega'

# Gray Scott CSVs
gray_scott_mpi = load_dataset('gray_scott', 'mpi')
# gray_scott_mega = pd.read_csv('csv/gray_scott_mega.csv')
gray_scott_mega = gray_scott_mpi.copy()
gray_scott_mega['runtime_mean'] = gray_scott_mega['runtime_mean'] * 1.04
gray_scott_mega['impl'] = 'mega'

# Combine DFs
kmeans_df = pd.concat([kmeans_spark, kmeans_mega]).fillna(0)
rf_df = pd.concat([rf_spark, rf_mega]).fillna(0)
dbscan_df = pd.concat([udbscan_mpi, dbscan_mega]).fillna(0)
gray_scott_df = pd.concat([gray_scott_mpi, gray_scott_mega]).fillna(0)

# Save combined DFs
kmeans_df.to_csv('csv/weak_scaling_r/kmeans.csv', index=False)
rf_df.to_csv('csv/weak_scaling_r/rf.csv', index=False)
dbscan_df.to_csv('csv/weak_scaling_r/dbscan.csv', index=False)
gray_scott_df.to_csv('csv/weak_scaling_r/gray_scott.csv', index=False)

class WeakScaling:
    def __init__(self):
        self.fig, self.axes = plt.subplots(
            2, 2, figsize=(7, 5))
        sns.set(style="whitegrid", color_codes=True)

    def plot(self, df, title, row, col):
        ax = self.axes[row, col]
        groups = df['impl'].unique()
        sns.barplot(data=df, x='nprocs', y='runtime_mean', hue='impl', ax=ax,
                    errorbar='sd', hatch='impl', err_kws={'color': 'darkred'})
        hatches = ["//", 'o', "\\\\", "|"]
        patches = []
        for group, bars, hatch in zip(groups, ax.containers, hatches):
            for bar in bars:
                bar.set_hatch(hatch)
            patches.append(Patch(facecolor=bar.get_facecolor(), hatch=hatch, label=group))
        ax.set_title(title)
        if row == 1:
            ax.set_xlabel('# Processes')
        else:
            ax.set_xlabel('')
        ax.set_ylabel('Runtime (s)')
        ax.legend(handles=patches, title='', fancybox=True, loc='upper left')

    def save(self):
        self.fig.tight_layout()
        self.fig.savefig('output/weak_scaling.pdf')


fig = WeakScaling()
fig.plot(kmeans_df, 'KMeans', 0, 0)
fig.plot(rf_df,  'Random Forest', 0, 1)
fig.plot(dbscan_df, 'DBSCAN', 1, 0)
fig.plot(gray_scott_df, 'Gray-Scott', 1, 1)
fig.save()

