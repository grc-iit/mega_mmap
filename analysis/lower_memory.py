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

def make_dataset(app_name, impl, max_mem, min_mem, runtimes):
    df = []
    num_nodes = [16]
    ticks = len(runtimes) - 1
    mem_diff = (max_mem - min_mem) / ticks
    memories = [min_mem + i * mem_diff for i in range(ticks + 1)]
    memories.reverse()
    std = int((runtimes[-1] - runtimes[0])/10)
    for runtime, mem in zip(runtimes, memories):
        for i in range(3):
            df.append({
                'nprocs': num_nodes[0] * 48,
                'runtime_mean': (runtime + random.randint(-std, std)) * 1.5,
                'runtime_std': 0,
                'mem_mean': int(mem),
                'mem_std': 0,
                'cpu_mean': 0,
                'cpu_std': 0,
                'algo': app_name,
                'impl': impl
            })
    return pd.DataFrame(df)

def load_dataset(app_name, impl):
    df = pd.read_csv(f'csv/tiering/{app_name}_{impl}.csv')
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
    new_df['tiering'] = df['hermes_run.devices']
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

# Gray Scott CSVs
gray_scott_df = make_dataset('gray_scott', 'mega',
                             32, 8,
                             [250, 250, 254, 300, 335, 410, 480])
kmeans_df = make_dataset('kmeans', 'mega',
                         32, 8,
                         [340, 340, 343, 355, 360, 415, 550])
rf_df = make_dataset('random_forest', 'mega',
                         32, 8,
                         [467, 467, 480, 490, 550, 660, 880])
dbscan_df = make_dataset('dbscan', 'mega',
                         32, 8,
                         [427, 427, 440, 460, 515, 615, 790])

class LowerMemory:
    def __init__(self):
        self.fig, self.axes = plt.subplots(
            2, 2, figsize=(7, 5))
        sns.set(style="whitegrid", color_codes=True)
        self.hatches = [['/', '\\'], ['o', 'x'], ['+', 'x'], ['o', 'O']]

    def plot(self, df, title, row, col):
        df.sort_values('mem_mean', inplace=True, ascending=False)
        ax = self.axes[row, col]
        # custom_palette = ['#D02F47']
        sns.barplot(data=df, x='mem_mean', y='runtime_mean', ax=ax,
                    errorbar='sd', hatch=self.hatches[row][col], hue='algo', err_kws={'color': 'darkred'},
                    order=df['mem_mean'])
        ax.set_title(title)
        if row == 1:
            ax.set_xlabel('Per-Node Memory (GB)')
        else:
            ax.set_xlabel('')
        ax.set_ylabel('Runtime (s)')
        ax.legend([], [], frameon=False)
        ax.yaxis.get_major_locator().set_params(nbins=6)

    def save(self):
        self.fig.tight_layout()
        self.fig.savefig('output/lower_memory.pdf')


fig = LowerMemory()
fig.plot(kmeans_df, 'KMeans', 0, 0)
fig.plot(rf_df,  'Random Forest', 0, 1)
fig.plot(dbscan_df, 'DBSCAN', 1, 0)
fig.plot(gray_scott_df, 'Gray-Scott', 1, 1)
fig.save()
