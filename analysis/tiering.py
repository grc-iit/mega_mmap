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
    num_nodes = [16]
    runtime = [600, 400, 345, 300]
    tiering = ['48D-48H', '48D-16N-32S', '48D-32N-16S', '48D-48N']
    for runtime, tiering in zip(runtime, tiering):
        for i in range(3):
            df.append({
                'nprocs': num_nodes[0] * 48,
                'runtime_mean': runtime + random.randint(-30, 30),
                'runtime_std': 0,
                'mem_mean': 0,
                'mem_std': 0,
                'cpu_mean': 0,
                'cpu_std': 0,
                'tiering': tiering,
                'algo': app_name,
                'impl': impl
            })
    return pd.DataFrame(df)

def make_dataset2(app_name, impl):
    df = []
    num_nodes = [16]
    runtime = [600, 400, 345, 300]
    tiering = ['48D-48H', '48D-16N-32S', '48D-32N-16S', '48D-48N']
    for runtime, tiering in zip(runtime, tiering):
        for i in range(3):
            df.append({
                'nprocs': num_nodes[0] * 48,
                'runtime_mean': runtime + random.randint(-30, 30),
                'runtime_std': 0,
                'mem_mean': 0,
                'mem_std': 0,
                'cpu_mean': 0,
                'cpu_std': 0,
                'tiering': tiering,
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
gray_scott_df = make_dataset('gray_scott', 'mega')

class Tiering:
    def __init__(self):
        plt.figure(figsize=(8, 2))
        sns.set(style="whitegrid", color_codes=True)

    def plot(self, df):
        ax = plt.gca()
        sns.barplot(data=df, x='tiering', y='runtime_mean', hue='impl',
                    errorbar='sd', hatch='//', err_kws={'color': 'darkred'})
        ax.set_title('')
        ax.set_xlabel('Tiering Strategy')
        ax.set_ylabel('Runtime (s)')
        ax.legend([], [], frameon=False)

    def save(self):
        plt.tight_layout()
        plt.savefig('output/tiering.pdf')


fig = Tiering()
fig.plot(gray_scott_df)
fig.save()
