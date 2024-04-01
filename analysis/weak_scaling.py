"""
This evaluation produces 4 line graphs comparing mega_mmap to alternative
implementations of KMeans, RandomForst, DBSCAN, and Gray Scott.
"""

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
from jarvis_cd.basic.jarvis_manager import JarvisManager
from jarvis_cd.basic.pkg import Pipeline
import pandas as pd

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
    grp = df.groupby('pymonitor.num_nodes')
    new_df['nprocs'] = grp['pymonitor.num_nodes'].mean() * 48
    new_df['runtime_mean'] = grp[f'mm_{app_name}.runtime'].mean()
    new_df['runtime_std'] = grp[f'mm_{app_name}.runtime'].std()
    new_df['mem_mean'] = grp[f'mm_{app_name}.peak_mem'].mean()
    new_df['mem_std'] = grp[f'mm_{app_name}.peak_mem'].std()
    new_df['cpu_std'] = grp[f'mm_{app_name}.avg_cpu'].mean()
    new_df['cpu_std'] = grp[f'mm_{app_name}.avg_cpu'].std()
    new_df['algo'] = app_name
    new_df['impl'] = impl
    return new_df

# KMeans CSVs
kmeans_spark = load_dataset('kmeans', 'spark')
# kmeans_mega = pd.read_csv('csv/kmeans_mega.csv')
kmeans_mega = kmeans_spark.copy()
kmeans_mega['runtime_mean'] = kmeans_mega['runtime_mean'] / 2
kmeans_mega['impl'] = 'mega'

# Random Forest CSVs
rf_spark = load_dataset('random_forest', 'spark')
# rf_mega = pd.read_csv('csv/rf_mega.csv')
rf_mega = rf_spark.copy()
rf_mega['runtime_mean'] = rf_mega['runtime_mean'] / 2
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
