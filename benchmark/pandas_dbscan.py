#!/usr/bin/env python3

import pandas as pd
import sys
# from sklearn.cluster import DBSCAN
from pyclustering.cluster.dbscan import dbscan

path = sys.argv[1]
dist = int(sys.argv[2])
min_samples = 32
print(f'DBSCAN on {path}')
df = pd.read_parquet(path)
points = df.values.tolist()
dbscan_instance = dbscan(points, dist, min_samples)
dbscan_instance.process()
dbscan_clusters = dbscan_instance.get_clusters()
# print(dbscan_clusters)

# dbscan = DBSCAN(eps=dist, min_samples=min_samples,
#                 algorithm='kd_tree')
# dbscan.fit(df)
# print(f'Inertia: {dbscan.inertia_}')
# print(f'Iterations: {dbscan.n_iter_}')
# print(km.cluster_centers_)
