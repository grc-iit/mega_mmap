#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn.cluster import DBSCAN

print('HERE!!!')
path = sys.argv[1]
dist = int(sys.argv[2])
min_samples = 32
print(f'KMeans on {path}')
df = pd.read_parquet(path)
dbscan = DBSCAN(eps=dist, min_samples=min_samples,
                algorithm='kd_tree')
dbscan.fit(df)
# print(f'Inertia: {dbscan.inertia_}')
# print(f'Iterations: {dbscan.n_iter_}')
# print(km.cluster_centers_)
