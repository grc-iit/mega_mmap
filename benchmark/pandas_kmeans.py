#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn.cluster import KMeans

path = sys.argv[1]
k = int(sys.argv[2])
max_iter = int(sys.argv[3])
nprocs = int(sys.argv[4])
print(f'KMeans on {path} with {k} clusters and {max_iter} iterations')
df = pd.read_parquet(path)
km = KMeans(n_clusters=k, max_iter=max_iter)
km.fit(df)
print(f'Inertia: {km.inertia_}')
print(f'Iterations: {km.n_iter_}')
print(km.cluster_centers_)
