#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn.cluster import KMeans

path = sys.argv[1]
print(f'KMeans on {path}')
df = pd.read_parquet(path)
km = KMeans(n_clusters=8, max_iter=300)
km.fit(df)
print(f'Inertia: {km.inertia_}')
print(f'Iterations: {km.n_iter_}')
print(km.cluster_centers_)
