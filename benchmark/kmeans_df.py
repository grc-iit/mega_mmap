"""
USAGE: python3 kmeans_df.py <out_dir>
"""
import pandas as pd
import sys
import mpi4py.MPI as MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create DF centers
df = pd.DataFrame(columns=['x', 'y'])
centers = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]]

path = sys.argv[1]
pd.to_parquet(df, 'kmeans{i}.parquet')
