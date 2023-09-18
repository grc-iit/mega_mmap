from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import struct

import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("LargeFileProcessing").getOrCreate()
def make_parquest_rdd():
    parquet_path = "/home/lukemartinlogan/hermes_data/*"
    df = pd.read_parquet('/home/lukemartinlogan/hermes_data/kmeans.bin_0_1')
    print(df.columns)
    exit(1)
    rdd = spark.read.parquet(parquet_path)
    feature_cols = ["x", "y"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    hermes_rdd = assembler.transform(rdd)
    return hermes_rdd

def make_iris_rdd():
    rdd_path = "/home/lukemartinlogan/Documents/Projects/PhD/mega_mmap/benchmark/iris.csv"
    rdd = spark.read.csv(rdd_path, header=True, inferSchema=True)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    iris_rdd = assembler.transform(rdd)
    return iris_rdd

# Read binary files as an RDD of (String, bytes)
rdd = make_parquest_rdd()
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(rdd)
print(model.clusterCenters())

# Stop Spark
spark.stop()
