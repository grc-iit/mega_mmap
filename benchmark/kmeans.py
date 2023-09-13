from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import struct

# Initialize Spark
spark = SparkSession.builder.appName("LargeFileProcessing").getOrCreate()
rdd_path = "/home/lukemartinlogan/Documents/Projects/PhD/mega_mmap/benchmark/iris.csv"
rdd = spark.read.csv(rdd_path, header=True, inferSchema=True)
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
iris_feature_df = assembler.transform(rdd)

# Read binary files as an RDD of (String, bytes)
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(iris_feature_df)
print(model.clusterCenters())

# Stop Spark
spark.stop()
