from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import struct
import sys

# Get cmd
path = sys.argv[1]

# Initialize Spark
spark = SparkSession.builder.appName("LargeFileProcessing").getOrCreate()
# conf = SparkConf().setAppName("BinaryToRDDExample").setMaster("spark://localhost:7077")
# sc = SparkContext(conf=conf)

# Define the path to your binary file directory
def make_parquet_rdd():
    parquet_path = f"{path}*"
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

rdd = make_parquet_rdd()
sorted_df = rdd.orderBy('x')

sorted_df.show(10)

# Stop Spark
spark.stop()