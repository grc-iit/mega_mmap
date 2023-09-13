from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import struct

# Initialize Spark
spark = SparkSession.builder.appName("LargeFileProcessing").getOrCreate()
# conf = SparkConf().setAppName("BinaryToRDDExample").setMaster("spark://localhost:7077")
# sc = SparkContext(conf=conf)

# Define the path to your binary file directory
#binary_files_path = "/home/lukemartinlogan/hermes_data/kmeans.bin"
#binary_files_rdd = sc.binaryFiles(binary_files_path)
rdd_path = "/home/lukemartinlogan/Documents/Projects/PhD/mega_mmap/benchmark/iris.csv"
rdd = spark.read.csv(rdd_path, header=True, inferSchema=True)
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
print(rdd)
iris_feature_df = assembler.transform(rdd)


# Read binary files as an RDD of (String, bytes)
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(iris_feature_df)
print(model.clusterCenters())

# # Convert binary data to RDD of floats
# def bytes_to_floats(byte_data):
#     float_size = 4  # Assuming each float is 4 bytes
#     float_count = len(byte_data) // float_size
#     return struct.unpack("f" * float_count, byte_data)
#
# float_data_rdd = binary_files_rdd.flatMap(lambda x: bytes_to_floats(x[1]))
# binary_files_rdd.sortByKey(ascending=True)

# Now you have an RDD of floats (float_data_rdd)
# float_data_rdd.take(10)  # Print the first 10 floats

# Stop Spark
spark.stop()