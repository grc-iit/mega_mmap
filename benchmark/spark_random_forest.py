"""
USAGE: spark-submit --driver-memory <size> kmeans.py <path>
"""

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import struct
import pandas as pd
import sys

# Get cmd
train_path = sys.argv[1]
test_path = sys.argv[2]

# Initialize Spark
spark = SparkSession.builder.appName("LargeFileProcessing").getOrCreate()
def make_parquet_rdd(path):
    parquet_path = f"{path}*"
    rdd = spark.read.parquet(parquet_path)
    feature_cols = ["x", "y", "class"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    hermes_rdd = assembler.transform(rdd)
    return hermes_rdd

# Read training data and fit
print("Beginning Random forest")
train_rdd = make_parquet_rdd(train_path)
rf = RandomForestClassifier(k=8, seed=1)
model = rf.fit(train_rdd)

# Read testing data and predict
test_rdd = make_parquet_rdd(test_path)
preds = model.transform(test_rdd)
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(preds)
print(f'Accuracy: {accuracy}')

# Stop Spark
spark.stop()
