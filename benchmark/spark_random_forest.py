"""
USAGE: spark-submit spark_random_forest.py <train_path> <test_path> <num_trees> <max_depth>
"""

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import struct
import pandas as pd
import sys

# Get cmd
print(sys.argv)
train_path = sys.argv[1]
test_path = sys.argv[2]
num_trees = int(sys.argv[3])
max_depth = int(sys.argv[4])

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
print(f'Beginning Random forest on {train_path} and {test_path} '
      f'with {num_trees} trees and max depth of {max_depth}')
train_rdd = make_parquet_rdd(train_path)
model = RandomForest.trainClassifier(
    train_rdd, numTrees=num_trees, maxDepth=max_depth, seed=1)

# Read testing data and predict
test_rdd = make_parquet_rdd(test_path)
preds = model.transform(test_rdd)
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(preds)
print(f'Accuracy: {accuracy}')

# Stop Spark
spark.stop()
