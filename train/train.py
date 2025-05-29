from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
import os

spark = SparkSession.builder.appName("BigDataModeling").getOrCreate()

for i in range(1, 4):
    df = spark.read.csv(f'data_batches/batch_{i}.csv', header=False, inferSchema=True)
    df = df.withColumnRenamed("_c1", "price").withColumnRenamed("_c4", "old_new").withColumnRenamed("_c5", "duration")

    indexer = StringIndexer(inputCol="old_new", outputCol="old_new_index")
    df = indexer.fit(df).transform(df)
    assembler = VectorAssembler(inputCols=["price", "old_new_index"], outputCol="features")
    data = assembler.transform(df)

    kmeans = KMeans().setK(3).setSeed(1)
    model = kmeans.fit(data)
    model.save(f"models/kmeans_model_{i}")
    print(f"Model {i} saved")
