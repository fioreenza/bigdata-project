from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
import os

spark = SparkSession.builder.appName("LinearRegressionTraining").getOrCreate()
os.makedirs("models", exist_ok=True)

# Load all batches
batches = []
for i in range(1, 4):
    df = spark.read.csv(f'data_batches/batch_{i}.csv', header=True, inferSchema=True)
    batches.append(df)
data = batches[0]
for df in batches[1:]:
    data = data.union(df)

# Cast Price to double and drop nulls
data = data.withColumn("Price_double", col("Price").cast("double")).na.drop(subset=["Price_double"])

# Index categorical columns
categorical_cols = ["Property Type", "Old/New", "Duration", "Town/City", "District", "County", "PPDCategory Type", "Record Status - monthly file only"]
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_index")
    data = indexer.fit(data).transform(data)

# Assemble features
feature_cols = [col+"_index" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Train Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="Price_double")
model = lr.fit(data)

# Save model
model.save("models/linear_regression_model")
print("Linear Regression model saved.")
