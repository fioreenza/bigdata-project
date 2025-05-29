from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
import os

# Inisialisasi Spark Session
spark = SparkSession.builder.appName("LinearRegressionTraining").getOrCreate()
os.makedirs("models", exist_ok=True)

# Hanya gunakan batch_1.csv
data = spark.read.csv('data_batches/batch_1.csv', header=True, inferSchema=True)

# Konversi kolom target ke tipe double dan drop baris yang null
data = data.withColumn("Price_double", col("Price").cast("double")).na.drop(subset=["Price_double"])

# Daftar kolom kategorikal
categorical_cols = ["Property Type", "Old/New", "Duration", "Town/City", "District", "County", "PPDCategory Type", "Record Status - monthly file only"]

# StringIndexer untuk setiap kolom kategorikal
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_index")
    data = indexer.fit(data).transform(data)

# VectorAssembler untuk gabungkan fitur
feature_cols = [col + "_index" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Latih model regresi linear
lr = LinearRegression(featuresCol="features", labelCol="Price_double")
model = lr.fit(data)

# Simpan model
model.save("models/linear_regression_model")
print("Linear Regression model saved.")
