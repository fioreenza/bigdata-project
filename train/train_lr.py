import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, StringIndexerModel, VectorAssembler
from pyspark.ml.regression import LinearRegressionModel, LinearRegression
from pyspark.sql.functions import col
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionEngine:
    def __init__(self, spark, dataset_path=None, model_dir="models/train_batch1", load_existing=False):
        self.spark = spark
        self.model_dir = model_dir
        self.categorical_cols = [
            "Property Type", "Old/New", "Duration", "Town/City", 
            "District", "County", "PPDCategory Type", "Record Status - monthly file only"
        ]
        self.indexers = {}
        self.model = None

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if load_existing:
            self.__load_model_and_indexers()
        else:
            if dataset_path is None:
                raise ValueError("dataset_path harus diberikan jika load_existing=False")
            self.__load_and_train(dataset_path)

    def __load_and_train(self, dataset_path):
        logger.info("Loading data...")
        df = self.spark.read.csv(dataset_path, header=True, inferSchema=True)
        df = df.withColumn("Price_double", col("Price").cast("double")).na.drop(subset=["Price_double"])

        for col_name in self.categorical_cols:
            logger.info(f"Indexing {col_name} ...")
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_index", handleInvalid="keep")
            model_indexer = indexer.fit(df)
            df = model_indexer.transform(df)

            indexer_path = os.path.join(self.model_dir, f"{col_name}_indexer")
            model_indexer.write().overwrite().save(indexer_path)
            self.indexers[col_name] = StringIndexerModel.load(indexer_path)

        feature_cols = [col + "_index" for col in self.categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)

        logger.info("Training Linear Regression model...")
        lr = LinearRegression(featuresCol="features", labelCol="Price_double")
        self.model = lr.fit(df)

        model_path = os.path.join(self.model_dir, "linear_regression_model")
        self.model.write().overwrite().save(model_path)
        logger.info("Model training completed and saved.")

    def __load_model_and_indexers(self):
        logger.info("Loading model and indexers from disk...")
        for col_name in self.categorical_cols:
            indexer_path = os.path.join(self.model_dir, f"{col_name}_indexer")
            if not os.path.exists(indexer_path):
                raise FileNotFoundError(f"Indexer model for {col_name} not found in {indexer_path}")
            self.indexers[col_name] = StringIndexerModel.load(indexer_path)

        model_path = os.path.join(self.model_dir, "linear_regression_model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Linear regression model not found in {model_path}")
        self.model = LinearRegressionModel.load(model_path)
        logger.info("Model and indexers loaded successfully.")

    def predict(self, input_dict):
        if self.model is None:
            raise ValueError("Model not trained or loaded yet.")

        input_df = self.spark.createDataFrame([input_dict])

        for col_name in self.categorical_cols:
            if col_name in input_dict:
                input_df = self.indexers[col_name].transform(input_df)
            else:
                input_df = input_df.withColumn(col_name + "_index", col(col_name).cast("string"))
                input_df = input_df.na.fill({col_name + "_index": "-1"})

        feature_cols = [col + "_index" for col in self.categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        input_df = assembler.transform(input_df)

        prediction = self.model.transform(input_df).select("prediction").collect()[0][0]
        return prediction


if __name__ == "__main__":
    spark = SparkSession.builder.appName("PricePredictionTraining").getOrCreate()
    engine = PricePredictionEngine(spark, dataset_path="data_batches/batch_1.csv", model_dir="models/train_batch1", load_existing=False)
    spark.stop()
    logger.info("Training completed and model saved.")
