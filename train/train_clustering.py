import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import StringIndexerModel
from pyspark.ml.clustering import KMeans, KMeansModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyClusteringEngine:
    """
    Engine untuk melakukan clustering properti menggunakan KMeans.
    """
    def __init__(self, spark, dataset_path=None, model_dir="models/train_batch_clustering", load_existing=False, k=3):
        self.spark = spark
        self.model_dir = model_dir
        self.k = k
        self.indexers = {}
        self.model = None
        self.features_col = "features"
        self.categorical_cols = [
            "Property Type", "Old/New", "Duration", "Town/City", 
            "District", "County", "PPDCategory Type", "Record Status - monthly file only"
        ]

        os.makedirs(self.model_dir, exist_ok=True)

        if load_existing:
            self._load_model_and_indexers()
        else:
            if not dataset_path:
                raise ValueError("dataset_path harus diberikan saat load_existing=False")
            self._load_and_train(dataset_path)

    def _load_and_train(self, dataset_path):
        logger.info(f"Loading data for clustering: {dataset_path}")
        df = self.spark.read.csv(dataset_path, header=True, inferSchema=True)

        # Index categorical columns
        for col_name in self.categorical_cols:
            if col_name in df.columns:
                logger.info(f"Indexing column: {col_name}")
                indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_idx", handleInvalid="keep")
                model_indexer = indexer.fit(df)
                df = model_indexer.transform(df)
                idx_path = os.path.join(self.model_dir, f"{col_name}_indexer")
                model_indexer.write().overwrite().save(idx_path)
                self.indexers[col_name] = model_indexer
            else:
                logger.warning(f"Column {col_name} tidak ada di dataset, diabaikan.")

        # Assemble features
        feature_cols = [c + "_idx" for c in self.indexers.keys()]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol=self.features_col)
        df_features = assembler.transform(df)

        # Train KMeans
        logger.info(f"Training KMeans clustering with k={self.k}")
        kmeans = KMeans(k=self.k, featuresCol=self.features_col, predictionCol="cluster")
        self.model = kmeans.fit(df_features)

        # Save model
        model_path = os.path.join(self.model_dir, "kmeans_model")
        self.model.write().overwrite().save(model_path)
        logger.info(f"KMeans model saved to {model_path}")

    def _load_model_and_indexers(self):
        logger.info(f"Loading clustering model and indexers from {self.model_dir}")
        # Load indexers
        for col_name in self.categorical_cols:
            idx_path = os.path.join(self.model_dir, f"{col_name}_indexer")
            if os.path.exists(idx_path):
                self.indexers[col_name] = StringIndexerModel.load(idx_path)
        # Load KMeans model
        model_path = os.path.join(self.model_dir, "kmeans_model")
        if os.path.exists(model_path):
            self.model = KMeansModel.load(model_path)
        else:
            raise FileNotFoundError(f"KMeans model tidak ditemukan di {model_path}")
        logger.info("Clustering model and indexers loaded successfully.")

    def predict_cluster(self, input_dict):
        if not self.model:
            raise ValueError("Model belum diload atau dilatih.")

        input_df = self.spark.createDataFrame([input_dict])
        # Apply indexers
        for col_name, indexer in self.indexers.items():
            input_df = indexer.transform(input_df)
        # Assemble features
        assembler = VectorAssembler(inputCols=[c + "_idx" for c in self.indexers.keys()], outputCol=self.features_col)
        df_feat = assembler.transform(input_df)
        # Predict cluster
        prediction = self.model.transform(df_feat).select("cluster").collect()[0][0]
        return int(prediction)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("PropertyClusteringTraining").getOrCreate()
    engine = PropertyClusteringEngine(
        spark,
        dataset_path="data_batches/batch_2.csv",
        model_dir="models/train_batch2",
        load_existing=False,
        k=3
    )
    spark.stop()
    logger.info("Clustering model training complete.")