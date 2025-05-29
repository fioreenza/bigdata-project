import os
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, StringIndexerModel, VectorAssembler
from pyspark.ml.regression import LinearRegressionModel, LinearRegression
from pyspark.sql.functions import col
from train_lr import PricePredictionEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

spark = SparkSession.builder \
    .appName("PricePredictionAPI") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()
model_dir = "models/train_batch1"
engine = PricePredictionEngine(spark, model_dir=model_dir, load_existing=True)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Cek apakah semua kolom wajib ada di input
        missing_cols = [col for col in engine.categorical_cols if col not in data]
        if missing_cols:
            return jsonify({"error": f"Missing columns in input: {missing_cols}"}), 400

        prediction = engine.predict(data)
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400
    
@app.route("/health", methods=["GET"])
def health_check():
    try:
        spark.catalog.listTables()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
