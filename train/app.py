import os
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from train_lr import PricePredictionEngine 
from train_clustering import PropertyClusteringEngine
from train_recomendation import PropertyRecommenderEngine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)


spark = SparkSession.builder \
    .appName("PropertyAnalysisAPI") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

price_model_dir = "models/train_batch1"
try:
    price_engine = PricePredictionEngine(spark, model_dir=price_model_dir, load_existing=True)
    logger.info(f"PricePredictionEngine loaded successfully from {price_model_dir}.")
except Exception as e:
    logger.error(f"Failed to load PricePredictionEngine: {e}", exc_info=True)
    price_engine = None 

clustering_model_dir = "models/train_batch2"
try:
    clustering_engine = PropertyClusteringEngine(spark, model_dir=clustering_model_dir, load_existing=True)
    logger.info(f"PropertyClusteringEngine loaded successfully from {clustering_model_dir}.")
except Exception as e:
    logger.error(f"Failed to load PropertyClusteringEngine: {e}", exc_info=True)
    clustering_engine = None

recommender_model_dir = "models/train_batch3" 
try:
    recommender_engine = PropertyRecommenderEngine(spark, model_dir=recommender_model_dir, load_existing=True)
    logger.info(f"PropertyRecommenderEngine loaded successfully from {recommender_model_dir}.")
except Exception as e:
    logger.error(f"Failed to load PropertyRecommenderEngine: {e}", exc_info=True)
    recommender_engine = None

@app.route("/predict", methods=["POST"])
def predict_price():
    """
    Endpoint to predict property price.
    Expects a JSON payload with property features.
    """
    if price_engine is None:
        logger.error("PricePredictionEngine not available.")
        return jsonify({"error": "Price prediction service is currently unavailable."}), 503

    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        missing_cols = [col for col in price_engine.categorical_cols if col not in data]
        if missing_cols:
            logger.warning(f"Missing columns for price prediction: {missing_cols}")
            return jsonify({"error": f"Missing columns in input for price prediction: {missing_cols}"}), 400

        prediction = price_engine.predict(data)
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        logger.error(f"Price prediction error: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during price prediction: {str(e)}"}), 500
    
@app.route("/clustering", methods=["POST"])
def cluster_property():
    """
    Endpoint untuk memprediksi cluster properti.
    Expects JSON payload dengan fitur properti.
    """
    if clustering_engine is None:
        return jsonify({"error": "Clustering service unavailable."}), 503
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400
        # Optional: parameter k override
        cluster_id = clustering_engine.predict_cluster(data)
        return jsonify({"cluster": cluster_id})
    except Exception as e:
        logger.error(f"Clustering error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/recommend", methods=["POST"])
def recommend_properties():
    """
    Endpoint to recommend similar properties.
    Expects a JSON payload with features of the property to find similarities for.
    """
    if recommender_engine is None:
        logger.error("PropertyRecommenderEngine not available.")
        return jsonify({"error": "Property recommendation service is currently unavailable."}), 503

    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        input_property_dict = data.get("property_features")
        if not input_property_dict or not isinstance(input_property_dict, dict):
            return jsonify({"error": "Missing or invalid 'property_features' in JSON payload. It should be a dictionary."}), 400

        num_recommendations = data.get("num_recommendations", 5)
        try:
            num_recommendations = int(num_recommendations)
            if num_recommendations <= 0:
                 raise ValueError("Number of recommendations must be positive.")
        except ValueError:
            return jsonify({"error": "'num_recommendations' must be a positive integer."}), 400
        
        input_property_id = data.get("input_property_id") # Optional
        id_column_name = data.get("id_column_name", "Transaction unique identifier") # Optional, with default

        if not any(key in input_property_dict for key in recommender_engine.active_categorical_cols):
             logger.warning("The 'property_features' dictionary does not contain any known categorical features.")

        recommendations_df = recommender_engine.recommend_similar(
            input_property_dict=input_property_dict,
            num_recommendations=num_recommendations,
            input_property_id=input_property_id,
            id_column_name=id_column_name
        )
        
        recommendations_list = [row.asDict() for row in recommendations_df.collect()]
        
        return jsonify({
            "message": f"Found {len(recommendations_list)} similar properties.",
            "recommendations": recommendations_list
        })
    except ValueError as ve:
        logger.warning(f"Recommendation input error: {ve}", exc_info=True)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Property recommendation error: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during property recommendation: {str(e)}"}), 500
        
@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for the API.
    Checks Spark session availability.
    """
    try:
        spark.catalog.listTables() 
        
        price_engine_status = "loaded" if price_engine else "failed_to_load"
        recommender_engine_status = "loaded" if recommender_engine else "failed_to_load"

        return jsonify({
            "status": "ok", 
            "spark_session": "active",
            "price_prediction_engine": price_engine_status,
            "property_recommender_engine": recommender_engine_status
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)