from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

app = FastAPI()
spark = SparkSession.builder.appName("APIModel").getOrCreate()

class InputData(BaseModel):
    price: float
    old_new_index: float

@app.post("/predict_cluster/{model_id}")
def predict_cluster(model_id: int, data: InputData):
    model = KMeansModel.load(f"models/kmeans_model_{model_id}")
    df = spark.createDataFrame([[data.price, data.old_new_index]], ["price", "old_new_index"])
    assembler = VectorAssembler(inputCols=["price", "old_new_index"], outputCol="features")
    df = assembler.transform(df)
    prediction = model.transform(df).collect()[0]['prediction']
    return {"cluster": int(prediction)}

@app.get("/health")
def health():
    return {"status": "ok"}
