from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Iris Prediction API")

# Load model saat startup
model = joblib.load("iris_model.pkl")

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "Iris Model API is running"}

@app.post("/predict")
def predict(data: IrisRequest):
    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(input_data)[0]

    class_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction_class": int(prediction),
        "prediction_name": class_names[prediction]
    }