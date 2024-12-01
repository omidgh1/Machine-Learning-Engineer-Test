from typing import List
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessing import DataPreprocessor
import uvicorn
import os

app = FastAPI()

# Load the trained model
model = joblib.load("random_forest_model.pkl")


class PredictionRequest(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.post("/predict/")
def predict(request: List[PredictionRequest]):
    data = [req.dict() for req in request]
    X = pd.DataFrame(data)
    preprocessor = DataPreprocessor()
    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    prediction = model.predict(X_transformed).tolist()

    return {"prediction": prediction}