from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model from MLflow registry or local
model = mlflow.pyfunc.load_model("models:/credit_risk_model/Production")

@app.get("/")
def read_root():
    return {"message": "Credit Risk Scoring API is up."}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerFeatures):
    df = pd.DataFrame([data.dict()])
    prob = model.predict(df)[0]
    return PredictionResponse(risk_probability=prob)
