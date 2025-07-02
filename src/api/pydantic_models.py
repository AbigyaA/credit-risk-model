from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_count: int
    Value_sum: float
    Value_mean: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    # Add more features as used in training

class PredictionResponse(BaseModel):
    risk_probability: float
