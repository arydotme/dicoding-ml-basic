import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionFeature(BaseModel):
    transaction_amount: float
    customer_age: float
    transaction_duration: float
    login_attempts: float
    account_balance: float
    location: str
    customer_occupation: str
    transaction_channel: str
    transaction_type: str

model = joblib.load("model/DecisionTreeClassifier_best.pkl")

@app.post("/predict")
def predict(data: TransactionFeature):
    X = np.array([[data.transaction_amount, data.customer_age, data.transaction_duration,
                   data.login_attempts, data.account_balance, data.location,
                   data.customer_occupation, data.transaction_channel, data.transaction_type]])
    try:
        prediction = model.predict(X)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")