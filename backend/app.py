import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
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

model = joblib.load("model/RandomForestClassifier.pkl")

@app.post("/predict")
def predict(data: TransactionFeature):
    try:
        df = pd.DataFrame([{
            "TransactionAmount": data.transaction_amount,
            "CustomerAge": data.customer_age,
            "TransactionDuration": data.transaction_duration,
            "LoginAttempts": data.login_attempts,
            "AccountBalance": data.account_balance,
            "Location": data.location,
            "CustomerOccupation": data.customer_occupation,
            "Channel": data.transaction_channel,
            "TransactionType": data.transaction_type
        }])

        prediction = model.predict(df)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        print("ERROR BACKEND:", e)   # WAJIB lihat ini di console
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")