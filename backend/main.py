from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models_loader import download_model
from investment_optimizer import optimize_investment
import asyncio
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    download_model()
    print("Server started and model loaded.")

@app.get("/predict")
async def get_forecast():
    # Dummy response for demo; replace with Transformer predictions
    return {"forecast": [100, 102, 105, 108, 110]}

@app.post("/invest")
async def post_investment():
    # Dummy data
    roi = [0.1, 0.15, 0.12]
    risk = [0.05, 0.1, 0.07]
    budget = 1000
    allocation = optimize_investment(np.array(roi), np.array(risk), budget)
    return {"allocation": allocation.tolist()}
