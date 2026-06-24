from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.prediction import router as prediction_router

app = FastAPI(title="CoinVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    prediction_router
)