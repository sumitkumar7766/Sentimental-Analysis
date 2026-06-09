import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .routers import predict, metrics, datasets

# 1. Initialize Database Tables
# Automatically creates the database schema (SQLite or PostgreSQL) if not already created
Base.metadata.create_all(bind=engine)

# 2. Create FastAPI instance
app = FastAPI(
    title="Hybrid Multimodal Sentiment Analysis API",
    description="Backend services providing deep-learning inference and explainability metrics for multimodal sentiment analysis.",
    version="1.0.0"
)

# 3. Configure CORS Policies
# Allows frontend dashboard (running on port 3000 by default) to make fetch requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Register Modality Routers
app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(datasets.router)

# 5. Core Health-check Route
@app.get("/")
def read_root():
    return {
        "status": "online",
        "platform": "Hybrid Multimodal Sentiment Analysis Platform",
        "apis": [
            "/predict",
            "/predict/emotion",
            "/metrics",
            "/datasets"
        ]
    }
