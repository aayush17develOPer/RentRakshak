# src/api/main.py
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List

from src.scraper.nobroker import NoBrokerScraper
from src.vision.analyzer import VisionAnomalyAnalyzer
from src.tabular.analyzer import PriceAnomalyAnalyzer
from src.nlp.analyzer import TextScamAnalyzer
from src.api.ensemble import EnsembleScorer
from src.core.logger import get_logger
from fastapi.middleware.cors import CORSMiddleware 

logger = get_logger(__name__)

app = FastAPI(title="RentRakshak API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class ListingRequest(BaseModel): url: HttpUrl
class Proof(BaseModel): category: str; description: str
class ListingResponse(BaseModel): scam_score: float; risk_level: str; proofs: List[Proof]; action_required: str

# --- Services ---
scraper = NoBrokerScraper()
ensemble_scorer = EnsembleScorer(VisionAnomalyAnalyzer(), PriceAnomalyAnalyzer(), TextScamAnalyzer())

@app.post("/check-listing", response_model=ListingResponse)
async def check_listing(request: ListingRequest):
    url_str = str(request.url)
    logger.info(f"--- Incoming Request for URL: {url_str} ---")
    
    try:
        listing_data = await scraper.extract_listing(url_str)
        result = await ensemble_scorer.calculate_risk(listing_data)
        logger.info("--- Request Processed Successfully ---")
        return ListingResponse(**result)
        
    except Exception as e:
        logger.error(f"Critical failure during API processing for {url_str} | Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to analyze listing. Please verify the URL.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn Server...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)