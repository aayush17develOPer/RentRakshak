# src/api/main.py
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List

from src.core.logger import get_logger
from src.scraper.nobroker import NoBrokerScraper
from src.vision.analyzer import VisionAnomalyAnalyzer
from src.vision.hasher import PHasher
from src.vision.image_downloader import ImageDownloader
from src.vision.hash_store import ImageHashStore
from src.vision.match_policy import MatchContextPolicy
from src.vision.aggregator import VisionAggregator
from src.tabular.analyzer import PriceAnomalyAnalyzer
from src.nlp.analyzer import TextScamAnalyzer
from src.api.ensemble import EnsembleScorer

logger = get_logger(__name__)

app = FastAPI(title="RentRakshak API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---
class ListingRequest(BaseModel):
    url: HttpUrl


class Proof(BaseModel):
    category: str
    description: str


class ListingResponse(BaseModel):
    scam_score: float
    risk_level: str
    proofs: List[Proof]
    action_required: str


# --- Service wiring (composition root) ---
DB_PATH = "data/rentrakshak.duckdb"

scraper = NoBrokerScraper()

vision_analyzer = VisionAnomalyAnalyzer(
    db_path=DB_PATH,
    similarity_threshold=8,
    downloader=ImageDownloader(),
    hasher=PHasher(),
    store=ImageHashStore(db_path=DB_PATH),
    policy=MatchContextPolicy(),
    aggregator=VisionAggregator(),
)

ensemble_scorer = EnsembleScorer(
    vision=vision_analyzer,
    price=PriceAnomalyAnalyzer(),
    text=TextScamAnalyzer(),
)


# --- Endpoint ---
@app.post("/check-listing", response_model=ListingResponse)
async def check_listing(request: ListingRequest):
    url_str = str(request.url)
    logger.info("--- Incoming request | url=%s ---", url_str)

    try:
        listing_data = await scraper.extract_listing(url_str)
        result = await ensemble_scorer.calculate_risk(listing_data)
        logger.info("--- Request processed successfully ---")
        return ListingResponse(**result)

    except Exception as e:
        logger.error(
            "Critical failure | url=%s | error=%s",
            url_str, str(e), exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze listing. Please verify the URL.",
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
