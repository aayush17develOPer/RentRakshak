# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

app = FastAPI(
    title="RentRakshak API",
    description="Pre-Visit Rental Fraud Detection API",
    version="1.0.0"
)

# --- Schemas ---
class ListingRequest(BaseModel):
    url: HttpUrl

class Proof(BaseModel):
    category: str  # e.g., "Vision", "Price", "Language"
    description: str

class ListingResponse(BaseModel):
    scam_score: float
    risk_level: str
    proofs: List[Proof]
    action_required: str

# --- Routes ---
@app.get("/health")
async def health_check():
    return {"status": "active", "module": "Pre-Visit Guard"}

@app.post("/check-listing", response_model=ListingResponse)
async def check_listing(request: ListingRequest):
    # TODO in Step 3: Inject the Scraper to get listing data
    # TODO in Step 4: Pass data to EnsembleScorer containing our IRiskAnalyzers
    
    # Placeholder response to verify the API works
    return ListingResponse(
        scam_score=0.0,
        risk_level="Pending Implementation",
        proofs=[],
        action_required="Awaiting ML services"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)