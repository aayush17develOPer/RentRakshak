# src/schemas/listing.py
from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class ScrapedListing(BaseModel):
    """Data contract for scraped listings. Strictly excludes PII."""
    url: HttpUrl
    platform: str
    price: float
    bhk: int
    area_sqft: float
    floor: str
    furnishing_status: str
    age: str
    gated: int  # 0 or 1
    amenities_count: int
    amenities_list: List[str]
    sector: str
    parent_sector: str
    description: str
    image_urls: List[HttpUrl]