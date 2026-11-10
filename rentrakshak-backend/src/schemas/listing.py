from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class ScrapedListing(BaseModel):
    """
    Data contract for scraped property listings.
    Strictly excludes owner/contact PII and unrelated NoBroker fields.
    """

    # Listing metadata
    url: HttpUrl
    platform: str = "nobroker"

    # Basic property information
    title: str = ""
    locality: str = ""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rent: Optional[float] = None
    maintenance_amount: Optional[float] = None
    area_sq_ft: Optional[float] = None
    deposit: Optional[float] = None

    # Property configuration
    bedroom_configuration: str = ""
    # property_type: str = ""
    parking: str = ""

    # Building information
    property_age: Optional[float] = None
    balcony: Optional[float] = None

    # Property overview
    furnishing_status: str = ""
    facing: str = ""
    # water_supply: str = ""
    floor: Optional[float] = None

    # Additional property attributes
    pet_allowed: str = ""
    gated_security: str = ""

    # Description and images
    image_urls: List[HttpUrl] = Field(default_factory=list)