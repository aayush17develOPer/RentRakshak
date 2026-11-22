from typing import Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl


class PhotoMeta(BaseModel):
    """
    Per-image metadata extracted from NoBroker's photos[] array.
    Carries everything needed for vision analysis context filtering
    without re-downloading or re-scraping.
    """

    image_url: str
    display_pic: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    image_source: str = ""          # e.g. "nobroker"
    uploaded_by: str = ""           # e.g. "OWNER"
    state: str = ""                 # e.g. "APPROVED"
    nb_duplicate_flag: bool = False  # NoBroker's own duplicate detection
    position: int = 0               # order in the listing gallery


class ScrapedListing(BaseModel):
    """
    Data contract for scraped property listings.
    All new fields are Optional with defaults so older serialized
    data and existing callers remain compatible.
    """

    # ── Listing identity ──────────────────────────────────────
    url: HttpUrl
    platform: str = "nobroker"
    nobroker_id: str = ""
    title: str = ""

    # ── Location ──────────────────────────────────────────────
    city: str = ""
    locality: str = ""
    nb_locality: str = ""
    nb_locality_id: str = ""
    society: str = ""
    street: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accurate_location: Optional[bool] = None
    location_accuracy_label: str = ""   # from aea__.ACCURATE_LOCATION

    # ── Price ─────────────────────────────────────────────────
    rent: Optional[float] = None
    deposit: Optional[float] = None
    maintenance_amount: Optional[float] = None

    # ── Unit attributes ───────────────────────────────────────
    bedroom_configuration: str = ""     # e.g. "1 RK", "2 BHK"
    bath_val: Optional[float] = None
    area_sq_ft: Optional[float] = None
    floor: Optional[float] = None
    total_floor: Optional[int] = None
    balcony: Optional[float] = None
    facing: str = ""
    furnishing_status: str = ""
    property_type: str = ""             # e.g. "Apartment"
    building_type: str = ""             # raw code e.g. "AP"
    property_age: Optional[float] = None
    parking: str = ""
    water_supply: str = ""
    lease_type: List[str] = Field(default_factory=list)

    # ── Amenities (flat booleans) ─────────────────────────────
    has_gym: bool = False
    has_pool: bool = False
    has_lift: bool = False
    has_security: bool = False

    # ── Additional attributes ─────────────────────────────────
    pet_allowed: str = ""
    gated_security: str = ""

    # ── Owner / poster ────────────────────────────────────────
    owner_id: str = ""

    # ── Description ───────────────────────────────────────────
    description: str = ""

    # ── Timestamps (epoch ms from NoBroker JSON) ──────────────
    creation_date: Optional[int] = None
    last_update_date: Optional[int] = None
    activation_date: Optional[int] = None
    last_activation_date: Optional[int] = None
    reactivation_source: str = ""

    # ── Images ────────────────────────────────────────────────
    image_urls: List[HttpUrl] = Field(default_factory=list)
    photos_meta: List[PhotoMeta] = Field(default_factory=list)

    # ── Raw snapshot (for re-parsing without re-scraping) ─────
    raw_json: Optional[Dict] = None
