import asyncio
import json
import time

from playwright.async_api import async_playwright
from typing import Dict, Any, List

from src.core.interfaces import IListingScraper
from src.core.logger import get_logger
from src.schemas.listing import PhotoMeta, ScrapedListing

logger = get_logger(__name__)


class NoBrokerScraper(IListingScraper):
    """
    Playwright-based dynamic scraper for NoBroker.

    Features:
    - Dynamic DOM extraction via window.nb.appState.
    - Full field extraction for vision + price analysis.
    - Per-image metadata (PhotoMeta) alongside image URLs.
    - Raw JSON snapshot for future re-parsing.
    - Comprehensive execution logging.
    """

    # ── Image URL construction ────────────────────────────────

    def _extract_images(self, photos_data: List[Dict]) -> List[str]:
        """
        Constructs image URLs from the photos[] array.
        Uses 'large' resolution, falls back to 'original' then 'medium'.
        """
        if not photos_data or not isinstance(photos_data, list):
            logger.debug("No photos data to extract image URLs from.")
            return []

        urls: List[str] = []
        for photo in photos_data:
            images_map = photo.get("imagesMap") or {}
            img_name = (
                images_map.get("large")
                or images_map.get("original")
                or images_map.get("medium")
            )
            if not img_name or not isinstance(img_name, str):
                continue

            prop_id = img_name.split("_")[0]
            if len(prop_id) == 32:
                urls.append(
                    f"https://images.nobroker.in/images/{prop_id}/{img_name}"
                )

        unique_urls = list(dict.fromkeys(urls))
        logger.info("Constructed %d image URLs from JSON data.", len(unique_urls))
        return unique_urls[:15]

    # ── Per-image metadata extraction ─────────────────────────

    def _extract_photos_meta(self, photos_data: List[Dict]) -> List[PhotoMeta]:
        """
        Builds a list of PhotoMeta from NoBroker's photos[] array.
        Captures dimensions, source, uploader, approval state, and
        NoBroker's own duplicate flag.
        """
        if not photos_data or not isinstance(photos_data, list):
            logger.debug("No photos data to extract metadata from.")
            return []

        meta_list: List[PhotoMeta] = []
        for idx, photo in enumerate(photos_data):
            images_map = photo.get("imagesMap") or {}
            img_name = (
                images_map.get("large")
                or images_map.get("original")
                or images_map.get("medium")
            )
            if not img_name or not isinstance(img_name, str):
                continue

            prop_id = img_name.split("_")[0]
            if len(prop_id) != 32:
                continue

            image_url = (
                f"https://images.nobroker.in/images/{prop_id}/{img_name}"
            )

            meta_list.append(
                PhotoMeta(
                    image_url=image_url,
                    display_pic=bool(photo.get("displayPic", False)),
                    width=photo.get("w"),
                    height=photo.get("h"),
                    image_source=photo.get("imageSource") or "",
                    uploaded_by=photo.get("uploadedBy") or "",
                    state=photo.get("state") or "",
                    nb_duplicate_flag=bool(photo.get("duplicate", False)),
                    position=idx,
                )
            )

        logger.info("Extracted PhotoMeta for %d images.", len(meta_list))
        return meta_list[:15]

    # ── Society name normalization ────────────────────────────

    @staticmethod
    def _normalize_society(raw: str) -> str:
        """
        Lowercase, strip whitespace, remove common suffixes.
        Helps with fuzzy matching across listings.
        """
        if not raw:
            return ""
        name = raw.strip().lower()
        for suffix in (
            " apartment", " apartments", " flats", " flat",
            " residency", " society", " tower", " towers",
        ):
            if name.endswith(suffix):
                name = name[: -len(suffix)].rstrip()
        return name

    # ── Playwright scraping ───────────────────────────────────

    async def _do_scrape(self, url: str) -> Dict[str, Any]:
        logger.info("Starting Playwright scrape | url=%s", url)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            try:
                await asyncio.sleep(3)
                await page.goto(
                    url, timeout=60000, wait_until="domcontentloaded"
                )

                # Poll for window.nb
                logger.debug("Polling for window.nb object...")
                is_ready = False
                for _ in range(15):
                    is_ready = await page.evaluate(
                        "() => typeof window !== 'undefined'"
                        " && typeof window.nb !== 'undefined'"
                    )
                    if is_ready:
                        break
                    await asyncio.sleep(1)

                if not is_ready:
                    logger.warning("window.nb did not load within timeout.")
                    return {}

                raw_details_str = await page.evaluate("""
                    () => {
                        const appState = window?.nb?.appState;
                        if (!appState || !appState.propertyDetails) return null;
                        const targetData = appState.propertyDetails.detailsData
                                           || appState.propertyDetails;
                        try {
                            return JSON.stringify(targetData);
                        } catch (err) {
                            const cache = new Set();
                            return JSON.stringify(targetData, (key, value) => {
                                if (typeof value === 'object' && value !== null) {
                                    if (cache.has(value)) return;
                                    cache.add(value);
                                }
                                return value;
                            });
                        }
                    }
                """)

                if not raw_details_str:
                    logger.warning("Evaluate returned null or empty.")
                    return {}

                details = json.loads(raw_details_str)
                logger.info(
                    "Raw JSON extracted | keys=%s",
                    list(details.keys())[:15],
                )

                return self._map_to_schema(url, details)

            except Exception as e:
                logger.error(
                    "Scraping failed | url=%s | error=%s",
                    url, str(e), exc_info=True,
                )
                raise
            finally:
                try:
                    if "context" in locals():
                        await context.close()
                    if "browser" in locals():
                        await browser.close()
                except Exception as cleanup_err:
                    logger.warning(
                        "Non-fatal cleanup error: %s", cleanup_err
                    )

    # ── JSON → ScrapedListing mapping ─────────────────────────

    def _map_to_schema(
        self, url: str, details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Maps the raw NoBroker JSON to ScrapedListing fields.
        Single place for all field extraction logic.
        """
        logger.info("Mapping raw JSON to ScrapedListing schema.")

        additional = details.get("aea__") or {}

        def get_display_value(key: str, default: str = "") -> str:
            entry = additional.get(key) or {}
            return entry.get("display_value") or default

        # ── Identity ──────────────────────────────────────────
        nobroker_id = details.get("id") or ""
        logger.debug("Extracted nobroker_id=%s", nobroker_id)

        # ── Location ──────────────────────────────────────────
        raw_society = details.get("society") or ""
        society = self._normalize_society(raw_society)
        city = (details.get("city") or "").strip().lower()
        logger.debug(
            "Location | city=%s | society_raw='%s' | society_norm='%s'",
            city, raw_society, society,
        )

        # ── Price ─────────────────────────────────────────────
        rent_val = details.get("rent")
        deposit_val = details.get("deposit")
        maint_val = (
            details.get("maintenanceAmount")
            if details.get("maintenance")
            else None
        )
        logger.debug(
            "Price | rent=%s | deposit=%s | maintenance=%s",
            rent_val, deposit_val, maint_val,
        )

        # ── Unit attributes ───────────────────────────────────
        building_type_code = details.get("buildingType") or ""
        prop_type_map = {"AP": "Apartment", "IH": "Independent House",
                         "IF": "Independent Floor", "VL": "Villa"}
        property_type = prop_type_map.get(
            building_type_code, building_type_code
        )

        # ── Amenities ─────────────────────────────────────────
        amenities = details.get("amenitiesMap") or {}
        logger.debug("Amenities map | %s", amenities)

        # ── Timestamps ────────────────────────────────────────
        creation_date = details.get("creationDate")
        last_update_date = details.get("lastUpdateDate")
        activation_date = details.get("activationDate")
        last_activation_date = details.get("lastActivationDate")
        reactivation_source = details.get("reactivationSource") or ""
        logger.debug(
            "Timestamps | created=%s | updated=%s | reactivation=%s",
            creation_date, last_update_date, reactivation_source,
        )

        # ── Owner ─────────────────────────────────────────────
        owner_id = details.get("ownerId") or ""
        logger.debug("Owner ID extracted | owner_id=%s", owner_id)

        # ── Description ───────────────────────────────────────
        description = (
            details.get("description")
            or details.get("combineDescription")
            or details.get("ownerDes")
            or ""
        )

        # ── Images ────────────────────────────────────────────
        photos_data = details.get("photos") or []
        image_urls = self._extract_images(photos_data)
        photos_meta = self._extract_photos_meta(photos_data)

        # ── Build validated model ─────────────────────────────
        listing_obj = ScrapedListing(
            # Identity
            url=url,
            platform="nobroker",
            nobroker_id=nobroker_id,
            title=(
                details.get("propertyTitle")
                or details.get("title")
                or ""
            ),
            # Location
            city=city,
            locality=(
                details.get("secondaryTitle")
                or details.get("completeStreetName")
                or details.get("street")
                or ""
            ),
            nb_locality=details.get("nbLocality") or "",
            nb_locality_id=details.get("nbLocalityId") or "",
            society=society,
            street=details.get("completeStreetName") or details.get("street") or "",
            latitude=details.get("latitude"),
            longitude=details.get("longitude"),
            accurate_location=details.get("accurateLocation"),
            location_accuracy_label=get_display_value(
                "ACCURATE_LOCATION"
            ),
            # Price
            rent=float(rent_val) if rent_val else None,
            deposit=float(deposit_val) if deposit_val else None,
            maintenance_amount=float(maint_val) if maint_val else None,
            # Unit attributes
            bedroom_configuration=details.get("typeDesc") or "",
            bath_val=(
                float(details["bathroom"])
                if details.get("bathroom") is not None
                else None
            ),
            area_sq_ft=(
                float(details["propertySize"])
                if details.get("propertySize") is not None
                else None
            ),
            floor=details.get("floor"),
            total_floor=details.get("totalFloor"),
            balcony=details.get("balconies"),
            facing=details.get("facingDesc") or "",
            furnishing_status=details.get("furnishingDesc") or "",
            property_type=property_type,
            building_type=building_type_code,
            property_age=details.get("propertyAge"),
            parking=details.get("parkingDesc") or "",
            water_supply=(
                (details.get("waterSupply") or "").replace("_", " ").title()
            ),
            lease_type=details.get("leaseTypeNew") or [],
            # Amenities
            has_gym=bool(amenities.get("GYM", False)),
            has_pool=bool(amenities.get("POOL", False)),
            has_lift=bool(amenities.get("LIFT", False) or details.get("lift", False)),
            has_security=bool(amenities.get("SECURITY", False)),
            # Additional
            pet_allowed=get_display_value("PET"),
            gated_security=get_display_value("GATED_SECURITY"),
            # Owner
            owner_id=owner_id,
            # Description
            description=description,
            # Timestamps
            creation_date=creation_date,
            last_update_date=last_update_date,
            activation_date=activation_date,
            last_activation_date=last_activation_date,
            reactivation_source=reactivation_source,
            # Images
            image_urls=image_urls,
            photos_meta=photos_meta,
            # Raw snapshot
            raw_json=details,
        )

        logger.info(
            "Schema mapping complete | nobroker_id=%s | images=%d | photos_meta=%d",
            nobroker_id, len(image_urls), len(photos_meta),
        )

        return listing_obj.model_dump(mode="json")

    # ── Async entrypoint ──────────────────────────────────────

    def _scrape_isolated(self, url: str) -> Dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._do_scrape(url))
        finally:
            loop.close()

    async def extract_listing(self, url: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._scrape_isolated, url)


# ── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":

    async def test_scraper():
        logger.info("Starting standalone NoBroker scraper test.")
        scraper = NoBrokerScraper()
        test_url = (
            "https://www.nobroker.in/property/"
            "1-rk-apartment-for-rent-in-1-cartarpuri-alia"
        )
        try:
            result = await scraper.extract_listing(test_url)
            print("\n--- Extracted Data ---")
            for key, value in result.items():
                if key == "raw_json":
                    print(f"{key}: <{len(str(value))} chars>")
                else:
                    print(f"{key}: {value}")
            logger.info("Scraper test completed successfully.")
        except Exception:
            logger.exception("Scraper test failed.")

    asyncio.run(test_scraper())
