
import asyncio
import time
import json

from datetime import datetime, timezone, timedelta
from playwright.async_api import async_playwright
from typing import Dict, Any, List

from src.core.interfaces import IListingScraper
from src.core.logger import get_logger
from src.schemas.listing import ScrapedListing

logger = get_logger(__name__)


class NoBrokerScraper(IListingScraper):
    """
    Playwright-based dynamic scraper for NoBroker/99acres.

    Features:
    - Dynamic DOM extraction.
    - Safe selector handling.
    - Image URL extraction.
    - Price, area and property metadata extraction.
    - Comprehensive execution logging.
    """


    async def _extract_images(self, page) -> List[str]:
        """Extracts property images by explicitly targeting the images.nobroker.in CDN."""
        try:
            image_urls = await page.evaluate("""
                Array.from(document.querySelectorAll('img'))
                     .map(img => img.src)
                     .filter(src => {
                         if (!src || !src.includes('http')) return false;
                         
                         const lowerSrc = src.toLowerCase();
                         
                         // 1. POSITIVE FILTER: Only accept images from the property CDN
                         if (!lowerSrc.includes('images.nobroker.in')) {
                             return false;
                         }
                         
                         // 2. NEGATIVE FILTER: Exclude stray UI elements just in case
                         if (lowerSrc.includes('logo') || 
                             lowerSrc.includes('icon') || 
                             lowerSrc.includes('avatar') || 
                             lowerSrc.includes('.svg') ||
                             lowerSrc.includes('placeholder')) {
                             return false;
                         }
                         
                         return true;
                     })
            """)
            
            # Deduplicate URLs (sometimes desktop and mobile views load the same image twice)
            unique_urls = list(dict.fromkeys(image_urls))
            logger.info(unique_urls)
            
            # Return up to 5 actual property images for vision analysis
            return unique_urls[:5] if unique_urls else []
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
            return []# ---------------------------------------------------------
    # Synchronous scraping implementation
    # ---------------------------------------------------------

    async def _do_scrape(self, url: str) -> Dict[str, Any]:
        logger.info(f"Starting isolated async Playwright scraper for URL: {url}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            try:
                await asyncio.sleep(3)
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                
                # Polling for app state
                logger.debug("Polling for window.nb object...")
                is_ready = False
                for _ in range(15):
                    is_ready = await page.evaluate("() => typeof window !== 'undefined' && typeof window.nb !== 'undefined'")
                    if is_ready:
                        break
                    await asyncio.sleep(1)

                if not is_ready:
                    logger.warning("window.nb object did not load within timeout.")
                    return {}

                # Stringify robustly in the browser
                raw_details_str = await page.evaluate("""
                    () => {
                        const appState = window?.nb?.appState;
                        if (!appState || !appState.propertyDetails) return null;
                        const targetData = appState.propertyDetails.detailsData || appState.propertyDetails;
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
                    logger.warning("Evaluate returned null or empty string.")
                    return {}

                details = json.loads(raw_details_str)
                logger.info(f"Extracted raw details: {details}")
                additional = details.get("aea__") or {}

                def get_display_value(key: str, default: str = "") -> str:
                    value = additional.get(key) or {}
                    return value.get("display_value") or default

                # Mapping to ScrapedListing Schema
                rent_val = details.get("rent")
                maint_val = details.get("maintenanceAmount") if details.get("maintenance") else None
                area_val = details.get("propertySize")
                deposit_val = details.get("deposit")
                bath_val = details.get("bathroom")

                bedroom_config = details.get("typeDesc") or ""
                building_type = details.get("buildingType")
                prop_type = {"AP": "Apartment"}.get(building_type, building_type or "")
                tenant_values = details.get("leaseTypeNew") or []
                pref_tenant = {"ANYONE": "Anyone"}.get(tenant_values[0], tenant_values[0]) if tenant_values else ""
                possession_val = {"READY": "Immediately"}.get(details.get("state"), details.get("formattedAvailableFrom") or "")
                
                current_situation = get_display_value("CURRENT_SITUATION")
                property_age = details.get("propertyAge")
                building_age = "Newly Constructed" if current_situation == "New Property" or property_age == 0 else (current_situation or str(property_age or ""))
                
                balconies = details.get("balconies")
                balcony_val = "NA" if balconies == 0 else str(balconies) if balconies is not None else ""

                posted_timestamp = details.get("reactivationReqDate") or details.get("activationDate") or details.get("creationDate")
                posted_on_val = ""
                if posted_timestamp:
                    try:
                        india_tz = timezone(timedelta(hours=5, minutes=30))
                        posted_on_val = datetime.fromtimestamp(posted_timestamp / 1000, tz=india_tz).strftime("%b %d, %Y")
                    except Exception:
                        pass

                floor_number = details.get("floor")
                total_floors = details.get("totalFloor")
                floor_val = f"{floor_number}/{total_floors}" if floor_number is not None and total_floors is not None else ""

                image_urls = await self._extract_images(page)

                # Instantiate the Pydantic model for strict validation
                listing_obj = ScrapedListing(
                    url=url,
                    platform="nobroker",
                    title=details.get("propertyTitle") or details.get("title") or "",
                    locality=details.get("secondaryTitle") or details.get("completeStreetName") or details.get("street") or "",
                    rent=float(rent_val) if rent_val else None,
                    maintenance_amount=float(maint_val) if maint_val else None,
                    area_sq_ft=float(area_val) if area_val else None,
                    deposit=float(deposit_val) if deposit_val else None,
                    bedroom_configuration=bedroom_config,
                    property_type=prop_type,
                    preferred_tenant=pref_tenant,
                    possession=possession_val,
                    parking=details.get("parkingDesc") or "",
                    building_age=building_age,
                    balcony=balcony_val,
                    posted_on=posted_on_val,
                    furnishing_status=details.get("furnishingDesc") or "",
                    facing=details.get("facingDesc") or "",
                    water_supply=(details.get("waterSupply") or "").replace("_", " ").title(),
                    floor=floor_val,
                    bathroom=int(bath_val) if bath_val else None,
                    pet_allowed=get_display_value("PET"),
                    non_veg_allowed=get_display_value("NON_VEG_ALLOWED"),
                    gated_security=get_display_value("GATED_SECURITY"),
                    description=details.get("description") or details.get("combineDescription") or details.get("ownerDes") or "",
                    image_urls=image_urls
                )

                logger.info("Successfully validated schema mapping.")
                # We return the model_dump() (dictionary) so it satisfies the IListingScraper interface contract
                return listing_obj.model_dump(mode='json')

            except Exception as e:
                logger.error(f"Scraping failed for {url} | Error: {str(e)}", exc_info=True)
                raise e
            finally:
                try:
                    if 'context' in locals():
                        await context.close()
                    if 'browser' in locals():
                        await browser.close()
                except Exception as cleanup_err:
                    logger.warning(f"Non-fatal cleanup error ignored: {cleanup_err}")

    def _scrape_isolated(self, url: str) -> Dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._do_scrape(url))
        finally:
            loop.close()

    async def extract_listing(self, url: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._scrape_isolated, url)
        """Implements the IListingScraper interface."""
        return await asyncio.to_thread(self._scrape_isolated, url)

# ---------------------------------------------------------
# Quick Test Block
# ---------------------------------------------------------

if __name__ == "__main__":

    async def test_scraper():

        logger.info(
            "Starting standalone NoBroker scraper test."
        )

        scraper = NoBrokerScraper()

        test_url = (
            "https://www.nobroker.in/property/"
            "1-rk-apartment-for-rent-in-1-cartarpuri-alia"
        )

        try:

            result = await scraper.extract_listing(test_url)

            print("\n--- Extracted Data ---")

            for key, value in result.items():

                print(f"{key}: {value}")

            logger.info(
                "Standalone scraper test completed successfully."
            )

        except Exception:

            logger.exception(
                "Standalone scraper test failed."
            )

    asyncio.run(test_scraper())
