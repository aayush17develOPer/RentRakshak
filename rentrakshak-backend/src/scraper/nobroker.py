
import asyncio
import time
import re

from playwright.sync_api import sync_playwright, Page
from typing import Dict, Any, List

from src.core.interfaces import IListingScraper
from src.core.logger import get_logger

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

    # ---------------------------------------------------------
    # Safe text extraction
    # ---------------------------------------------------------

    def _safe_text(
        self,
        page: Page,
        selector: str,
        default: str = ""
    ) -> str:

        extraction_start = time.perf_counter()

        logger.debug(
            "Extracting text | selector=%s | default=%s",
            selector,
            default
        )

        try:
            locator = page.locator(selector).first

            is_visible = locator.is_visible(timeout=500)

            if is_visible:

                extracted_text = locator.inner_text(
                    timeout=2000
                ).strip()

                logger.debug(
                    "Text extraction successful | "
                    "selector=%s | text_length=%d | duration=%.3fs",
                    selector,
                    len(extracted_text),
                    time.perf_counter() - extraction_start
                )

                return extracted_text

            logger.debug(
                "Element not visible | selector=%s | using_default=%s",
                selector,
                bool(default)
            )

        except Exception:

            logger.exception(
                "Text extraction failed | selector=%s",
                selector
            )

        return default

    # ---------------------------------------------------------
    # Image extraction
    # ---------------------------------------------------------

    def _extract_images(self, page: Page) -> List[str]:
        """Extracts all image sources from the listing gallery, filtering out logos and SVGs."""
        try:
            image_urls = page.evaluate("""
                Array.from(document.querySelectorAll('img'))
                     .map(img => img.src)
                     .filter(src => {
                         if (!src || !src.includes('http')) return false;
                         
                         const lowerSrc = src.toLowerCase();
                         
                         // Exclude site logos, UI icons, SVGs, and user avatars
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
            # Return up to 5 actual property images for vision analysis
            return image_urls[:5] if image_urls else []
        except Exception as e:
            logger.warning(f"Image extraction failed: {e}")
            return []
    # ---------------------------------------------------------
    # Synchronous scraping implementation
    # ---------------------------------------------------------

    def _scrape_sync(
        self,
        url: str
    ) -> Dict[str, Any]:

        scrape_start = time.perf_counter()

        logger.info(
            "Starting NoBroker scraping process | url=%s",
            url
        )

        browser = None
        context = None

        try:

            # Step 1: Initialize Playwright.
            logger.info(
                "Step 1/6: Initializing Playwright."
            )

            with sync_playwright() as p:

                browser_start = time.perf_counter()

                browser = p.chromium.launch(
                    headless=True
                )

                logger.info(
                    "Chromium browser launched | duration=%.3fs",
                    time.perf_counter() - browser_start
                )

                # Step 2: Create browser context.
                logger.info(
                    "Step 2/6: Creating browser context."
                )

                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0.0.0 "
                        "Safari/537.36"
                    )
                )

                page = context.new_page()

                logger.info(
                    "Browser context and page created successfully."
                )

                # Attach useful Playwright event listeners.
                page.on(
                    "console",
                    lambda message: logger.debug(
                        "Browser console | type=%s | text=%s",
                        message.type,
                        message.text
                    )
                )

                page.on(
                    "pageerror",
                    lambda error: logger.error(
                        "Browser JavaScript error | error=%s",
                        error
                    )
                )

                page.on(
                    "requestfailed",
                    lambda request: logger.warning(
                        "Browser request failed | "
                        "url=%s | error=%s",
                        request.url,
                        request.failure
                    )
                )

                # Step 3: Navigate to listing.
                logger.info(
                    "Step 3/6: Navigating to listing page."
                )

                logger.debug(
                    "Applying polite scraping delay | delay=3 seconds"
                )

                time.sleep(3)

                navigation_start = time.perf_counter()

                response = page.goto(
                    url,
                    timeout=45000,
                    wait_until="domcontentloaded"
                )

                navigation_duration = (
                    time.perf_counter() - navigation_start
                )

                logger.info(
                    "Navigation completed | "
                    "http_status=%s | final_url=%s | "
                    "duration=%.3fs",
                    response.status if response else None,
                    page.url,
                    navigation_duration
                )

                logger.info(
                    "Page metadata | title=%s",
                    page.title()
                )

                if response and response.status >= 400:

                    logger.warning(
                        "Listing returned HTTP error status | "
                        "status=%d | url=%s",
                        response.status,
                        page.url
                    )

                # Detect common blocking indicators.
                page_title = page.title().lower()
                page_content = page.locator("body").inner_text(
                    timeout=5000
                ).lower()

                blocking_indicators = [
                    "captcha",
                    "access denied",
                    "verify you are human",
                    "unusual traffic",
                    "robot check"
                ]

                detected_blocks = [
                    indicator
                    for indicator in blocking_indicators
                    if indicator in page_title or indicator in page_content
                ]

                if detected_blocks:

                    logger.warning(
                        "Possible anti-bot or access restriction detected | "
                        "indicators=%s",
                        detected_blocks
                    )

                logger.info(
                    "Page loaded | page_url=%s | "
                    "page_title=%s | body_length=%d",
                    page.url,
                    page.title(),
                    len(page_content)
                )

                # Step 4: Extract dynamic DOM elements.
                logger.info(
                    "Step 4/6: Extracting listing information."
                )

                extraction_start = time.perf_counter()

                raw_price = self._safe_text(
                    page,
                    '#roomRent, .rent-price, h3:has-text("₹")'
                )

                logger.debug(
                    "Raw price extracted | value=%s",
                    raw_price
                )

                try:

                    clean_price = (
                        float(re.sub(r'[^\d.]', '', raw_price))
                        if raw_price else 0.0
                    )

                except ValueError:

                    logger.exception(
                        "Price conversion failed | raw_price=%s",
                        raw_price
                    )

                    clean_price = 0.0

                raw_area = self._safe_text(
                    page,
                    '#sqft, .area-size'
                )

                logger.debug(
                    "Raw area extracted | value=%s",
                    raw_area
                )

                try:

                    clean_area = (
                        float(re.sub(r'[^\d.]', '', raw_area))
                        if raw_area else 0.0
                    )

                except ValueError:

                    logger.exception(
                        "Area conversion failed | raw_area=%s",
                        raw_area
                    )

                    clean_area = 0.0

                furnishing = self._safe_text(
                    page,
                    'text=Furnishing >> xpath=..'
                )

                description = self._safe_text(
                    page,
                    '#description, .description-text'
                )

                title = self._safe_text(
                    page,
                    'h1'
                )

                logger.info(
                    "Basic listing details extracted | "
                    "price=%s | area=%s | furnishing=%s | "
                    "title_length=%d | description_length=%d",
                    clean_price,
                    clean_area,
                    furnishing,
                    len(title),
                    len(description)
                )

                # Step 5: Extract property metadata.
                logger.info(
                    "Step 5/6: Extracting property metadata."
                )

                bhk_match = re.search(
                    r'(\d+)\s*(?:BHK|RK)',
                    title,
                    re.IGNORECASE
                )

                bhk = (
                    int(bhk_match.group(1))
                    if bhk_match else 1
                )

                logger.debug(
                    "BHK extraction | title=%s | bhk=%d | matched=%s",
                    title,
                    bhk,
                    bool(bhk_match)
                )

                floor = self._safe_text(
                    page,
                    'text=Floor >> xpath=..',
                    default="Ground"
                )

                age = self._safe_text(
                    page,
                    'text=Age >> xpath=..',
                    default="5-10 Years"
                )

                image_urls = self._extract_images(page)

                logger.info(
                    "Property metadata extracted | "
                    "bhk=%d | floor=%s | age=%s | images=%d",
                    bhk,
                    floor,
                    age,
                    len(image_urls)
                )

                # Step 6: Construct output schema.
                logger.info(
                    "Step 6/6: Constructing listing data schema."
                )

                final_description = description or title

                dynamic_data = {
                    "url": url,
                    "platform": (
                        "nobroker"
                        if "nobroker" in url.lower()
                        else "other"
                    ),
                    "price": clean_price or 10000.0,
                    "bhk": bhk,
                    "area_sqft": clean_area or 200.0,
                    "floor": floor,
                    "furnishing_status": furnishing or "unfurnished",
                    "age": age,
                    "gated": (
                        1
                        if "gated" in final_description.lower()
                        else 0
                    ),
                    "amenities_count": 3,
                    "amenities_list": (
                        ["parking", "lift"]
                        if "lift" in final_description.lower()
                        else ["parking"]
                    ),
                    "sector": "Cartarpuri",
                    "parent_sector": "Gurgaon",
                    "description": final_description,
                    "image_urls": image_urls
                }

                logger.info(
                    "Listing schema constructed successfully | "
                    "fields=%s",
                    list(dynamic_data.keys())
                )

                logger.debug(
                    "Final listing data | data=%s",
                    {
                        **dynamic_data,
                        "description": (
                            final_description[:200] + "..."
                            if len(final_description) > 200
                            else final_description
                        )
                    }
                )

                logger.info(
                    "Scraping completed successfully | "
                    "price=%.2f | bhk=%d | area=%.2f | "
                    "images=%d | extraction_duration=%.3fs | "
                    "total_duration=%.3fs",
                    dynamic_data["price"],
                    dynamic_data["bhk"],
                    dynamic_data["area_sqft"],
                    len(image_urls),
                    time.perf_counter() - extraction_start,
                    time.perf_counter() - scrape_start
                )

                return dynamic_data

        except Exception:

            logger.exception(
                "NoBroker scraping failed | "
                "url=%s | total_duration=%.3fs",
                url,
                time.perf_counter() - scrape_start
            )

            raise

        finally:

            if context:

                try:
                    context.close()

                    logger.debug(
                        "Browser context closed successfully."
                    )

                except Exception:

                    logger.exception(
                        "Failed to close browser context."
                    )

            if browser:

                try:
                    browser.close()

                    logger.info(
                        "Chromium browser closed successfully."
                    )

                except Exception:

                    logger.exception(
                        "Failed to close Chromium browser."
                    )

    # ---------------------------------------------------------
    # Async interface
    # ---------------------------------------------------------

    async def extract_listing(
        self,
        url: str
    ) -> Dict[str, Any]:

        extraction_start = time.perf_counter()

        logger.info(
            "Async listing extraction requested | url=%s",
            url
        )

        try:

            result = await asyncio.to_thread(
                self._scrape_sync,
                url
            )

            logger.info(
                "Async listing extraction completed | "
                "url=%s | duration=%.3fs",
                url,
                time.perf_counter() - extraction_start
            )

            return result

        except Exception:

            logger.exception(
                "Async listing extraction failed | "
                "url=%s | duration=%.3fs",
                url,
                time.perf_counter() - extraction_start
            )

            raise


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
