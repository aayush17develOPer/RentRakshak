# src/scraper/nobroker.py
import asyncio
from playwright.async_api import async_playwright
from typing import Dict, Any
from src.core.interfaces import IListingScraper

class NoBrokerScraper(IListingScraper):
    """
    Playwright-based scraper for NoBroker/99acres.
    Implements the IListingScraper interface.
    """
    
    async def extract_listing(self, url: str) -> Dict[str, Any]:
        async with async_playwright() as p:
            # Launch headless browser for zero-budget constraints
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                # Use a standard user agent to avoid basic bot blocks
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            try:
                # 1. Polite scraping: Enforce the 3-second delay
                await asyncio.sleep(3)
                await page.goto(url, timeout=60000, wait_until="domcontentloaded")
                
                # 2. Extract DOM Elements 
                # Note: These are example selectors based on typical NoBroker structure.
                # In production, use robust XPath or CSS selectors.
                
                # Example: Extracting price from a specific class/id
                # raw_price = await page.locator('#roomRent').inner_text()
                
                # Example: Relational extraction using the requirement's exact logic:
                # furnishing = await page.locator('text=Furnishing').locator('..').inner_text()

                # 3. Assemble and return the raw dictionary
                # (Mocked data representing a successful parse for LLD purposes)
                raw_data = {
                    "url": url,
                    "platform": "nobroker",
                    "price": 8000.0,
                    "bhk": 1,
                    "area_sqft": 180.0,
                    "floor": "Ground",
                    "furnishing_status": "unfurnished",
                    "age": "5-10 Years",
                    "gated": 0,
                    "amenities_count": 1,
                    "amenities_list": ["parking"],
                    "sector": "Cartarpuri",
                    "parent_sector": "Gurgaon",
                    "description": "Standard 1 RK available for rent immediately. No brokers.",
                    "image_urls": [
                        "https://example-image-bucket.com/room_view_1.jpg",
                        "https://example-image-bucket.com/kitchen.jpg"
                    ]
                }
                
                return raw_data
                
            except Exception as e:
                # Log failure without crashing the FastAPI thread
                print(f"Scraping failed for {url}: {str(e)}")
                raise e
            finally:
                await browser.close()

# --- Quick Test Block ---
if __name__ == "__main__":
    async def test_scraper():
        scraper = NoBrokerScraper()
        result = await scraper.extract_listing("https://www.nobroker.in/property/1-rk-apartment-for-rent-in-1-cartarpuri-alia")
        print(result)
        
    asyncio.run(test_scraper())