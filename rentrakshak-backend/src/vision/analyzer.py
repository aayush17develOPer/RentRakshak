# src/vision/analyzer.py
import duckdb
import imagehash
from PIL import Image
import httpx
import io
from typing import Dict, Any, Tuple
from src.core.interfaces import IRiskAnalyzer

class VisionAnomalyAnalyzer(IRiskAnalyzer):
    """
    Implements Layer 1 of the Vision Defense: Internal pHash Collisions.
    Analyzes images to detect if brokers are reusing the same fake photos.
    """
    
    def __init__(self, db_path: str = "data/rentrakshak.duckdb"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the DuckDB table for storing image hashes if it doesn't exist."""
        with duckdb.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS image_hashes (
                    listing_url VARCHAR,
                    image_url VARCHAR,
                    phash VARCHAR
                )
            """)

    async def _download_image(self, urls: list[str]) -> tuple[Image.Image, str]:
        """
        Attempts to download an image from a list of URLs with production safeguards.
        Returns the PIL Image and the URL that successfully downloaded.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
        }
        
        # follow_redirects=True is critical for CDN-hosted images
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for url in urls:
                try:
                    response = await client.get(url, timeout=10.0, headers=headers)
                    response.raise_for_status()
                    
                    # Validate that the response is actually an image
                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        print(f"Skipping {url}: Invalid content type '{content_type}'")
                        continue
                        
                    img = Image.open(io.BytesIO(response.content))
                    return img, url
                    
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    continue  # Try the next URL in the list
                    
        raise ValueError("All provided image URLs failed to download or were invalid.")

    async def analyze(self, listing_data: Dict[str, Any]) -> Tuple[float, str]:
        image_urls = listing_data.get("image_urls", [])
        if not image_urls:
            return 0.0, "No images found to analyze."

        current_listing_url = listing_data.get("url")
        
        try:
            # 1. Download with fallback support
            img, successful_url = await self._download_image(image_urls)
            img_hash = imagehash.phash(img)
            
            with duckdb.connect(self.db_path) as con:
                # 2. Fetch existing hashes to check for collisions
                query = "SELECT listing_url, phash FROM image_hashes WHERE listing_url != ?"
                existing_records = con.execute(query, [current_listing_url]).fetchall()
                
                # 3. Calculate Hamming Distance
                for record_url, record_hash_str in existing_records:
                    stored_hash = imagehash.hex_to_hash(record_hash_str)
                    
                    if img_hash - stored_hash < 8:
                        score = 100.0
                        proof = f"Photo reused in another listing: {record_url}"
                        return score, proof
                
                # 4. Store the hash of the image that actually worked
                con.execute("""
                    INSERT INTO image_hashes (listing_url, image_url, phash) 
                    VALUES (?, ?, ?)
                """, [current_listing_url, successful_url, str(img_hash)])
            
            return 0.0, "Image appears authentic. No internal duplicates found."
            
        except ValueError as ve:
            # Graceful degradation: The URLs were unreachable, but we shouldn't crash the ensemble
            return 0.0, f"Vision analysis skipped: {str(ve)}"
        except Exception as e:
            print(f"Vision analysis failed for {current_listing_url}: {e}")
            return 0.0, "Vision analysis skipped due to processing error."
        
# --- Quick Test Block ---
if __name__ == "__main__":
    import asyncio
    
    async def test_vision():
        analyzer = VisionAnomalyAnalyzer()
        
        # Simulating two listings using the exact same image
        listing_1 = {
            "url": "https://example.com/listing-1",
            "image_urls": ["https://picsum.photos/id/237/200/300"] # Dummy dog image
        }
        listing_2 = {
            "url": "https://example.com/listing-2",
            "image_urls": ["https://picsum.photos/id/237/200/300"] # Same dummy dog image
        }
        
        print("Processing Listing 1...")
        score1, proof1 = await analyzer.analyze(listing_1)
        print(f"Score: {score1}, Proof: {proof1}\n")
        
        print("Processing Listing 2 (Should flag as duplicate)...")
        score2, proof2 = await analyzer.analyze(listing_2)
        print(f"Score: {score2}, Proof: {proof2}")

    asyncio.run(test_vision())