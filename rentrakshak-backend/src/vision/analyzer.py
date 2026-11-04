
import os
import io
import time

import duckdb
import imagehash
import httpx

from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, Tuple, List

from src.core.interfaces import IRiskAnalyzer
from src.core.logger import get_logger

logger = get_logger(__name__)


class VisionAnomalyAnalyzer(IRiskAnalyzer):
    """
    Implements Layer 1 of the Vision Defense:
    Internal Perceptual Hash (pHash) Collision Detection.

    Detects potential reuse of identical or visually similar
    property images across different listings.

    Features:
    - Asynchronous image downloads.
    - Multiple image URL fallback support.
    - Perceptual hashing using pHash.
    - DuckDB-based persistent hash storage.
    - Hamming distance-based duplicate detection.
    - Comprehensive execution logging.
    """

    def __init__(
        self,
        db_path: str = "data/rentrakshak.duckdb",
        similarity_threshold: int = 8
    ):

        init_start = time.perf_counter()

        self.db_path = db_path
        self.similarity_threshold = similarity_threshold

        logger.info(
            "Initializing VisionAnomalyAnalyzer | "
            "db_path=%s | similarity_threshold=%d",
            self.db_path,
            self.similarity_threshold
        )

        try:
            # Resolve database path.
            absolute_db_path = os.path.abspath(self.db_path)

            self.db_path = absolute_db_path

            logger.debug(
                "Resolved DuckDB path | path=%s",
                self.db_path
            )

            # Ensure parent directory exists.
            db_directory = os.path.dirname(self.db_path)

            if db_directory:
                os.makedirs(
                    db_directory,
                    exist_ok=True
                )

                logger.debug(
                    "Database directory verified | path=%s",
                    db_directory
                )

            self._init_db()

            logger.info(
                "VisionAnomalyAnalyzer initialized successfully | "
                "duration=%.3fs",
                time.perf_counter() - init_start
            )

        except Exception:
            logger.exception(
                "VisionAnomalyAnalyzer initialization failed | "
                "db_path=%s",
                self.db_path
            )
            raise

    # ---------------------------------------------------------
    # Database initialization
    # ---------------------------------------------------------

    def _init_db(self):
        """
        Initializes the DuckDB table for storing image hashes.
        """

        db_start = time.perf_counter()

        logger.info(
            "Initializing DuckDB image hash storage | db_path=%s",
            self.db_path
        )

        try:
            with duckdb.connect(self.db_path) as con:

                con.execute("""
                    CREATE TABLE IF NOT EXISTS image_hashes (
                        listing_url VARCHAR,
                        image_url VARCHAR,
                        phash VARCHAR
                    )
                """)

                # Verify table creation.
                table_info = con.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'image_hashes'
                """).fetchall()

                record_count = con.execute("""
                    SELECT COUNT(*) FROM image_hashes
                """).fetchone()[0]

                logger.info(
                    "DuckDB table ready | "
                    "columns=%s | existing_records=%d",
                    table_info,
                    record_count
                )

            logger.info(
                "Database initialization completed | duration=%.3fs",
                time.perf_counter() - db_start
            )

        except Exception:
            logger.exception(
                "DuckDB initialization failed | db_path=%s",
                self.db_path
            )
            raise

    # ---------------------------------------------------------
    # Image downloading
    # ---------------------------------------------------------

    async def _download_image(
        self,
        urls: List[str]
    ) -> Tuple[Image.Image, str]:
        """
        Attempts to download an image from multiple URLs.

        Returns:
            Tuple containing:
            - Successfully decoded PIL image.
            - URL from which the image was downloaded.
        """

        download_start = time.perf_counter()

        logger.info(
            "Starting image download process | total_urls=%d",
            len(urls)
        )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
        }

        successful_downloads = 0

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(10.0)
        ) as client:

            for index, url in enumerate(urls, start=1):

                url_start = time.perf_counter()

                logger.info(
                    "Attempting image download | "
                    "url_index=%d/%d | url=%s",
                    index,
                    len(urls),
                    url
                )

                try:

                    response = await client.get(
                        url,
                        headers=headers
                    )

                    download_duration = (
                        time.perf_counter() - url_start
                    )

                    logger.info(
                        "Image HTTP response received | "
                        "status_code=%d | content_type=%s | "
                        "content_length=%d | duration=%.3fs",
                        response.status_code,
                        response.headers.get("Content-Type", ""),
                        len(response.content),
                        download_duration
                    )

                    response.raise_for_status()

                    # Validate content type.
                    content_type = response.headers.get(
                        "Content-Type",
                        ""
                    ).lower()

                    if not content_type.startswith("image/"):

                        logger.warning(
                            "Skipping URL because response is not an image | "
                            "url=%s | content_type=%s",
                            url,
                            content_type
                        )

                        continue

                    # Decode image.
                    decode_start = time.perf_counter()

                    img = Image.open(
                        io.BytesIO(response.content)
                    )

                    logger.info(
                        "Image decoded successfully | "
                        "format=%s | dimensions=%s | "
                        "mode=%s | decode_duration=%.3fs",
                        img.format,
                        img.size,
                        img.mode,
                        time.perf_counter() - decode_start
                    )

                    # Force actual decoding while response buffer is available.
                    img.load()

                    successful_downloads += 1

                    logger.info(
                        "Image download process succeeded | "
                        "successful_url=%s | total_duration=%.3fs",
                        url,
                        time.perf_counter() - download_start
                    )

                    return img, url

                except httpx.HTTPStatusError:

                    logger.exception(
                        "HTTP error while downloading image | url=%s",
                        url
                    )

                except httpx.TimeoutException:

                    logger.exception(
                        "Image download timed out | url=%s",
                        url
                    )

                except UnidentifiedImageError:

                    logger.exception(
                        "Downloaded content could not be decoded as an image | "
                        "url=%s",
                        url
                    )

                except Exception:

                    logger.exception(
                        "Unexpected image download failure | url=%s",
                        url
                    )

                logger.warning(
                    "Image URL failed. Trying next fallback URL | "
                    "url_index=%d",
                    index
                )

        logger.error(
            "All image URLs failed | total_urls=%d | duration=%.3fs",
            len(urls),
            time.perf_counter() - download_start
        )

        raise ValueError(
            "All provided image URLs failed to download or were invalid."
        )

    # ---------------------------------------------------------
    # Main analysis pipeline
    # ---------------------------------------------------------

    async def analyze(
        self,
        listing_data: Dict[str, Any]
    ) -> Tuple[float, str]:

        analysis_start = time.perf_counter()

        current_listing_url = None

        logger.info(
            "Starting vision anomaly analysis."
        )

        try:

            # Step 1: Validate input.
            if not isinstance(listing_data, dict):

                logger.error(
                    "Invalid listing_data type | received_type=%s",
                    type(listing_data).__name__
                )

                return 0.0, "Invalid listing data provided."

            current_listing_url = listing_data.get("url")

            image_urls = listing_data.get(
                "image_urls",
                []
            )

            logger.info(
                "Listing data received | "
                "listing_url=%s | image_count=%d",
                current_listing_url,
                len(image_urls) if isinstance(image_urls, list) else -1
            )

            if not current_listing_url:

                logger.warning(
                    "Listing URL missing from input data."
                )

            if not isinstance(image_urls, list):

                logger.warning(
                    "image_urls must be a list | received_type=%s",
                    type(image_urls).__name__
                )

                return 0.0, "Invalid image URL collection."

            if not image_urls:

                logger.warning(
                    "No images found for listing | listing_url=%s",
                    current_listing_url
                )

                return 0.0, "No images found to analyze."

            valid_urls = [
                url for url in image_urls
                if isinstance(url, str) and url.strip()
            ]

            if not valid_urls:

                logger.warning(
                    "No valid image URLs found | listing_url=%s",
                    current_listing_url
                )

                return 0.0, "No valid image URLs provided."

            logger.info(
                "Input validation completed | valid_image_urls=%d",
                len(valid_urls)
            )

            # Step 2: Download image.
            logger.info(
                "Step 1/4: Downloading listing image."
            )

            img, successful_url = await self._download_image(
                valid_urls
            )

            logger.info(
                "Image downloaded | listing_url=%s | image_url=%s",
                current_listing_url,
                successful_url
            )

            # Step 3: Generate perceptual hash.
            logger.info(
                "Step 2/4: Generating perceptual hash."
            )

            hash_start = time.perf_counter()

            img_hash = imagehash.phash(img)

            img_hash_str = str(img_hash)

            logger.info(
                "Perceptual hash generated | "
                "phash=%s | hash_duration=%.3fs",
                img_hash_str,
                time.perf_counter() - hash_start
            )

            # Step 4: Check existing hashes.
            logger.info(
                "Step 3/4: Checking existing image hashes in DuckDB."
            )

            db_start = time.perf_counter()

            with duckdb.connect(self.db_path) as con:

                query = """
                    SELECT listing_url, image_url, phash
                    FROM image_hashes
                    WHERE listing_url IS DISTINCT FROM ?
                """

                existing_records = con.execute(
                    query,
                    [current_listing_url]
                ).fetchall()

                logger.info(
                    "Existing image hashes retrieved | "
                    "record_count=%d | query_duration=%.3fs",
                    len(existing_records),
                    time.perf_counter() - db_start
                )

                # Step 5: Compare Hamming distances.
                logger.info(
                    "Step 4/4: Calculating Hamming distances | "
                    "comparison_count=%d | threshold=%d",
                    len(existing_records),
                    self.similarity_threshold
                )

                comparison_start = time.perf_counter()

                closest_match = None
                minimum_distance = None

                for record_url, record_image_url, record_hash_str in existing_records:

                    try:

                        stored_hash = imagehash.hex_to_hash(
                            record_hash_str
                        )

                        hamming_distance = img_hash - stored_hash

                        logger.debug(
                            "Image hash comparison | "
                            "existing_listing=%s | "
                            "existing_image=%s | "
                            "stored_hash=%s | "
                            "hamming_distance=%d",
                            record_url,
                            record_image_url,
                            record_hash_str,
                            hamming_distance
                        )

                        if (
                            minimum_distance is None
                            or hamming_distance < minimum_distance
                        ):

                            minimum_distance = hamming_distance

                            closest_match = record_url

                        if hamming_distance < self.similarity_threshold:

                            logger.warning(
                                "Potential duplicate image detected | "
                                "current_listing=%s | "
                                "matched_listing=%s | "
                                "hamming_distance=%d | threshold=%d",
                                current_listing_url,
                                record_url,
                                hamming_distance,
                                self.similarity_threshold
                            )

                            score = 100.0

                            proof = (
                                "Photo reused in another listing: "
                                f"{record_url}"
                            )

                            logger.info(
                                "Vision analysis completed with duplicate detected | "
                                "score=%.2f | duration=%.3fs",
                                score,
                                time.perf_counter() - analysis_start
                            )

                            return score, proof

                    except Exception:

                        logger.exception(
                            "Failed to compare stored image hash | "
                            "listing_url=%s | stored_hash=%s",
                            record_url,
                            record_hash_str
                        )

                        continue

                logger.info(
                    "Image comparison completed | "
                    "comparisons=%d | closest_match=%s | "
                    "minimum_hamming_distance=%s | duration=%.3fs",
                    len(existing_records),
                    closest_match,
                    minimum_distance,
                    time.perf_counter() - comparison_start
                )

                # Step 6: Store current image hash.
                logger.info(
                    "Storing image hash in DuckDB | "
                    "listing_url=%s | image_url=%s",
                    current_listing_url,
                    successful_url
                )

                insert_start = time.perf_counter()

                con.execute(
                    """
                    INSERT INTO image_hashes (
                        listing_url,
                        image_url,
                        phash
                    )
                    VALUES (?, ?, ?)
                    """,
                    [
                        current_listing_url,
                        successful_url,
                        img_hash_str
                    ]
                )

                logger.info(
                    "Image hash stored successfully | "
                    "insert_duration=%.3fs",
                    time.perf_counter() - insert_start
                )

            logger.info(
                "Vision analysis completed successfully | "
                "listing_url=%s | score=0.0 | "
                "duplicate_found=False | total_duration=%.3fs",
                current_listing_url,
                time.perf_counter() - analysis_start
            )

            return (
                0.0,
                "Image appears authentic. No internal duplicates found."
            )

        except ValueError as ve:

            logger.warning(
                "Vision analysis skipped due to invalid or unavailable image | "
                "listing_url=%s | reason=%s | duration=%.3fs",
                current_listing_url,
                str(ve),
                time.perf_counter() - analysis_start
            )

            return (
                0.0,
                f"Vision analysis skipped: {str(ve)}"
            )

        except Exception:

            logger.exception(
                "Vision analysis failed | "
                "listing_url=%s | duration=%.3fs",
                current_listing_url,
                time.perf_counter() - analysis_start
            )

            return (
                0.0,
                "Vision analysis skipped due to processing error."
            )


# ---------------------------------------------------------
# Quick Test Block
# ---------------------------------------------------------

if __name__ == "__main__":

    import asyncio

    async def test_vision():

        logger.info(
            "Starting standalone VisionAnomalyAnalyzer test."
        )

        analyzer = VisionAnomalyAnalyzer()

        # Simulating two listings using the exact same image.
        listing_1 = {
            "url": "https://example.com/listing-1",
            "image_urls": [
                "https://picsum.photos/id/237/200/300"
            ]
        }

        listing_2 = {
            "url": "https://example.com/listing-2",
            "image_urls": [
                "https://picsum.photos/id/237/200/300"
            ]
        }

        logger.info("Processing Listing 1.")

        score1, proof1 = await analyzer.analyze(listing_1)

        print(
            f"Listing 1 | Score: {score1} | Proof: {proof1}\n"
        )

        logger.info("Processing Listing 2 (expected duplicate).")

        score2, proof2 = await analyzer.analyze(listing_2)

        print(
            f"Listing 2 | Score: {score2} | Proof: {proof2}"
        )

        logger.info(
            "Standalone vision test completed."
        )

    asyncio.run(test_vision())
