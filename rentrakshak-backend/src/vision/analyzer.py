
import os
import io
import time

import duckdb
import imagehash
import httpx
import math

from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, Optional, Tuple, List

from src.core.interfaces import IRiskAnalyzer
from src.core.logger import get_logger

logger = get_logger(__name__)

@staticmethod
def _is_location_nearby(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
    threshold_meters: float = 500.0
) -> bool:
    """
    Calculates the Haversine distance between two coordinates.
    Returns True if the distance is within the threshold (e.g., same society/building).
    Returns False if any coordinate is missing or distance exceeds threshold.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return False
        
    R = 6371000  # Radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi_1) * math.cos(phi_2) * \
        math.sin(delta_lambda / 2.0) ** 2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance <= threshold_meters


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
                        phash VARCHAR,
                        latitude FLOAT,
                        longitude FLOAT,
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

    async def _download_images(
        self,
        urls: List[str]
    ) -> List[Tuple[Image.Image, str]]:
        """
        Attempts to download an image from multiple URLs.

        Returns:
            List of tuples containing:
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

        downloaded_images : List[Tuple[Image.Image, str]] = []

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(50.0)
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

                    download_duration = time.perf_counter() - url_start

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

                    # Force actual decoding.
                    img.load()

                    logger.info(
                        "Image decoded successfully | "
                        "format=%s | dimensions=%s | "
                        "mode=%s | decode_duration=%.3fs",
                        img.format,
                        img.size,
                        img.mode,
                        time.perf_counter() - decode_start
                    )

                    successful_downloads += 1

                    # Accumulate image and its corresponding URL.
                    downloaded_images.append((img, url))

                    logger.info(
                        "Image download succeeded | "
                        "successful_url=%s | successful_downloads=%d | "
                        "total_duration=%.3fs",
                        url,
                        successful_downloads,
                        time.perf_counter() - download_start
                    )

                    continue


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
                    "Image URL failed. Continuing to next URL | "
                    "url_index=%d",
                    index
                )

            # Return all successfully downloaded images after loop completion.
            logger.info(
                "Image download batch completed | "
                "total_urls=%d | successful_downloads=%d | "
                "failed_downloads=%d | total_duration=%.3fs",
                len(urls),
                len(downloaded_images),
                len(urls) - len(downloaded_images),
                time.perf_counter() - download_start
            )

            return downloaded_images

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

                return 0.0, "Listing URL missing from input data."

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

            # Step 2: Download all valid images.
            logger.info(
                "Step 1/4: Downloading listing images."
            )

            download_start = time.perf_counter()

            downloaded_images = await self._download_images(valid_urls)

            if not downloaded_images:

                logger.warning(
                    "No images could be downloaded | "
                    "listing_url=%s | attempted_urls=%d | duration=%.3fs",
                    current_listing_url,
                    len(valid_urls),
                    time.perf_counter() - download_start
                )

                return (
                    0.0,
                    "Vision analysis skipped: No valid images could be downloaded."
                )

            logger.info(
                "Images downloaded successfully | "
                "listing_url=%s | downloaded_count=%d | "
                "attempted_urls=%d | duration=%.3fs",
                current_listing_url,
                len(downloaded_images),
                len(valid_urls),
                time.perf_counter() - download_start
            )

            # Step 3: Analyze each downloaded image.
            logger.info(
                "Step 2/4: Starting individual image analysis."
            )

            duplicate_matches = []

            successful_analyses = 0
            failed_analyses = 0

            total_images = len(downloaded_images)

            # Open one DuckDB connection for the entire batch.
            with duckdb.connect(self.db_path) as con:

                existing_records = con.execute("""
                SELECT listing_url, image_url, phash, latitude, longitude
                FROM image_hashes
                WHERE listing_url IS DISTINCT FROM ?
                """, [current_listing_url]).fetchall()

                for index, image_data in enumerate(
                    downloaded_images,
                    start=1
                ):

                    # Validate downloaded image tuple.
                    if (
                        not isinstance(image_data, tuple)
                        or len(image_data) != 2
                    ):

                        failed_analyses += 1

                        logger.warning(
                            "Invalid downloaded image entry | "
                            "image_index=%d/%d | received_type=%s",
                            index,
                            total_images,
                            type(image_data).__name__
                        )

                        continue

                    img, image_url = image_data

                    logger.info(
                        "Processing downloaded image | "
                        "image_index=%d/%d | image_url=%s",
                        index,
                        total_images,
                        image_url
                    )

                    try:

                        # Analyze one image using the helper function.
                        matched_listing = await self._analyze_single_image(
                            img=img,
                            image_url=image_url,
                            current_listing_url=current_listing_url,
                            latitude=listing_data.get("latitude"),
                            longitude=listing_data.get("longitude"),
                            existing_records=existing_records,
                            con=con
                        )

                        successful_analyses += 1

                        if matched_listing:

                            duplicate_matches.append(
                                (image_url, matched_listing)
                            )

                            logger.warning(
                                "Duplicate image detected | "
                                "image_index=%d/%d | "
                                "image_url=%s | matched_listing=%s",
                                index,
                                total_images,
                                image_url,
                                matched_listing
                            )

                        else:

                            logger.info(
                                "No duplicate detected for image | "
                                "image_index=%d/%d | image_url=%s",
                                index,
                                total_images,
                                image_url
                            )

                    except Exception:

                        failed_analyses += 1

                        logger.exception(
                            "Individual image analysis failed | "
                            "image_index=%d/%d | image_url=%s",
                            index,
                            total_images,
                            image_url
                        )


            # Step 4: Aggregate results from all images.
            logger.info(
                "Step 3/4: Aggregating image analysis results."
            )

            logger.info(
                "Image analysis summary | "
                "listing_url=%s | total_images=%d | "
                "successful_analyses=%d | failed_analyses=%d | "
                "duplicate_matches=%d",
                current_listing_url,
                total_images,
                successful_analyses,
                failed_analyses,
                len(duplicate_matches)
            )

            # If at least one duplicate is detected,
            # classify the listing as potentially fraudulent.
            if duplicate_matches:

                score = 100.0

                duplicate_proofs = [
                    (
                        f"Photo reused in another listing: "
                        f"{matched_listing} "
                        f"(image: {image_url})"
                    )
                    for image_url, matched_listing in duplicate_matches
                ]

                proof = "; ".join(duplicate_proofs)

                logger.warning(
                    "Vision analysis completed with duplicate images detected | "
                    "listing_url=%s | duplicate_count=%d | "
                    "score=%.2f | duration=%.3fs",
                    current_listing_url,
                    len(duplicate_matches),
                    score,
                    time.perf_counter() - analysis_start
                )

                return score, proof

            # If every individual analysis failed, report that
            # the vision analysis could not produce a result.
            if successful_analyses == 0:

                logger.warning(
                    "Vision analysis could not process any downloaded images | "
                    "listing_url=%s | total_images=%d | "
                    "failed_analyses=%d | duration=%.3fs",
                    current_listing_url,
                    total_images,
                    failed_analyses,
                    time.perf_counter() - analysis_start
                )

                return (
                    0.0,
                    "Vision analysis skipped: All downloaded images failed processing."
                )

            # No duplicates found among successfully analyzed images.
            logger.info(
                "Step 4/4: Vision analysis completed successfully | "
                "listing_url=%s | score=0.0 | "
                "duplicate_found=False | successful_analyses=%d | "
                "failed_analyses=%d | total_duration=%.3fs",
                current_listing_url,
                successful_analyses,
                failed_analyses,
                time.perf_counter() - analysis_start
            )

            return (
                0.0,
                "No duplicate images detected across the listing's downloaded images."
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

    async def _analyze_single_image(
    self,
    img: Image.Image,
    image_url: str,
    current_listing_url: str,
    latitude: Optional[float],
    longitude: Optional[float],
    # Updated tuple to include lat and lon from existing records
    existing_records: List[Tuple[str, str, str, Optional[float], Optional[float]]],
    con
) -> Optional[str]:
        """
        Analyze a single downloaded image.

        Returns:
            Matched listing URL if a duplicate is detected.
            None if no duplicate is detected.
        """

        image_start = time.perf_counter()

        logger.info(
            "Starting single image analysis | image_url=%s",
            image_url
        )

        # Step 1: Generate perceptual hash.
        logger.info("Generating perceptual hash")

        hash_start = time.perf_counter()

        img_hash = imagehash.phash(img)
        img_hash_str = str(img_hash)

        logger.info(
            "Perceptual hash generated | "
            "phash=%s | hash_duration=%.3fs",
            img_hash_str,
            time.perf_counter() - hash_start
        )

        # Step 2: Retrieve existing image hashes.
        logger.info(
            "Checking existing image hashes in DuckDB"
        )

        db_start = time.perf_counter()

        # Step 3: Compare Hamming distances.
        logger.info(
            "Calculating Hamming distances | "
            "comparison_count=%d | threshold=%d",
            len(existing_records),
            self.similarity_threshold
        )

        comparison_start = time.perf_counter()

        closest_match = None
        minimum_distance = None
        duplicate_match = None

        # Unpacking now includes record_lat and record_lon
        for record_url, record_image_url, record_hash_str, record_lat, record_lon in existing_records:

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
                    
                    # --- NEW LOGIC: Geographical Proximity Check ---
                    if _is_location_nearby(latitude, longitude, record_lat, record_lon):
                        logger.info(
                            "Similar image ignored | properties are located nearby (same society/building) | "
                            "current_listing=%s | matched_listing=%s",
                            current_listing_url,
                            record_url
                        )
                        # Skip this record as a legitimate identical layout/society image
                        continue

                    duplicate_match = record_url

                    logger.warning(
                        "Potential duplicate image detected | "
                        "current_listing=%s | "
                        "current_image=%s | "
                        "matched_listing=%s | "
                        "matched_image=%s | "
                        "hamming_distance=%d | threshold=%d",
                        current_listing_url,
                        image_url,
                        record_url,
                        record_image_url,
                        hamming_distance,
                        self.similarity_threshold
                    )

                    # One matching record is sufficient for this image.
                    break

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

        # Step 4: Store hash only if no duplicate was detected.
        if duplicate_match is None:

            logger.info(
                "Storing image hash in DuckDB | "
                "listing_url=%s | image_url=%s",
                current_listing_url,
                image_url
            )

            insert_start = time.perf_counter()

            # Updated query to store latitude and longitude
            con.execute(
                """
                INSERT INTO image_hashes (
                    listing_url,
                    image_url,
                    phash,
                    latitude,
                    longitude
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    current_listing_url,
                    image_url,
                    img_hash_str,
                    latitude,
                    longitude
                ]
            )

            logger.info(
                "Image hash stored successfully | "
                "insert_duration=%.3fs",
                time.perf_counter() - insert_start
            )

        logger.info(
            "Single image analysis completed | "
            "image_url=%s | duplicate_found=%s | "
            "duration=%.3fs",
            image_url,
            duplicate_match is not None,
            time.perf_counter() - image_start
        )

        return duplicate_match
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
