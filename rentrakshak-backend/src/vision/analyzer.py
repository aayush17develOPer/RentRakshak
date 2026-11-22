"""
Vision anomaly detector — orchestrator.

Coordinates the pipeline: download → hash → search → policy → aggregate.
All business logic lives in the injected dependencies.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import imagehash

from src.core.interfaces import IRiskAnalyzer
from src.core.logger import get_logger
from src.vision.aggregator import VisionAggregator
from src.vision.hash_store import ImageHashRecord, ImageHashStore
from src.vision.hasher import ImageHasher, PHasher
from src.vision.image_downloader import ImageDownloader
from src.vision.match_policy import MatchContextPolicy, MatchResult

logger = get_logger(__name__)


class VisionAnomalyAnalyzer(IRiskAnalyzer):
    """
    Implements the vision-based duplicate image detection.

    Detects reuse of identical or near-identical property images
    across different listings using perceptual hashing and
    context-aware match filtering.
    """

    def __init__(
        self,
        db_path: str = "data/rentrakshak.duckdb",
        similarity_threshold: int = 8,
        downloader: Optional[ImageDownloader] = None,
        hasher: Optional[ImageHasher] = None,
        store: Optional[ImageHashStore] = None,
        policy: Optional[MatchContextPolicy] = None,
        aggregator: Optional[VisionAggregator] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.downloader = downloader or ImageDownloader()
        self.hasher = hasher or PHasher()
        self.store = store or ImageHashStore(db_path=db_path)
        self.policy = policy or MatchContextPolicy()
        self.aggregator = aggregator or VisionAggregator()

        logger.info(
            "VisionAnomalyAnalyzer initialized | "
            "similarity_threshold=%d | db_path=%s",
            similarity_threshold, db_path,
        )

    async def analyze(
        self, listing_data: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Full vision analysis pipeline.

        Returns:
            (score, proof) where score is 0.0–1.0 and proof is
            a human-readable explanation.
        """
        start = time.perf_counter()
        listing_url = listing_data.get("url", "")

        logger.info("Vision analysis started | listing_url=%s", listing_url)

        # ── Step 1: Validate input ────────────────────────────
        image_urls = listing_data.get("image_urls", [])
        if not isinstance(image_urls, list) or not image_urls:
            logger.warning(
                "No images to analyze | listing_url=%s", listing_url
            )
            return 0.0, "No images found to analyze."

        valid_urls = [u for u in image_urls if isinstance(u, str) and u.strip()]
        if not valid_urls:
            return 0.0, "No valid image URLs provided."

        # Extract listing context for policy evaluation.
        current_city = listing_data.get("city", "")
        current_society = listing_data.get("society", "")
        current_owner = listing_data.get("owner_id", "")
        current_lat = listing_data.get("latitude")
        current_lon = listing_data.get("longitude")
        nobroker_id = listing_data.get("nobroker_id", "")

        # ── Step 2: Download images ───────────────────────────
        logger.info("Step 1/4: Downloading images.")
        downloaded = await self.downloader.download_all(valid_urls)
        if not downloaded:
            logger.warning(
                "No images downloaded | listing_url=%s", listing_url
            )
            return 0.0, "Vision analysis skipped: no images could be downloaded."

        # ── Step 3: Hash, search, and evaluate ────────────────
        logger.info("Step 2/4: Hashing and comparing images.")
        existing_records = self.store.get_all_except(listing_url)

        match_results: List[Tuple[str, MatchResult]] = []
        analyzed_count = 0

        for dl_img in downloaded:
            try:
                img_hash_str = self.hasher.compute(dl_img.img)
                analyzed_count += 1

                # Brute-force compare against all existing hashes.
                best_match = self._find_best_match(
                    img_hash_str, existing_records
                )

                if best_match is not None:
                    candidate = best_match
                    # Evaluate context policy.
                    mr = self.policy.evaluate(
                        current_url=listing_url,
                        current_owner=current_owner,
                        current_society=current_society,
                        current_city=current_city,
                        current_lat=current_lat,
                        current_lon=current_lon,
                        candidate_url=candidate.listing_url,
                        candidate_owner=candidate.owner_id,
                        candidate_society=candidate.society_normalized,
                        candidate_city=candidate.city,
                        candidate_lat=candidate.latitude,
                        candidate_lon=candidate.longitude,
                    )

                    match_results.append((dl_img.url, mr))

                    if not mr.suppressed:
                        logger.warning(
                            "Duplicate detected | image=%s | "
                            "matched_listing=%s | relationship=%s | "
                            "weight=%.2f",
                            dl_img.url, candidate.listing_url,
                            mr.relationship.value, mr.weight,
                        )

                # Store hash for future comparisons
                # (regardless of match, so the corpus grows).
                self.store.insert(
                    ImageHashRecord(
                        nobroker_id=nobroker_id,
                        listing_url=listing_url,
                        image_url=dl_img.url,
                        phash=img_hash_str,
                        latitude=current_lat,
                        longitude=current_lon,
                        city=current_city,
                        society_normalized=current_society,
                        owner_id=current_owner,
                        image_position=dl_img.position,
                        is_cover=dl_img.is_cover,
                        width=dl_img.width,
                        height=dl_img.height,
                        image_source="",
                        uploaded_by="",
                        nb_duplicate_flag=False,
                    )
                )

            except Exception:
                logger.exception(
                    "Failed to process image | url=%s", dl_img.url
                )

        # ── Step 4: Aggregate ─────────────────────────────────
        logger.info("Step 3/4: Aggregating results.")
        result = self.aggregator.aggregate(
            match_results=match_results,
            total_images=len(valid_urls),
            analyzed_images=analyzed_count,
        )

        logger.info(
            "Step 4/4: Vision analysis complete | listing_url=%s | "
            "score=%.3f | confidence=%.2f | duration=%.3fs",
            listing_url, result.score, result.confidence,
            time.perf_counter() - start,
        )

        # Scale to 0-100 to match IRiskAnalyzer contract.
        return result.score * 100, result.proof

    # ── Private helpers ───────────────────────────────────────

    def _find_best_match(
        self,
        img_hash_str: str,
        existing_records: list,
    ) -> Optional[ImageHashRecord]:
        """
        Brute-force scan for the closest hash below threshold.
        Returns the best-matching ImageHashRecord or None.
        """
        current_hash = imagehash.hex_to_hash(img_hash_str)
        best_record = None
        best_distance = None

        for record in existing_records:
            try:
                stored_hash = imagehash.hex_to_hash(record.phash)
                distance = current_hash - stored_hash

                if distance < self.similarity_threshold:
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_record = record
            except Exception:
                logger.debug(
                    "Skipped invalid hash | listing=%s | phash=%s",
                    record.listing_url, record.phash,
                )

        if best_record:
            logger.debug(
                "Best match found | distance=%d | matched_listing=%s",
                best_distance, best_record.listing_url,
            )

        return best_record


# ── Quick test ────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def test_vision():
        logger.info("Starting standalone VisionAnomalyAnalyzer test.")
        analyzer = VisionAnomalyAnalyzer()

        listing_1 = {
            "url": "https://example.com/listing-1",
            "image_urls": ["https://picsum.photos/id/237/200/300"],
            "city": "gurgaon",
            "society": "test society",
            "owner_id": "owner1",
        }
        listing_2 = {
            "url": "https://example.com/listing-2",
            "image_urls": ["https://picsum.photos/id/237/200/300"],
            "city": "mumbai",
            "society": "other society",
            "owner_id": "owner2",
        }

        score1, proof1 = await analyzer.analyze(listing_1)
        print(f"Listing 1 | Score: {score1} | Proof: {proof1}\n")

        score2, proof2 = await analyzer.analyze(listing_2)
        print(f"Listing 2 | Score: {score2} | Proof: {proof2}")

    asyncio.run(test_vision())
