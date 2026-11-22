"""
Vision score aggregator.

Combines per-image match results into a single listing-level
risk score with confidence and human-readable proof.
"""

from dataclasses import dataclass
from typing import List, Tuple

from src.core.logger import get_logger
from src.vision.match_policy import MatchResult

logger = get_logger(__name__)


@dataclass
class AggregatedVisionResult:
    """Final output of the vision anomaly detector."""

    score: float        # 0.0 (clean) to 1.0 (all images duplicated)
    confidence: float   # 0.0 (no images analyzed) to 1.0 (all analyzed)
    proof: str          # human-readable evidence string


class VisionAggregator:
    """
    Aggregates per-image MatchResults into a listing-level score.

    Rules:
    - Only unsuppressed matches count toward the score.
    - Score = weighted fraction of images with matches.
    - Single match caps at 0.5 (require >=2 for top severity).
    - Confidence = analyzed / total. Below 0.5 = low confidence.
    """

    def __init__(self, single_match_cap: float = 0.5):
        self.single_match_cap = single_match_cap
        logger.info(
            "VisionAggregator initialized | single_match_cap=%.2f",
            single_match_cap,
        )

    def aggregate(
        self,
        match_results: List[Tuple[str, MatchResult]],
        total_images: int,
        analyzed_images: int,
    ) -> AggregatedVisionResult:
        """
        Args:
            match_results: list of (image_url, MatchResult) for images
                           that had a confirmed hash match.
            total_images: number of images in the listing.
            analyzed_images: number successfully hashed and compared.
        """
        # Filter to unsuppressed matches only.
        active = [
            (url, mr) for url, mr in match_results if not mr.suppressed
        ]

        if not active:
            confidence = (
                analyzed_images / total_images if total_images > 0 else 0.0
            )
            logger.info(
                "No active matches after policy filter | "
                "suppressed=%d | confidence=%.2f",
                len(match_results) - len(active), confidence,
            )
            return AggregatedVisionResult(
                score=0.0,
                confidence=confidence,
                proof="No duplicate images detected after context filtering.",
            )

        # Weighted score: sum of weights / analyzed count.
        total_weight = sum(mr.weight for _, mr in active)
        raw_score = total_weight / analyzed_images if analyzed_images > 0 else 0.0

        # Cap at single_match_cap if only 1 unique matched listing.
        matched_listings = {mr.reason for _, mr in active}
        if len(active) == 1 and raw_score > self.single_match_cap:
            raw_score = self.single_match_cap

        score = min(raw_score, 1.0)
        confidence = analyzed_images / total_images if total_images > 0 else 0.0

        # Build proof string.
        proof_parts = []
        for img_url, mr in active:
            proof_parts.append(
                f"[{mr.relationship.value}] weight={mr.weight:.2f} | "
                f"image={img_url} | {mr.reason}"
            )
        proof = "; ".join(proof_parts)

        logger.info(
            "Aggregation complete | active_matches=%d | suppressed=%d | "
            "score=%.3f | confidence=%.2f",
            len(active), len(match_results) - len(active),
            score, confidence,
        )

        return AggregatedVisionResult(
            score=score, confidence=confidence, proof=proof,
        )
