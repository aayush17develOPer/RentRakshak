"""
Match context policy for vision duplicate detection.

Decides the relationship between two listings whose images matched
and assigns a weight (0.0 = ignore, 1.0 = strongest signal).
Replaces the old bare Haversine-only check.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


class Relationship(str, Enum):
    SAME_LISTING = "same_listing"
    SAME_OWNER_SAME_UNIT = "same_owner_same_unit"
    SAME_SOCIETY = "same_society"
    NEARBY_SAME_CITY = "nearby_same_city"
    SAME_CITY_DIFFERENT_LOCALITY = "same_city_different_locality"
    DIFFERENT_CITY = "different_city"
    UNKNOWN = "unknown"


@dataclass
class MatchResult:
    """Outcome of evaluating one image-pair match in context."""

    relationship: Relationship
    weight: float               # 0.0 (ignore) to 1.0 (strongest)
    suppressed: bool            # True = do not count toward score
    reason: str                 # human-readable audit string


class MatchContextPolicy:
    """
    Evaluates the relationship between two listings and decides
    how much weight an image match between them should carry.

    Rules applied in order (first match wins):
    1. Same listing URL          → suppress (weight 0).
    2. Same owner + same society → suppress (probable relist).
    3. Same society              → suppress (shared amenity/builder photos).
    4. Nearby (<threshold) same city → low weight.
    5. Same city, different area → medium-high weight.
    6. Different city            → full weight.
    """

    def __init__(
        self,
        nearby_threshold_meters: float = 500.0,
        weight_nearby: float = 0.3,
        weight_same_city: float = 0.7,
        weight_different_city: float = 1.0,
    ):
        self.nearby_threshold = nearby_threshold_meters
        self.weight_nearby = weight_nearby
        self.weight_same_city = weight_same_city
        self.weight_different_city = weight_different_city

        logger.info(
            "MatchContextPolicy initialized | nearby_thresh=%.0fm | "
            "weights=(nearby=%.2f, same_city=%.2f, diff_city=%.2f)",
            nearby_threshold_meters, weight_nearby,
            weight_same_city, weight_different_city,
        )

    def evaluate(
        self,
        # Current listing
        current_url: str,
        current_owner: str,
        current_society: str,
        current_city: str,
        current_lat: Optional[float],
        current_lon: Optional[float],
        # Candidate (matched) listing
        candidate_url: str,
        candidate_owner: str,
        candidate_society: str,
        candidate_city: str,
        candidate_lat: Optional[float],
        candidate_lon: Optional[float],
    ) -> MatchResult:
        """Evaluate the relationship and return a MatchResult."""

        # Rule 1: same listing
        if current_url == candidate_url:
            return self._suppress(
                Relationship.SAME_LISTING, "Same listing URL"
            )

        # Rule 2: same owner + same society (relist)
        if (
            current_owner
            and candidate_owner
            and current_owner == candidate_owner
            and current_society
            and candidate_society
            and current_society == candidate_society
        ):
            return self._suppress(
                Relationship.SAME_OWNER_SAME_UNIT,
                f"Same owner ({current_owner}) in same society",
            )

        # Rule 3: same society (normalized names match)
        if (
            current_society
            and candidate_society
            and current_society == candidate_society
        ):
            return self._suppress(
                Relationship.SAME_SOCIETY,
                f"Same society ({current_society})",
            )

        # Rule 4: nearby in same city
        current_city_norm = current_city.strip().lower()
        candidate_city_norm = candidate_city.strip().lower()
        same_city = (
            current_city_norm
            and candidate_city_norm
            and current_city_norm == candidate_city_norm
        )

        if same_city:
            dist = self._haversine(
                current_lat, current_lon, candidate_lat, candidate_lon
            )
            if dist is not None and dist <= self.nearby_threshold:
                return MatchResult(
                    relationship=Relationship.NEARBY_SAME_CITY,
                    weight=self.weight_nearby,
                    suppressed=False,
                    reason=(
                        f"Nearby ({dist:.0f}m) in same city "
                        f"({current_city_norm})"
                    ),
                )

            # Rule 5: same city, different area
            return MatchResult(
                relationship=Relationship.SAME_CITY_DIFFERENT_LOCALITY,
                weight=self.weight_same_city,
                suppressed=False,
                reason=(
                    f"Same city ({current_city_norm}), "
                    f"distance={dist:.0f}m" if dist else
                    f"Same city ({current_city_norm}), distance unknown"
                ),
            )

        # Rule 6: different city
        if candidate_city_norm:
            return MatchResult(
                relationship=Relationship.DIFFERENT_CITY,
                weight=self.weight_different_city,
                suppressed=False,
                reason=(
                    f"Different cities "
                    f"({current_city_norm} vs {candidate_city_norm})"
                ),
            )

        # Fallback: insufficient location data
        return MatchResult(
            relationship=Relationship.UNKNOWN,
            weight=self.weight_same_city,
            suppressed=False,
            reason="Insufficient location data to determine relationship",
        )

    # ── Helpers ───────────────────────────────────────────────

    def _suppress(
        self, relationship: Relationship, reason: str
    ) -> MatchResult:
        logger.info("Match suppressed | relationship=%s | reason=%s",
                     relationship.value, reason)
        return MatchResult(
            relationship=relationship,
            weight=0.0,
            suppressed=True,
            reason=reason,
        )

    @staticmethod
    def _haversine(
        lat1: Optional[float],
        lon1: Optional[float],
        lat2: Optional[float],
        lon2: Optional[float],
    ) -> Optional[float]:
        """
        Haversine distance in meters. Returns None if any coord is missing.
        """
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            return None

        R = 6_371_000  # Earth radius in meters
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)

        a = (
            math.sin(dp / 2) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
