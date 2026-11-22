"""
Persistent image hash storage backed by DuckDB.

Owns the `image_hashes` table schema, migration of old columns,
and all read/write operations.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import duckdb

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageHashRecord:
    """One row in the image_hashes table."""

    nobroker_id: str
    listing_url: str
    image_url: str
    phash: str
    latitude: Optional[float]
    longitude: Optional[float]
    city: str
    society_normalized: str
    owner_id: str
    image_position: int
    is_cover: bool
    width: Optional[int]
    height: Optional[int]
    image_source: str
    uploaded_by: str
    nb_duplicate_flag: bool


class ImageHashStore:
    """
    DuckDB-backed store for image perceptual hashes.
    Handles schema creation, migration, inserts, and queries.
    """

    _COLUMNS = [
        ("nobroker_id", "VARCHAR"),
        ("listing_url", "VARCHAR"),
        ("image_url", "VARCHAR"),
        ("phash", "VARCHAR"),
        ("latitude", "FLOAT"),
        ("longitude", "FLOAT"),
        ("city", "VARCHAR"),
        ("society_normalized", "VARCHAR"),
        ("owner_id", "VARCHAR"),
        ("image_position", "INT"),
        ("is_cover", "BOOLEAN"),
        ("width", "INT"),
        ("height", "INT"),
        ("image_source", "VARCHAR"),
        ("uploaded_by", "VARCHAR"),
        ("nb_duplicate_flag", "BOOLEAN"),
        ("created_at", "TIMESTAMP DEFAULT current_timestamp"),
    ]

    def __init__(self, db_path: str = "data/rentrakshak.duckdb"):
        self.db_path = os.path.abspath(db_path)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        logger.info("ImageHashStore initializing | db_path=%s", self.db_path)
        self._init_table()

    # ── Schema management ─────────────────────────────────────

    def _init_table(self) -> None:
        """Create table if missing, then migrate (add new columns)."""
        start = time.perf_counter()

        with duckdb.connect(self.db_path) as con:
            # Create with full schema if table does not exist.
            col_defs = ", ".join(
                f"{name} {dtype}" for name, dtype in self._COLUMNS
            )
            con.execute(
                f"CREATE TABLE IF NOT EXISTS image_hashes ({col_defs})"
            )

            # Migrate: add any column that is missing.
            existing = {
                row[0]
                for row in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'image_hashes'"
                ).fetchall()
            }

            for col_name, col_type in self._COLUMNS:
                if col_name not in existing:
                    # Strip DEFAULT clause for ALTER ADD.
                    base_type = col_type.split(" DEFAULT")[0].strip()
                    con.execute(
                        f"ALTER TABLE image_hashes "
                        f"ADD COLUMN {col_name} {base_type}"
                    )
                    logger.info(
                        "Migrated: added column %s (%s)", col_name, base_type
                    )

            count = con.execute(
                "SELECT COUNT(*) FROM image_hashes"
            ).fetchone()[0]

        logger.info(
            "ImageHashStore ready | existing_records=%d | duration=%.3fs",
            count, time.perf_counter() - start,
        )

    # ── Queries ───────────────────────────────────────────────

    def get_all_except(
        self, listing_url: str
    ) -> List[ImageHashRecord]:
        """Return all records except those belonging to the given listing."""
        with duckdb.connect(self.db_path) as con:
            rows = con.execute(
                """
                SELECT nobroker_id, listing_url, image_url, phash,
                       latitude, longitude, city, society_normalized,
                       owner_id, image_position, is_cover,
                       width, height, image_source, uploaded_by,
                       nb_duplicate_flag
                FROM image_hashes
                WHERE listing_url IS DISTINCT FROM ?
                """,
                [listing_url],
            ).fetchall()

        records = [
            ImageHashRecord(
                nobroker_id=r[0] or "",
                listing_url=r[1] or "",
                image_url=r[2] or "",
                phash=r[3] or "",
                latitude=r[4],
                longitude=r[5],
                city=r[6] or "",
                society_normalized=r[7] or "",
                owner_id=r[8] or "",
                image_position=r[9] or 0,
                is_cover=bool(r[10]) if r[10] is not None else False,
                width=r[11],
                height=r[12],
                image_source=r[13] or "",
                uploaded_by=r[14] or "",
                nb_duplicate_flag=bool(r[15]) if r[15] is not None else False,
            )
            for r in rows
        ]

        logger.debug(
            "Fetched %d records (excluding listing_url=%s)",
            len(records), listing_url,
        )
        return records

    # ── Insert ────────────────────────────────────────────────

    def insert(self, record: ImageHashRecord) -> None:
        """Insert a single image hash record."""
        with duckdb.connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO image_hashes (
                    nobroker_id, listing_url, image_url, phash,
                    latitude, longitude, city, society_normalized,
                    owner_id, image_position, is_cover,
                    width, height, image_source, uploaded_by,
                    nb_duplicate_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.nobroker_id, record.listing_url,
                    record.image_url, record.phash,
                    record.latitude, record.longitude,
                    record.city, record.society_normalized,
                    record.owner_id, record.image_position,
                    record.is_cover, record.width, record.height,
                    record.image_source, record.uploaded_by,
                    record.nb_duplicate_flag,
                ],
            )
        logger.debug(
            "Inserted hash | listing=%s | image=%s | phash=%s",
            record.listing_url, record.image_url, record.phash,
        )

    def record_count(self) -> int:
        """Return total number of stored hashes."""
        with duckdb.connect(self.db_path) as con:
            return con.execute(
                "SELECT COUNT(*) FROM image_hashes"
            ).fetchone()[0]
