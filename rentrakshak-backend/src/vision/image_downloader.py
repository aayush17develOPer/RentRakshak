"""
Image download and quality filtering.

Downloads listing images via async HTTP, filters out junk
(too small, too few bytes, undecodable), and returns a list
of DownloadedImage objects ready for hashing.
"""

import io
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from PIL import Image, UnidentifiedImageError

from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DownloadedImage:
    """A successfully downloaded and validated image."""

    img: Image.Image
    url: str
    width: int = 0
    height: int = 0
    position: int = 0
    is_cover: bool = False


class ImageDownloader:
    """
    Downloads images from URLs and applies quality filters.

    Filters:
    - Minimum dimensions (default 100x100).
    - Minimum byte size (default 5 KB) to skip placeholders/logos.
    - Must decode as a valid image.
    """

    MIN_DIMENSION: int = 100
    MIN_BYTES: int = 5_000

    def __init__(
        self,
        timeout_seconds: float = 50.0,
        min_dimension: int = 100,
        min_bytes: int = 5_000,
    ):
        self.timeout_seconds = timeout_seconds
        self.MIN_DIMENSION = min_dimension
        self.MIN_BYTES = min_bytes
        logger.info(
            "ImageDownloader initialized | timeout=%.1fs | "
            "min_dim=%d | min_bytes=%d",
            timeout_seconds, min_dimension, min_bytes,
        )

    async def download_all(
        self,
        urls: List[str],
        positions: Optional[List[int]] = None,
        cover_index: Optional[int] = None,
    ) -> List[DownloadedImage]:
        """
        Download and filter a batch of image URLs.

        Args:
            urls: image URLs to download.
            positions: per-URL gallery position (index in listing).
            cover_index: index in `urls` that is the cover photo.

        Returns:
            List of DownloadedImage for images that passed quality checks.
        """
        batch_start = time.perf_counter()
        logger.info("Starting image download batch | total_urls=%d", len(urls))

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        }

        results: List[DownloadedImage] = []

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(self.timeout_seconds),
        ) as client:
            for idx, url in enumerate(urls):
                downloaded = await self._download_one(
                    client, url, headers, idx, len(urls)
                )
                if downloaded is None:
                    continue

                # Apply quality filter.
                img = downloaded
                w, h = img.size

                if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
                    logger.info(
                        "Image dropped (too small) | url=%s | size=%dx%d",
                        url, w, h,
                    )
                    continue

                pos = positions[idx] if positions and idx < len(positions) else idx
                is_cover = (cover_index is not None and idx == cover_index)

                results.append(
                    DownloadedImage(
                        img=img,
                        url=url,
                        width=w,
                        height=h,
                        position=pos,
                        is_cover=is_cover,
                    )
                )

        logger.info(
            "Download batch complete | downloaded=%d | dropped=%d | "
            "duration=%.3fs",
            len(results),
            len(urls) - len(results),
            time.perf_counter() - batch_start,
        )
        return results

    async def _download_one(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        index: int,
        total: int,
    ) -> Optional[Image.Image]:
        """Download and decode a single image. Returns None on failure."""
        logger.debug(
            "Downloading image %d/%d | url=%s", index + 1, total, url
        )
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                logger.info(
                    "Skipped (not an image) | url=%s | content_type=%s",
                    url, content_type,
                )
                return None

            raw_bytes = resp.content
            if len(raw_bytes) < self.MIN_BYTES:
                logger.info(
                    "Skipped (too few bytes) | url=%s | bytes=%d",
                    url, len(raw_bytes),
                )
                return None

            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
            logger.debug(
                "Image decoded | url=%s | format=%s | size=%s",
                url, img.format, img.size,
            )
            return img

        except httpx.HTTPStatusError as e:
            logger.warning("HTTP error | url=%s | status=%d", url, e.response.status_code)
        except httpx.TimeoutException:
            logger.warning("Timeout | url=%s", url)
        except UnidentifiedImageError:
            logger.warning("Undecodable image | url=%s", url)
        except Exception:
            logger.exception("Unexpected download error | url=%s", url)

        return None
