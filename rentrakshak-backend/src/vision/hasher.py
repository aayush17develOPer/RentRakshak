"""
Image hashing abstraction.

Interface `ImageHasher` with pluggable implementations.
Currently: PHasher (perceptual hash via imagehash library).
Later: EmbeddingHasher (CLIP/DINOv2 vectors) — same interface, zero change elsewhere.
"""

from abc import ABC, abstractmethod

import imagehash
from PIL import Image

from src.core.logger import get_logger

logger = get_logger(__name__)


class ImageHasher(ABC):
    """Computes a string hash representation of an image."""

    @abstractmethod
    def compute(self, img: Image.Image) -> str:
        """Return a hex string hash for the given PIL image."""
        ...


class PHasher(ImageHasher):
    """Perceptual hash using the DCT-based pHash algorithm."""

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        logger.info(
            "PHasher initialized | hash_size=%d | output_bits=%d",
            hash_size, hash_size * hash_size,
        )

    def compute(self, img: Image.Image) -> str:
        h = imagehash.phash(img, hash_size=self.hash_size)
        hash_str = str(h)
        logger.debug("pHash computed | hash=%s", hash_str)
        return hash_str
