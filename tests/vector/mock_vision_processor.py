"""Mock vision processor for testing without CLIP dependencies."""

import hashlib
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MockVisionProcessor:
    """Mock vision processor for testing without CLIP dependencies."""

    def __init__(self, model_name: str = "mock", cache_dir: Optional[str] = None):
        """Initialize mock processor."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        logger.info("Initialized mock vision processor (for testing)")

    def encode_image(self, image_data: str) -> Optional[List[float]]:
        """Return mock embedding vector."""
        # Return deterministic mock embedding for testing
        hash_obj = hashlib.md5(image_data[:100].encode())
        seed = int(hash_obj.hexdigest()[:8], 16)

        # Generate consistent mock embedding
        np.random.seed(seed % (2**32))
        mock_embedding = np.random.normal(0, 1, 512).tolist()

        return mock_embedding

    def encode_images_batch(
        self, image_data_list: List[str], batch_size: int = 32
    ) -> List[Optional[List[float]]]:
        """Return mock embeddings for batch."""
        return [self.encode_image(img) for img in image_data_list]

    def validate_image_data(self, image_data: str) -> bool:
        """Always return True for mock validation."""
        return len(image_data) > 0

    def get_supported_formats(self) -> List[str]:
        """Return mock supported formats."""
        return ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

    def get_model_info(self) -> dict:
        """Return mock model info."""
        return {
            "model_name": self.model_name,
            "device": "mock",
            "embedding_size": 512,
            "input_resolution": 224,
            "is_initialized": True,
        }