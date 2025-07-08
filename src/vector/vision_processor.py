"""Vision processor for anime image embeddings using CLIP model.

Handles image preprocessing, embedding generation, and caching for efficient
visual similarity search in anime data.
"""

import base64
import io
import logging
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VisionProcessor:
    """CLIP-based vision processor for anime image embeddings."""

    def __init__(self, model_name: str = "ViT-B/32", cache_dir: Optional[str] = None):
        """Initialize vision processor with CLIP model.

        Args:
            model_name: CLIP model name (default: ViT-B/32)
            cache_dir: Optional cache directory for model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.preprocess = None
        self.device = None

        # Initialize CLIP model
        self._init_clip_model()

    def _init_clip_model(self):
        """Initialize CLIP model for image encoding."""
        try:
            import clip
            import torch

            # Set device (CPU for compatibility, GPU if available)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load CLIP model
            self.model, self.preprocess = clip.load(
                self.model_name, device=self.device, download_root=self.cache_dir
            )

            # Set to evaluation mode
            self.model.eval()

            logger.info(
                f"Initialized CLIP model ({self.model_name}) on device: {self.device}"
            )

        except ImportError as e:
            logger.error(
                "CLIP dependencies not installed. Install with: pip install torch torchvision clip-by-openai"
            )
            raise ImportError("CLIP dependencies missing") from e
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise

    def encode_image(self, image_data: str) -> Optional[List[float]]:
        """Encode image to embedding vector using CLIP.

        Args:
            image_data: Base64 encoded image data

        Returns:
            512-dimensional embedding vector or None if processing fails
        """
        try:
            if not self.model:
                logger.error("CLIP model not initialized")
                return None

            # Decode base64 image
            image = self._decode_base64_image(image_data)
            if image is None:
                return None

            # Preprocess image
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return None

            # Generate embedding
            embedding = self._generate_embedding(processed_image)
            return embedding

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None

    def _decode_base64_image(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image data to PIL Image.

        Args:
            image_data: Base64 encoded image string

        Returns:
            PIL Image object or None if decoding fails
        """
        try:
            # Handle data URL format (data:image/jpeg;base64,...)
            if image_data.startswith("data:"):
                # Extract base64 part after comma
                base64_part = image_data.split(",", 1)[1]
            else:
                base64_part = image_data

            # Decode base64
            image_bytes = base64.b64decode(base64_part)

            # Create PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def _preprocess_image(self, image: Image.Image) -> Optional["torch.Tensor"]:
        """Preprocess PIL image for CLIP model.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed image tensor or None if preprocessing fails
        """
        try:
            if not self.preprocess:
                logger.error("CLIP preprocessor not available")
                return None

            # Apply CLIP preprocessing
            processed = self.preprocess(image).unsqueeze(0).to(self.device)
            return processed

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None

    def _generate_embedding(
        self, processed_image: "torch.Tensor"
    ) -> Optional[List[float]]:
        """Generate embedding vector from preprocessed image.

        Args:
            processed_image: Preprocessed image tensor

        Returns:
            512-dimensional embedding vector or None if generation fails
        """
        try:
            import torch

            with torch.no_grad():
                # Generate image features
                image_features = self.model.encode_image(processed_image)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                # Convert to list
                embedding = image_features.cpu().numpy().flatten().tolist()

                return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def encode_images_batch(
        self, image_data_list: List[str], batch_size: int = 32
    ) -> List[Optional[List[float]]]:
        """Encode multiple images in batches for efficiency.

        Args:
            image_data_list: List of base64 encoded images
            batch_size: Number of images to process per batch

        Returns:
            List of embedding vectors (same order as input)
        """
        try:
            embeddings = []
            total_images = len(image_data_list)

            logger.info(f"Processing {total_images} images in batches of {batch_size}")

            for i in range(0, total_images, batch_size):
                batch = image_data_list[i : i + batch_size]
                batch_embeddings = []

                for image_data in batch:
                    embedding = self.encode_image(image_data)
                    batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

                # Log progress
                processed = min(i + batch_size, total_images)
                logger.info(f"Processed {processed}/{total_images} images")

            return embeddings

        except Exception as e:
            logger.error(f"Batch image encoding failed: {e}")
            return [None] * len(image_data_list)

    def validate_image_data(self, image_data: str) -> bool:
        """Validate if image data can be processed.

        Args:
            image_data: Base64 encoded image data

        Returns:
            True if image data is valid, False otherwise
        """
        try:
            image = self._decode_base64_image(image_data)
            return image is not None
        except Exception:
            return False

    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats.

        Returns:
            List of supported image format extensions
        """
        return ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_size": 512,  # CLIP ViT-B/32 embedding size
            "input_resolution": 224,  # CLIP standard input size
            "is_initialized": self.model is not None,
        }


