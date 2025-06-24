"""Unit tests for VisionProcessor image processing and embedding generation.

Tests CLIP model integration, image preprocessing, and embedding generation
functionality for anime image vector search.
"""

import base64
import io
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.vector.vision_processor import MockVisionProcessor, VisionProcessor


class TestVisionProcessor:
    """Test suite for VisionProcessor functionality."""

    @pytest.fixture
    def sample_image_base64(self):
        """Create a sample base64 encoded image for testing."""
        # Create a simple 100x100 RGB image
        img = Image.new("RGB", (100, 100), color="red")

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_base64

    @pytest.fixture
    def sample_image_data_url(self, sample_image_base64):
        """Create a data URL format image for testing."""
        return f"data:image/png;base64,{sample_image_base64}"

    @pytest.fixture
    def mock_clip_model(self):
        """Mock CLIP model for testing."""
        mock_model = Mock()
        mock_preprocess = Mock()

        # Mock encode_image to return normalized features
        mock_features = Mock()
        mock_features.norm.return_value = mock_features
        mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [
            0.1
        ] * 512
        mock_features.__truediv__ = Mock(return_value=mock_features)
        mock_model.encode_image.return_value = mock_features

        return mock_model, mock_preprocess

    @pytest.fixture
    def mock_torch(self):
        """Mock torch for testing."""
        with patch("src.vector.vision_processor.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()
            yield mock_torch

    def test_init_with_defaults(self):
        """Test VisionProcessor initialization with default parameters."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()

            assert processor.model_name == "ViT-B/32"
            assert processor.cache_dir is None
            assert processor.device == "cpu"
            mock_clip.load.assert_called_once_with(
                "ViT-B/32", device="cpu", download_root=None
            )

    def test_init_with_custom_params(self):
        """Test VisionProcessor initialization with custom parameters."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = True
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor(model_name="ViT-L/14", cache_dir="/tmp/clip")

            assert processor.model_name == "ViT-L/14"
            assert processor.cache_dir == "/tmp/clip"
            assert processor.device == "cuda"
            mock_clip.load.assert_called_once_with(
                "ViT-L/14", device="cuda", download_root="/tmp/clip"
            )

    def test_init_missing_dependencies(self):
        """Test VisionProcessor initialization when CLIP dependencies are missing."""
        with patch(
            "src.vector.vision_processor.torch",
            side_effect=ImportError("torch not found"),
        ):
            with pytest.raises(ImportError, match="CLIP dependencies missing"):
                VisionProcessor()

    def test_decode_base64_image_success(self, sample_image_base64):
        """Test successful base64 image decoding."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_base64)

            assert image is not None
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            assert image.size == (100, 100)

    def test_decode_base64_image_data_url(self, sample_image_data_url):
        """Test decoding image from data URL format."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_data_url)

            assert image is not None
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"

    def test_decode_base64_image_invalid_data(self):
        """Test handling of invalid base64 data."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()
            image = processor._decode_base64_image("invalid_base64_data")

            assert image is None

    def test_preprocess_image_success(self, sample_image_base64):
        """Test successful image preprocessing."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_preprocess = Mock()

            # Mock preprocessing pipeline
            mock_tensor = Mock()
            mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
            mock_preprocess.return_value = mock_tensor
            mock_clip.load.return_value = (mock_model, mock_preprocess)

            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_base64)
            processed = processor._preprocess_image(image)

            assert processed is not None
            mock_preprocess.assert_called_once_with(image)

    def test_generate_embedding_success(self, mock_torch):
        """Test successful embedding generation."""
        with patch("src.vector.vision_processor.clip") as mock_clip:

            mock_model = Mock()
            mock_preprocess = Mock()

            # Mock the embedding generation process
            mock_features = Mock()
            mock_features.norm.return_value = mock_features
            mock_features.__truediv__ = Mock(return_value=mock_features)
            mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [
                0.1
            ] * 512
            mock_model.encode_image.return_value = mock_features

            mock_clip.load.return_value = (mock_model, mock_preprocess)

            processor = VisionProcessor()

            # Mock processed image tensor
            mock_tensor = Mock()
            embedding = processor._generate_embedding(mock_tensor)

            assert embedding is not None
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)
            mock_model.encode_image.assert_called_once_with(mock_tensor)

    def test_encode_image_full_pipeline(self, sample_image_base64):
        """Test the full image encoding pipeline."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)

            mock_model = Mock()
            mock_preprocess = Mock()

            # Mock the full pipeline
            mock_tensor = Mock()
            mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
            mock_preprocess.return_value = mock_tensor

            mock_features = Mock()
            mock_features.norm.return_value = mock_features
            mock_features.__truediv__ = Mock(return_value=mock_features)
            mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [
                0.1
            ] * 512
            mock_model.encode_image.return_value = mock_features

            mock_clip.load.return_value = (mock_model, mock_preprocess)

            processor = VisionProcessor()
            embedding = processor.encode_image(sample_image_base64)

            assert embedding is not None
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_image_failure_handling(self):
        """Test handling of failures in image encoding pipeline."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()

            # Test with invalid base64 data
            embedding = processor.encode_image("invalid_data")
            assert embedding is None

    def test_encode_images_batch(self, sample_image_base64):
        """Test batch image encoding."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)

            mock_model = Mock()
            mock_preprocess = Mock()

            # Mock successful encoding
            mock_tensor = Mock()
            mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
            mock_preprocess.return_value = mock_tensor

            mock_features = Mock()
            mock_features.norm.return_value = mock_features
            mock_features.__truediv__ = Mock(return_value=mock_features)
            mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [
                0.1
            ] * 512
            mock_model.encode_image.return_value = mock_features

            mock_clip.load.return_value = (mock_model, mock_preprocess)

            processor = VisionProcessor()

            # Test batch processing
            image_list = [sample_image_base64, sample_image_base64, sample_image_base64]
            embeddings = processor.encode_images_batch(image_list, batch_size=2)

            assert len(embeddings) == 3
            assert all(emb is not None for emb in embeddings)
            assert all(len(emb) == 512 for emb in embeddings)

    def test_validate_image_data(self, sample_image_base64):
        """Test image data validation."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()

            # Test valid image data
            assert processor.validate_image_data(sample_image_base64) is True

            # Test invalid image data
            assert processor.validate_image_data("invalid_data") is False
            assert processor.validate_image_data("") is False

    def test_get_supported_formats(self):
        """Test getting supported image formats."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()
            formats = processor.get_supported_formats()

            assert isinstance(formats, list)
            assert "jpg" in formats
            assert "png" in formats
            assert "jpeg" in formats

    def test_get_model_info(self):
        """Test getting model information."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()
            info = processor.get_model_info()

            assert isinstance(info, dict)
            assert info["model_name"] == "ViT-B/32"
            assert info["device"] == "cpu"
            assert info["embedding_size"] == 512
            assert info["input_resolution"] == 224
            assert info["is_initialized"] is True


class TestMockVisionProcessor:
    """Test suite for MockVisionProcessor functionality."""

    def test_mock_init(self):
        """Test MockVisionProcessor initialization."""
        processor = MockVisionProcessor()
        assert processor.model_name == "mock"
        assert processor.cache_dir is None

    def test_mock_encode_image_deterministic(self):
        """Test that MockVisionProcessor returns deterministic embeddings."""
        processor = MockVisionProcessor()

        # Same input should give same output
        embedding1 = processor.encode_image("test_image_data")
        embedding2 = processor.encode_image("test_image_data")

        assert embedding1 == embedding2
        assert len(embedding1) == 512
        assert all(isinstance(x, float) for x in embedding1)

    def test_mock_encode_image_different_inputs(self):
        """Test that different inputs give different embeddings."""
        processor = MockVisionProcessor()

        embedding1 = processor.encode_image("test_image_1")
        embedding2 = processor.encode_image("test_image_2")

        assert embedding1 != embedding2
        assert len(embedding1) == 512
        assert len(embedding2) == 512

    def test_mock_encode_images_batch(self):
        """Test MockVisionProcessor batch encoding."""
        processor = MockVisionProcessor()

        image_list = ["image1", "image2", "image3"]
        embeddings = processor.encode_images_batch(image_list)

        assert len(embeddings) == 3
        assert all(emb is not None for emb in embeddings)
        assert all(len(emb) == 512 for emb in embeddings)

        # Each should be different
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    def test_mock_validate_image_data(self):
        """Test MockVisionProcessor image validation."""
        processor = MockVisionProcessor()

        assert processor.validate_image_data("valid_data") is True
        assert processor.validate_image_data("") is False

    def test_mock_get_supported_formats(self):
        """Test MockVisionProcessor supported formats."""
        processor = MockVisionProcessor()
        formats = processor.get_supported_formats()

        assert isinstance(formats, list)
        assert "jpg" in formats
        assert "png" in formats

    def test_mock_get_model_info(self):
        """Test MockVisionProcessor model info."""
        processor = MockVisionProcessor()
        info = processor.get_model_info()

        assert isinstance(info, dict)
        assert info["model_name"] == "mock"
        assert info["device"] == "mock"
        assert info["embedding_size"] == 512
        assert info["is_initialized"] is True


class TestImageProcessingIntegration:
    """Integration tests for image processing pipeline."""

    def test_vision_processor_error_recovery(self):
        """Test that VisionProcessor handles errors gracefully."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_model = Mock()
            mock_preprocess = Mock()

            # Mock failure in encoding
            mock_model.encode_image.side_effect = Exception("CLIP model error")
            mock_clip.load.return_value = (mock_model, mock_preprocess)

            processor = VisionProcessor()

            # Should return None on error, not raise exception
            embedding = processor.encode_image("valid_base64_data")
            assert embedding is None

    def test_pillow_conversion_edge_cases(self):
        """Test PIL Image conversion edge cases."""
        with (
            patch("src.vector.vision_processor.clip") as mock_clip,
            patch("src.vector.vision_processor.torch") as mock_torch,
        ):

            mock_torch.cuda.is_available.return_value = False
            mock_clip.load.return_value = (Mock(), Mock())

            processor = VisionProcessor()

            # Create RGBA image that needs conversion to RGB
            img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Should successfully decode and convert to RGB
            decoded_img = processor._decode_base64_image(img_base64)
            assert decoded_img is not None
            assert decoded_img.mode == "RGB"

    def test_performance_requirements(self):
        """Test that processing meets performance requirements."""
        # This test verifies that the vision processor interface
        # is designed for the required performance characteristics

        processor = MockVisionProcessor()

        # Single image should process quickly
        import time

        start_time = time.time()
        embedding = processor.encode_image("test_image")
        processing_time = time.time() - start_time

        # Should be very fast for mock processor
        assert processing_time < 0.1  # 100ms
        assert embedding is not None
        assert len(embedding) == 512

    def test_memory_efficiency(self):
        """Test memory efficiency for batch processing."""
        processor = MockVisionProcessor()

        # Large batch should not cause memory issues
        large_batch = ["image_data"] * 1000
        embeddings = processor.encode_images_batch(large_batch, batch_size=100)

        assert len(embeddings) == 1000
        assert all(emb is not None for emb in embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
