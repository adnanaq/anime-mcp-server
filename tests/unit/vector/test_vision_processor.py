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

    def test_init_with_defaults(self):
        """Test VisionProcessor initialization with default parameters."""
        # Mock the imports that happen inside _init_clip_model
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            assert processor.model_name == "ViT-B/32"
            assert processor.cache_dir is None
            assert processor.device == "cpu"
            assert processor.model == mock_model
            assert processor.preprocess == mock_preprocess
            mock_clip.load.assert_called_once_with(
                "ViT-B/32", device="cpu", download_root=None
            )

    def test_init_with_custom_params(self):
        """Test VisionProcessor initialization with custom parameters."""
        # Mock the imports that happen inside _init_clip_model
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor(model_name="ViT-L/14", cache_dir="/tmp/clip")

            assert processor.model_name == "ViT-L/14"
            assert processor.cache_dir == "/tmp/clip"
            assert processor.device == "cuda"
            mock_clip.load.assert_called_once_with(
                "ViT-L/14", device="cuda", download_root="/tmp/clip"
            )

    def test_init_missing_dependencies(self):
        """Test VisionProcessor initialization when CLIP dependencies are missing."""

        # Mock import error for clip
        def mock_import(name, *args, **kwargs):
            if name == "clip":
                raise ImportError("No module named 'clip'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="CLIP dependencies missing"):
                VisionProcessor()

    def test_init_clip_model_failure(self):
        """Test VisionProcessor initialization when CLIP model loading fails."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_clip.load.side_effect = Exception("Model loading failed")

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            with pytest.raises(Exception, match="Model loading failed"):
                VisionProcessor()

    def test_decode_base64_image_success(self, sample_image_base64):
        """Test successful base64 image decoding."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_base64)

            assert image is not None
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            assert image.size == (100, 100)

    def test_decode_base64_image_data_url(self, sample_image_data_url):
        """Test decoding image from data URL format."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_data_url)

            assert image is not None
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"

    def test_decode_base64_image_invalid_data(self):
        """Test handling of invalid base64 data."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            image = processor._decode_base64_image("invalid_base64_data")

            assert image is None

    def test_decode_base64_image_rgba_conversion(self):
        """Test RGBA to RGB conversion."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        # Create RGBA image
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            image = processor._decode_base64_image(img_base64)

            assert image is not None
            assert image.mode == "RGB"

    def test_preprocess_image_success(self, sample_image_base64):
        """Test successful image preprocessing."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()

        # Mock preprocessing pipeline
        mock_tensor = Mock()
        mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
        mock_preprocess.return_value = mock_tensor
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_base64)
            processed = processor._preprocess_image(image)

            assert processed is not None
            mock_preprocess.assert_called_once_with(image)

    def test_preprocess_image_no_preprocessor(self):
        """Test preprocessing when preprocessor is None."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_clip.load.return_value = (mock_model, None)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            processor.preprocess = None

            img = Image.new("RGB", (100, 100), color="red")
            processed = processor._preprocess_image(img)

            assert processed is None

    def test_preprocess_image_failure(self, sample_image_base64):
        """Test preprocessing failure handling."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_preprocess.side_effect = Exception("Preprocessing failed")
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            image = processor._decode_base64_image(sample_image_base64)
            processed = processor._preprocess_image(image)

            assert processed is None

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)

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

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Mock processed image tensor
            mock_tensor = Mock()
            embedding = processor._generate_embedding(mock_tensor)

            assert embedding is not None
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)
            mock_model.encode_image.assert_called_once_with(mock_tensor)

    def test_generate_embedding_failure(self):
        """Test embedding generation failure."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)

        mock_model = Mock()
        mock_preprocess = Mock()
        mock_model.encode_image.side_effect = Exception("Encoding failed")
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            mock_tensor = Mock()
            embedding = processor._generate_embedding(mock_tensor)

            assert embedding is None

    def test_encode_image_no_model(self):
        """Test encoding when model is not initialized."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            processor.model = None  # Simulate uninitialized model

            embedding = processor.encode_image("test_data")
            assert embedding is None

    def test_encode_image_full_pipeline(self, sample_image_base64):
        """Test the full image encoding pipeline."""
        mock_clip = Mock()
        mock_torch = Mock()
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

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            embedding = processor.encode_image(sample_image_base64)

            assert embedding is not None
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_image_failure_handling(self):
        """Test handling of failures in image encoding pipeline."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Test with invalid base64 data
            embedding = processor.encode_image("invalid_data")
            assert embedding is None

    def test_encode_image_general_exception(self, sample_image_base64):
        """Test general exception handling in encode_image."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Mock _decode_base64_image to raise an exception
            processor._decode_base64_image = Mock(
                side_effect=Exception("Unexpected error")
            )

            embedding = processor.encode_image(sample_image_base64)
            assert embedding is None

    def test_encode_image_preprocess_failure(self, sample_image_base64):
        """Test encode_image when preprocessing returns None."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Mock _preprocess_image to return None
            processor._preprocess_image = Mock(return_value=None)

            embedding = processor.encode_image(sample_image_base64)
            assert embedding is None

    def test_encode_images_batch(self, sample_image_base64):
        """Test batch image encoding."""
        mock_clip = Mock()
        mock_torch = Mock()
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

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Test batch processing
            image_list = [sample_image_base64, sample_image_base64, sample_image_base64]
            embeddings = processor.encode_images_batch(image_list, batch_size=2)

            assert len(embeddings) == 3
            assert all(emb is not None for emb in embeddings)
            assert all(len(emb) == 512 for emb in embeddings)

    def test_encode_images_batch_exception(self):
        """Test batch encoding exception handling."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Mock encode_image to raise an exception
            processor.encode_image = Mock(side_effect=Exception("Batch error"))

            image_list = ["image1", "image2"]
            embeddings = processor.encode_images_batch(image_list)

            # Should return list of None values
            assert len(embeddings) == 2
            assert all(emb is None for emb in embeddings)

    def test_validate_image_data(self, sample_image_base64):
        """Test image data validation."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()

            # Test valid image data
            assert processor.validate_image_data(sample_image_base64) is True

            # Test invalid image data
            assert processor.validate_image_data("invalid_data") is False
            assert processor.validate_image_data("") is False

    def test_validate_image_data_exception(self):
        """Test validation when exception occurs."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            processor._decode_base64_image = Mock(
                side_effect=Exception("Validation error")
            )

            assert processor.validate_image_data("test_data") is False

    def test_get_supported_formats(self):
        """Test getting supported image formats."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
            processor = VisionProcessor()
            formats = processor.get_supported_formats()

            assert isinstance(formats, list)
            assert "jpg" in formats
            assert "png" in formats
            assert "jpeg" in formats

    def test_get_model_info(self):
        """Test getting model information."""
        mock_clip = Mock()
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip.load.return_value = (mock_model, mock_preprocess)

        with patch.dict("sys.modules", {"clip": mock_clip, "torch": mock_torch}):
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

    def test_mock_init_custom(self):
        """Test MockVisionProcessor initialization with custom params."""
        processor = MockVisionProcessor(model_name="custom_mock", cache_dir="/tmp")
        assert processor.model_name == "custom_mock"
        assert processor.cache_dir == "/tmp"

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

    def test_mock_encode_images_batch_with_batch_size(self):
        """Test MockVisionProcessor batch encoding with custom batch size."""
        processor = MockVisionProcessor()

        image_list = ["image1", "image2", "image3", "image4", "image5"]
        embeddings = processor.encode_images_batch(image_list, batch_size=2)

        assert len(embeddings) == 5
        assert all(emb is not None for emb in embeddings)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
