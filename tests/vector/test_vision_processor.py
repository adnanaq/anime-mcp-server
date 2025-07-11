"""Tests for vision processor supporting multiple embedding models."""

import pytest
from unittest.mock import MagicMock, patch, Mock
import numpy as np
from PIL import Image
import base64
import io

from src.vector.vision_processor import VisionProcessor


class TestVisionProcessor:
    """Test cases for vision processor."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.image_embedding_provider = "clip"
        settings.image_embedding_model = "ViT-B/32"
        settings.image_embedding_model_fallback = "ViT-L/14"
        settings.model_cache_dir = None
        settings.enable_model_fallback = True
        settings.model_warm_up = False
        settings.siglip_input_resolution = 384
        settings.jinaclip_input_resolution = 512
        return settings

    @pytest.fixture
    def mock_clip_model(self):
        """Mock CLIP model."""
        model = MagicMock()
        model.encode_image.return_value = MagicMock()
        model.encode_image.return_value.norm.return_value = MagicMock()
        model.encode_image.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 512
        return model

    @pytest.fixture
    def mock_siglip_model(self):
        """Mock SigLIP model."""
        model = MagicMock()
        model.config.projection_dim = 768
        model.get_image_features.return_value = MagicMock()
        model.get_image_features.return_value.norm.return_value = MagicMock()
        model.get_image_features.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 768
        return model

    @pytest.fixture
    def mock_jinaclip_model(self):
        """Mock JinaCLIP model."""
        model = MagicMock()
        model.get_image_features.return_value = MagicMock()
        model.get_image_features.return_value.norm.return_value = MagicMock()
        model.get_image_features.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 768
        return model

    @pytest.fixture
    def sample_image_b64(self):
        """Sample base64 encoded image."""
        # Create a simple 10x10 red image
        img = Image.new('RGB', (10, 10), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def test_init_with_clip_provider(self, mock_settings):
        """Test initialization with CLIP provider."""
        mock_settings.image_embedding_provider = "clip"
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            
            assert processor.provider == "clip"
            assert processor.model_name == "ViT-B/32"
            assert processor.current_model is not None
            assert processor.current_model["provider"] == "clip"

    def test_init_with_siglip_provider(self, mock_settings):
        """Test initialization with SigLIP provider."""
        mock_settings.image_embedding_provider = "siglip"
        mock_settings.image_embedding_model = "google/siglip-so400m-patch14-384"
        
        with patch('src.vector.modern_vision_processor.SiglipModel') as mock_siglip_model, \
             patch('src.vector.modern_vision_processor.SiglipProcessor') as mock_siglip_processor:
            
            mock_model = MagicMock()
            mock_model.config.projection_dim = 768
            mock_siglip_model.from_pretrained.return_value = mock_model
            mock_siglip_processor.from_pretrained.return_value = MagicMock()
            
            processor = VisionProcessor(mock_settings)
            
            assert processor.provider == "siglip"
            assert processor.current_model["provider"] == "siglip"
            assert processor.current_model["embedding_size"] == 768

    def test_init_with_jinaclip_provider(self, mock_settings):
        """Test initialization with JinaCLIP provider."""
        mock_settings.image_embedding_provider = "jinaclip"
        mock_settings.image_embedding_model = "jinaai/jina-clip-v2"
        
        with patch('src.vector.modern_vision_processor.AutoModel') as mock_auto_model, \
             patch('src.vector.modern_vision_processor.AutoProcessor') as mock_auto_processor:
            
            mock_auto_model.from_pretrained.return_value = MagicMock()
            mock_auto_processor.from_pretrained.return_value = MagicMock()
            
            processor = VisionProcessor(mock_settings)
            
            assert processor.provider == "jinaclip"
            assert processor.current_model["provider"] == "jinaclip"
            assert processor.current_model["embedding_size"] == 768

    def test_detect_model_provider(self, mock_settings):
        """Test model provider detection."""
        processor = ModernVisionProcessor.__new__(ModernVisionProcessor)
        
        # Test SigLIP detection
        assert processor._detect_model_provider("google/siglip-so400m-patch14-384") == "siglip"
        assert processor._detect_model_provider("siglip-base") == "siglip"
        
        # Test JinaCLIP detection
        assert processor._detect_model_provider("jinaai/jina-clip-v2") == "jinaclip"
        assert processor._detect_model_provider("jina-clip-v1") == "jinaclip"
        
        # Test CLIP detection (default)
        assert processor._detect_model_provider("ViT-B/32") == "clip"
        assert processor._detect_model_provider("RN50") == "clip"
        assert processor._detect_model_provider("unknown-model") == "clip"

    def test_encode_image_with_clip(self, mock_settings, sample_image_b64):
        """Test image encoding with CLIP model."""
        mock_settings.image_embedding_provider = "clip"
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip, \
             patch('src.vector.modern_vision_processor.torch') as mock_torch:
            
            # Mock CLIP model
            mock_model = MagicMock()
            mock_preprocess = MagicMock()
            mock_clip.load.return_value = (mock_model, mock_preprocess)
            
            # Mock torch operations
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            
            # Mock image features
            mock_features = MagicMock()
            mock_features.norm.return_value = mock_features
            mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 512
            mock_model.encode_image.return_value = mock_features
            
            processor = VisionProcessor(mock_settings)
            embedding = processor.encode_image(sample_image_b64)
            
            assert embedding is not None
            assert len(embedding) == 512
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_image_with_siglip(self, mock_settings, sample_image_b64):
        """Test image encoding with SigLIP model."""
        mock_settings.image_embedding_provider = "siglip"
        mock_settings.image_embedding_model = "google/siglip-so400m-patch14-384"
        
        with patch('src.vector.modern_vision_processor.SiglipModel') as mock_siglip_model, \
             patch('src.vector.modern_vision_processor.SiglipProcessor') as mock_siglip_processor, \
             patch('src.vector.modern_vision_processor.torch') as mock_torch:
            
            # Mock SigLIP model
            mock_model = MagicMock()
            mock_model.config.projection_dim = 768
            mock_processor = MagicMock()
            mock_siglip_model.from_pretrained.return_value = mock_model
            mock_siglip_processor.from_pretrained.return_value = mock_processor
            
            # Mock torch operations
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            
            # Mock image features
            mock_features = MagicMock()
            mock_features.norm.return_value = mock_features
            mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 768
            mock_model.get_image_features.return_value = mock_features
            
            processor = VisionProcessor(mock_settings)
            embedding = processor.encode_image(sample_image_b64)
            
            assert embedding is not None
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_image_with_jinaclip(self, mock_settings, sample_image_b64):
        """Test image encoding with JinaCLIP model."""
        mock_settings.image_embedding_provider = "jinaclip"
        mock_settings.image_embedding_model = "jinaai/jina-clip-v2"
        
        with patch('src.vector.modern_vision_processor.AutoModel') as mock_auto_model, \
             patch('src.vector.modern_vision_processor.AutoProcessor') as mock_auto_processor, \
             patch('src.vector.modern_vision_processor.torch') as mock_torch:
            
            # Mock JinaCLIP model
            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_processor.from_pretrained.return_value = mock_processor
            
            # Mock torch operations
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            
            # Mock image features
            mock_features = MagicMock()
            mock_features.norm.return_value = mock_features
            mock_features.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 768
            mock_model.get_image_features.return_value = mock_features
            
            processor = VisionProcessor(mock_settings)
            embedding = processor.encode_image(sample_image_b64)
            
            assert embedding is not None
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_image_with_fallback(self, mock_settings, sample_image_b64):
        """Test image encoding with fallback model."""
        mock_settings.image_embedding_provider = "clip"
        mock_settings.enable_model_fallback = True
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            # Mock primary model failure
            mock_primary_model = MagicMock()
            mock_primary_preprocess = MagicMock()
            mock_clip.load.side_effect = [
                (mock_primary_model, mock_primary_preprocess),  # Primary model
                (MagicMock(), MagicMock())  # Fallback model
            ]
            
            processor = VisionProcessor(mock_settings)
            
            # Mock primary model encoding failure
            with patch.object(processor, '_encode_image_with_model') as mock_encode:
                mock_encode.side_effect = [None, [0.1] * 512]  # First call fails, second succeeds
                
                embedding = processor.encode_image(sample_image_b64)
                
                assert embedding is not None
                assert len(embedding) == 512
                assert mock_encode.call_count == 2

    def test_encode_images_batch(self, mock_settings):
        """Test batch image encoding."""
        mock_settings.image_embedding_provider = "clip"
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            
            # Mock encode_image method
            with patch.object(processor, 'encode_image') as mock_encode:
                mock_encode.return_value = [0.1] * 512
                
                image_list = ["image1", "image2", "image3"]
                embeddings = processor.encode_images_batch(image_list, batch_size=2)
                
                assert len(embeddings) == 3
                assert all(len(emb) == 512 for emb in embeddings)
                assert mock_encode.call_count == 3

    def test_switch_model(self, mock_settings):
        """Test model switching."""
        mock_settings.image_embedding_provider = "clip"
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            
            # Test switching to SigLIP
            with patch.object(processor, '_create_model') as mock_create:
                mock_create.return_value = {
                    "provider": "siglip",
                    "model_name": "google/siglip-so400m-patch14-384",
                    "embedding_size": 768
                }
                
                success = processor.switch_model("siglip", "google/siglip-so400m-patch14-384")
                
                assert success is True
                assert processor.provider == "siglip"
                assert processor.model_name == "google/siglip-so400m-patch14-384"

    def test_get_model_info(self, mock_settings):
        """Test getting model information."""
        mock_settings.image_embedding_provider = "clip"
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            info = processor.get_model_info()
            
            assert info["provider"] == "clip"
            assert info["model_name"] == "ViT-B/32"
            assert info["embedding_size"] == 512
            assert info["input_resolution"] == 224
            assert "supports_text" in info
            assert "supports_image" in info

    def test_decode_base64_image(self, mock_settings, sample_image_b64):
        """Test base64 image decoding."""
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            image = processor._decode_base64_image(sample_image_b64)
            
            assert image is not None
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"

    def test_decode_base64_image_with_data_url(self, mock_settings, sample_image_b64):
        """Test base64 image decoding with data URL format."""
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            data_url = f"data:image/png;base64,{sample_image_b64}"
            image = processor._decode_base64_image(data_url)
            
            assert image is not None
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"

    def test_validate_image_data(self, mock_settings, sample_image_b64):
        """Test image data validation."""
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            
            # Test valid image data
            assert processor.validate_image_data(sample_image_b64) is True
            
            # Test invalid image data
            assert processor.validate_image_data("invalid_base64") is False
            assert processor.validate_image_data("") is False

    def test_get_supported_formats(self, mock_settings):
        """Test getting supported image formats."""
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            formats = processor.get_supported_formats()
            
            assert isinstance(formats, list)
            assert "jpg" in formats
            assert "jpeg" in formats
            assert "png" in formats
            assert "webp" in formats

    def test_error_handling(self, mock_settings):
        """Test error handling in various scenarios."""
        mock_settings.image_embedding_provider = "clip"
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            # Test initialization failure
            mock_clip.load.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                ModernVisionProcessor(mock_settings)

    def test_warm_up_models(self, mock_settings):
        """Test model warm-up functionality."""
        mock_settings.image_embedding_provider = "clip"
        mock_settings.model_warm_up = True
        
        with patch('src.vector.modern_vision_processor.clip') as mock_clip:
            mock_clip.load.return_value = (MagicMock(), MagicMock())
            
            processor = VisionProcessor(mock_settings)
            
            # Mock warm-up method
            with patch.object(processor, '_warm_up_models') as mock_warm_up:
                processor._warm_up_models()
                mock_warm_up.assert_called_once()


class TestVisionProcessorIntegration:
    """Integration tests for vision processor."""

    @pytest.mark.integration
    def test_real_initialization_with_clip(self):
        """Test real initialization with CLIP model."""
        try:
            from src.config import get_settings
            settings = get_settings()
            settings.image_embedding_provider = "clip"
            settings.image_embedding_model = "ViT-B/32"
            settings.model_warm_up = False
            
            with patch('src.vector.modern_vision_processor.torch.cuda.is_available', return_value=False):
                processor = ModernVisionProcessor(settings)
                
                assert processor.provider == "clip"
                assert processor.current_model is not None
                assert processor.current_model["provider"] == "clip"
                
        except ImportError:
            pytest.skip("CLIP dependencies not available")

    @pytest.mark.integration
    def test_model_switching_integration(self):
        """Test model switching integration."""
        try:
            from src.config import get_settings
            settings = get_settings()
            settings.image_embedding_provider = "clip"
            settings.model_warm_up = False
            
            with patch('src.vector.modern_vision_processor.torch.cuda.is_available', return_value=False):
                processor = ModernVisionProcessor(settings)
                
                # Get initial model info
                initial_info = processor.get_model_info()
                assert initial_info["provider"] == "clip"
                
                # Test model info structure
                assert "embedding_size" in initial_info
                assert "input_resolution" in initial_info
                assert "supports_text" in initial_info
                assert "supports_image" in initial_info
                
        except ImportError:
            pytest.skip("Dependencies not available")