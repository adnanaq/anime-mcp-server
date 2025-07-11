"""Tests for text processor supporting multiple embedding models."""

import pytest
from unittest.mock import MagicMock, patch, Mock
import numpy as np

from src.vector.text_processor import TextProcessor


class TestTextProcessor:
    """Test cases for text processor."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.text_embedding_provider = "fastembed"
        settings.text_embedding_model = "BAAI/bge-small-en-v1.5"
        settings.text_embedding_model_fallback = "BAAI/bge-base-en-v1.5"
        settings.model_cache_dir = None
        settings.enable_model_fallback = True
        settings.model_warm_up = False
        settings.bge_model_version = "v1.5"
        settings.bge_model_size = "small"
        settings.bge_enable_multilingual = False
        settings.bge_max_length = 512
        return settings

    @pytest.fixture
    def mock_fastembed_model(self):
        """Mock FastEmbed model."""
        model = MagicMock()
        model.embed.return_value = [np.array([0.1] * 384)]
        return model

    @pytest.fixture
    def mock_hf_model(self):
        """Mock HuggingFace model."""
        model = MagicMock()
        model.config.hidden_size = 768
        return model

    @pytest.fixture
    def mock_sentence_transformer_model(self):
        """Mock Sentence Transformers model."""
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 768
        model.max_seq_length = 512
        model.encode.return_value = np.array([0.1] * 768)
        return model

    def test_init_with_fastembed_provider(self, mock_settings):
        """Test initialization with FastEmbed provider."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            assert processor.provider == "fastembed"
            assert processor.model_name == "BAAI/bge-small-en-v1.5"
            assert processor.current_model is not None
            assert processor.current_model["provider"] == "fastembed"

    def test_init_with_huggingface_provider(self, mock_settings):
        """Test initialization with HuggingFace provider."""
        mock_settings.text_embedding_provider = "huggingface"
        mock_settings.text_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        with patch('src.vector.modern_text_processor.AutoModel') as mock_auto_model, \
             patch('src.vector.modern_text_processor.AutoTokenizer') as mock_auto_tokenizer:
            
            mock_model = MagicMock()
            mock_model.config.hidden_size = 384
            mock_tokenizer = MagicMock()
            mock_tokenizer.model_max_length = 512
            
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            processor = TextProcessor(mock_settings)
            
            assert processor.provider == "huggingface"
            assert processor.current_model["provider"] == "huggingface"
            assert processor.current_model["embedding_size"] == 384

    def test_init_with_sentence_transformers_provider(self, mock_settings):
        """Test initialization with Sentence Transformers provider."""
        mock_settings.text_embedding_provider = "sentence-transformers"
        mock_settings.text_embedding_model = "all-MiniLM-L6-v2"
        
        with patch('src.vector.modern_text_processor.SentenceTransformer') as mock_sentence_transformer:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.max_seq_length = 512
            mock_sentence_transformer.return_value = mock_model
            
            processor = TextProcessor(mock_settings)
            
            assert processor.provider == "sentence-transformers"
            assert processor.current_model["provider"] == "sentence-transformers"
            assert processor.current_model["embedding_size"] == 384

    def test_detect_model_provider(self, mock_settings):
        """Test model provider detection."""
        processor = ModernTextProcessor.__new__(ModernTextProcessor)
        
        # Test FastEmbed detection
        assert processor._detect_model_provider("BAAI/bge-small-en-v1.5") == "fastembed"
        assert processor._detect_model_provider("sentence-transformers/all-MiniLM-L6-v2") == "fastembed"
        assert processor._detect_model_provider("intfloat/e5-small-v2") == "fastembed"
        
        # Test Sentence Transformers detection
        assert processor._detect_model_provider("sentence-transformers/all-mpnet-base-v2") == "sentence-transformers"
        
        # Test HuggingFace detection (default)
        assert processor._detect_model_provider("bert-base-uncased") == "huggingface"
        assert processor._detect_model_provider("unknown-model") == "huggingface"

    def test_get_fastembed_embedding_size(self, mock_settings):
        """Test FastEmbed embedding size detection."""
        processor = ModernTextProcessor.__new__(ModernTextProcessor)
        
        # Test known models
        assert processor._get_fastembed_embedding_size("BAAI/bge-small-en-v1.5") == 384
        assert processor._get_fastembed_embedding_size("BAAI/bge-base-en-v1.5") == 768
        assert processor._get_fastembed_embedding_size("BAAI/bge-large-en-v1.5") == 1024
        
        # Test unknown model (default)
        assert processor._get_fastembed_embedding_size("unknown-model") == 384

    def test_is_multilingual_model(self, mock_settings):
        """Test multilingual model detection."""
        processor = ModernTextProcessor.__new__(ModernTextProcessor)
        
        # Test multilingual models
        assert processor._is_multilingual_model("BAAI/bge-m3") is True
        assert processor._is_multilingual_model("xlm-roberta-base") is True
        assert processor._is_multilingual_model("distilbert-base-multilingual-cased") is True
        
        # Test non-multilingual models
        assert processor._is_multilingual_model("BAAI/bge-small-en-v1.5") is False
        assert processor._is_multilingual_model("bert-base-uncased") is False

    def test_encode_text_with_fastembed(self, mock_settings):
        """Test text encoding with FastEmbed model."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_model = MagicMock()
            mock_model.embed.return_value = [np.array([0.1] * 384)]
            mock_text_embedding.return_value = mock_model
            
            processor = TextProcessor(mock_settings)
            embedding = processor.encode_text("test anime query")
            
            assert embedding is not None
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_text_with_huggingface(self, mock_settings):
        """Test text encoding with HuggingFace model."""
        mock_settings.text_embedding_provider = "huggingface"
        mock_settings.text_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        with patch('src.vector.modern_text_processor.AutoModel') as mock_auto_model, \
             patch('src.vector.modern_text_processor.AutoTokenizer') as mock_auto_tokenizer, \
             patch('src.vector.modern_text_processor.torch') as mock_torch:
            
            # Mock model and tokenizer
            mock_model = MagicMock()
            mock_model.config.hidden_size = 384
            mock_tokenizer = MagicMock()
            mock_tokenizer.model_max_length = 512
            mock_tokenizer.return_value = MagicMock()
            
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
            
            # Mock torch operations
            mock_torch.cuda.is_available.return_value = False
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            
            # Mock model outputs
            mock_outputs = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value.norm.return_value = MagicMock()
            mock_outputs.last_hidden_state.mean.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [0.1] * 384
            mock_model.return_value = mock_outputs
            
            processor = TextProcessor(mock_settings)
            embedding = processor.encode_text("test anime query")
            
            assert embedding is not None
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_text_with_sentence_transformers(self, mock_settings):
        """Test text encoding with Sentence Transformers model."""
        mock_settings.text_embedding_provider = "sentence-transformers"
        mock_settings.text_embedding_model = "all-MiniLM-L6-v2"
        
        with patch('src.vector.modern_text_processor.SentenceTransformer') as mock_sentence_transformer:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.max_seq_length = 512
            mock_model.encode.return_value = np.array([0.1] * 384)
            mock_sentence_transformer.return_value = mock_model
            
            processor = TextProcessor(mock_settings)
            embedding = processor.encode_text("test anime query")
            
            assert embedding is not None
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

    def test_encode_text_with_fallback(self, mock_settings):
        """Test text encoding with fallback model."""
        mock_settings.text_embedding_provider = "fastembed"
        mock_settings.enable_model_fallback = True
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            # Mock primary and fallback models
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Mock primary model encoding failure
            with patch.object(processor, '_encode_text_with_model') as mock_encode:
                mock_encode.side_effect = [None, [0.1] * 384]  # First call fails, second succeeds
                
                embedding = processor.encode_text("test anime query")
                
                assert embedding is not None
                assert len(embedding) == 384
                assert mock_encode.call_count == 2

    def test_encode_text_empty_input(self, mock_settings):
        """Test text encoding with empty input."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Test empty string
            embedding = processor.encode_text("")
            assert embedding is not None
            assert len(embedding) == 384
            assert all(x == 0.0 for x in embedding)
            
            # Test whitespace-only string
            embedding = processor.encode_text("   ")
            assert embedding is not None
            assert len(embedding) == 384
            assert all(x == 0.0 for x in embedding)

    def test_encode_texts_batch(self, mock_settings):
        """Test batch text encoding."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Mock encode_text method
            with patch.object(processor, 'encode_text') as mock_encode:
                mock_encode.return_value = [0.1] * 384
                
                text_list = ["anime query 1", "anime query 2", "anime query 3"]
                embeddings = processor.encode_texts_batch(text_list, batch_size=2)
                
                assert len(embeddings) == 3
                assert all(len(emb) == 384 for emb in embeddings)
                assert mock_encode.call_count == 3

    def test_switch_model(self, mock_settings):
        """Test model switching."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Test switching to HuggingFace
            with patch.object(processor, '_create_model') as mock_create:
                mock_create.return_value = {
                    "provider": "huggingface",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_size": 384
                }
                
                success = processor.switch_model("huggingface", "sentence-transformers/all-MiniLM-L6-v2")
                
                assert success is True
                assert processor.provider == "huggingface"
                assert processor.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_model_info(self, mock_settings):
        """Test getting model information."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            info = processor.get_model_info()
            
            assert info["provider"] == "fastembed"
            assert info["model_name"] == "BAAI/bge-small-en-v1.5"
            assert info["embedding_size"] == 384
            assert info["max_length"] == 512
            assert "batch_size" in info
            assert "supports_multilingual" in info

    def test_get_bge_model_name(self, mock_settings):
        """Test BGE model name generation."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Test regular BGE model
            mock_settings.bge_model_version = "v1.5"
            mock_settings.bge_model_size = "small"
            mock_settings.bge_enable_multilingual = False
            assert processor.get_bge_model_name() == "BAAI/bge-small-en-v1.5"
            
            # Test multilingual BGE model
            mock_settings.bge_enable_multilingual = True
            assert processor.get_bge_model_name() == "BAAI/bge-m3"
            
            # Test M3 model
            mock_settings.bge_model_version = "m3"
            assert processor.get_bge_model_name() == "BAAI/bge-m3"
            
            # Test reranker model
            mock_settings.bge_model_version = "reranker"
            mock_settings.bge_model_size = "large"
            mock_settings.bge_enable_multilingual = False
            assert processor.get_bge_model_name() == "BAAI/bge-reranker-large"

    def test_validate_text(self, mock_settings):
        """Test text validation."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Test valid text
            assert processor.validate_text("valid text") is True
            
            # Test invalid text types
            assert processor.validate_text(123) is False
            assert processor.validate_text(None) is False
            assert processor.validate_text([]) is False
            
            # Test very long text
            long_text = "a" * 10000
            assert processor.validate_text(long_text) is False

    def test_create_zero_vector(self, mock_settings):
        """Test zero vector creation."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            zero_vector = processor._create_zero_vector()
            
            assert len(zero_vector) == 384
            assert all(x == 0.0 for x in zero_vector)

    def test_error_handling(self, mock_settings):
        """Test error handling in various scenarios."""
        mock_settings.text_embedding_provider = "fastembed"
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            # Test initialization failure
            mock_text_embedding.side_effect = Exception("Model loading failed")
            
            with pytest.raises(Exception):
                ModernTextProcessor(mock_settings)

    def test_warm_up_models(self, mock_settings):
        """Test model warm-up functionality."""
        mock_settings.text_embedding_provider = "fastembed"
        mock_settings.model_warm_up = True
        
        with patch('src.vector.modern_text_processor.TextEmbedding') as mock_text_embedding:
            mock_text_embedding.return_value = MagicMock()
            
            processor = TextProcessor(mock_settings)
            
            # Mock warm-up method
            with patch.object(processor, '_warm_up_models') as mock_warm_up:
                processor._warm_up_models()
                mock_warm_up.assert_called_once()


class TestTextProcessorIntegration:
    """Integration tests for text processor."""

    @pytest.mark.integration
    def test_real_initialization_with_fastembed(self):
        """Test real initialization with FastEmbed model."""
        try:
            from src.config import get_settings
            settings = get_settings()
            settings.text_embedding_provider = "fastembed"
            settings.text_embedding_model = "BAAI/bge-small-en-v1.5"
            settings.model_warm_up = False
            
            processor = ModernTextProcessor(settings)
            
            assert processor.provider == "fastembed"
            assert processor.current_model is not None
            assert processor.current_model["provider"] == "fastembed"
            
        except ImportError:
            pytest.skip("FastEmbed dependencies not available")

    @pytest.mark.integration
    def test_model_switching_integration(self):
        """Test model switching integration."""
        try:
            from src.config import get_settings
            settings = get_settings()
            settings.text_embedding_provider = "fastembed"
            settings.model_warm_up = False
            
            processor = ModernTextProcessor(settings)
            
            # Get initial model info
            initial_info = processor.get_model_info()
            assert initial_info["provider"] == "fastembed"
            
            # Test model info structure
            assert "embedding_size" in initial_info
            assert "max_length" in initial_info
            assert "supports_multilingual" in initial_info
            
        except ImportError:
            pytest.skip("Dependencies not available")

    @pytest.mark.integration
    def test_encode_text_integration(self):
        """Test text encoding integration."""
        try:
            from src.config import get_settings
            settings = get_settings()
            settings.text_embedding_provider = "fastembed"
            settings.model_warm_up = False
            
            processor = ModernTextProcessor(settings)
            
            # Test text encoding
            embedding = processor.encode_text("This is a test anime query")
            
            assert embedding is not None
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)
            
        except ImportError:
            pytest.skip("Dependencies not available")