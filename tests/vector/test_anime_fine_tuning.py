"""Comprehensive tests for anime domain-specific fine-tuning system.

Tests the complete fine-tuning pipeline including dataset preparation,
character recognition, art style classification, and genre enhancement.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import torch

from src.vector.anime_fine_tuning import AnimeFineTuner, FineTuningConfig
from src.vector.anime_dataset import AnimeDataset, DataSample
from src.vector.character_recognition import CharacterRecognitionFinetuner
from src.vector.art_style_classifier import ArtStyleClassifier
from src.vector.genre_enhancement import GenreEnhancementFinetuner


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.enable_fine_tuning = True
    settings.fine_tuning_data_path = "test_data.json"
    settings.fine_tuning_model_dir = "test_models"
    settings.fine_tuning_use_lora = True
    settings.fine_tuning_lora_r = 8
    settings.fine_tuning_lora_alpha = 32
    settings.fine_tuning_lora_dropout = 0.1
    settings.fine_tuning_batch_size = 4
    settings.fine_tuning_learning_rate = 1e-4
    settings.fine_tuning_num_epochs = 2
    settings.character_recognition_enabled = True
    settings.art_style_classification_enabled = True
    settings.genre_enhancement_enabled = True
    settings.text_embedding_provider = "sentence-transformers"
    settings.text_embedding_model = "all-MiniLM-L6-v2"
    settings.vision_embedding_provider = "sentence-transformers"
    settings.vision_embedding_model = "clip-ViT-B-32"
    return settings


@pytest.fixture
def sample_anime_data():
    """Sample anime data for testing."""
    return [
        {
            "title": "Attack on Titan",
            "type": "TV",
            "synopsis": "Humanity fights against giant humanoid Titans to survive.",
            "tags": ["action", "drama", "supernatural", "military"],
            "studios": ["Wit Studio", "Mappa"],
            "sources": ["https://myanimelist.net/anime/16498/"],
            "picture": "https://example.com/aot.jpg",
            "animeSeason": {"year": 2013, "season": "SPRING"}
        },
        {
            "title": "Your Name",
            "type": "MOVIE",
            "synopsis": "Two teenagers share a profound, magical connection through their dreams.",
            "tags": ["romance", "supernatural", "drama"],
            "studios": ["CoMix Wave Films"],
            "sources": ["https://myanimelist.net/anime/32281/"],
            "picture": "https://example.com/yourname.jpg",
            "animeSeason": {"year": 2016, "season": "SUMMER"}
        },
        {
            "title": "Studio Ghibli Movie",
            "type": "MOVIE",
            "synopsis": "A magical adventure in a fantasy world.",
            "tags": ["adventure", "fantasy", "family"],
            "studios": ["Studio Ghibli"],
            "sources": ["https://myanimelist.net/anime/164/"],
            "picture": "https://example.com/ghibli.jpg",
            "animeSeason": {"year": 1988, "season": "SPRING"}
        }
    ]


@pytest.fixture
def mock_text_processor():
    """Mock text processor."""
    processor = MagicMock()
    processor.encode_text.return_value = np.array([0.1] * 384)
    processor.get_model_info.return_value = {"embedding_size": 384}
    return processor


@pytest.fixture
def mock_vision_processor():
    """Mock vision processor."""
    processor = MagicMock()
    processor.encode_image.return_value = np.array([0.2] * 512)
    processor.get_model_info.return_value = {"embedding_size": 512}
    return processor


class TestAnimeDataset:
    """Test cases for AnimeDataset."""
    
    def test_dataset_creation(self, sample_anime_data, mock_text_processor, mock_vision_processor):
        """Test dataset creation from anime data."""
        config = FineTuningConfig()
        dataset = AnimeDataset(
            anime_data=sample_anime_data,
            text_processor=mock_text_processor,
            vision_processor=mock_vision_processor,
            config=config,
            augment_data=False
        )
        
        assert len(dataset.samples) == len(sample_anime_data)
        assert len(dataset.character_vocab) >= 0
        assert len(dataset.art_style_vocab) >= 0
        assert len(dataset.genre_vocab) >= 0
    
    def test_dataset_augmentation(self, sample_anime_data, mock_text_processor, mock_vision_processor):
        """Test dataset with augmentation enabled."""
        config = FineTuningConfig()
        dataset = AnimeDataset(
            anime_data=sample_anime_data,
            text_processor=mock_text_processor,
            vision_processor=mock_vision_processor,
            config=config,
            augment_data=True
        )
        
        # Should have more samples than original data due to augmentation
        assert len(dataset.samples) > len(sample_anime_data)
    
    def test_dataset_getitem(self, sample_anime_data, mock_text_processor, mock_vision_processor):
        """Test dataset item retrieval."""
        config = FineTuningConfig()
        dataset = AnimeDataset(
            anime_data=sample_anime_data,
            text_processor=mock_text_processor,
            vision_processor=mock_vision_processor,
            config=config,
            augment_data=False
        )
        
        item = dataset[0]
        
        assert 'anime_id' in item
        assert 'title' in item
        assert 'text' in item
        assert 'text_embedding' in item
        assert 'character_labels' in item
        assert 'art_style_label' in item
        assert 'genre_labels' in item
        assert 'metadata' in item
    
    def test_vocabulary_creation(self, sample_anime_data, mock_text_processor, mock_vision_processor):
        """Test vocabulary creation from dataset."""
        config = FineTuningConfig()
        dataset = AnimeDataset(
            anime_data=sample_anime_data,
            text_processor=mock_text_processor,
            vision_processor=mock_vision_processor,
            config=config,
            augment_data=False
        )
        
        vocab_sizes = dataset.get_vocab_sizes()
        
        assert 'character' in vocab_sizes
        assert 'art_style' in vocab_sizes
        assert 'genre' in vocab_sizes
        assert all(size >= 0 for size in vocab_sizes.values())
    
    def test_class_weights(self, sample_anime_data, mock_text_processor, mock_vision_processor):
        """Test class weight calculation."""
        config = FineTuningConfig()
        dataset = AnimeDataset(
            anime_data=sample_anime_data,
            text_processor=mock_text_processor,
            vision_processor=mock_vision_processor,
            config=config,
            augment_data=False
        )
        
        class_weights = dataset.get_class_weights()
        
        assert 'character' in class_weights
        assert 'art_style' in class_weights
        assert 'genre' in class_weights
        assert all(isinstance(weight, torch.Tensor) for weight in class_weights.values())


class TestCharacterRecognitionFinetuner:
    """Test cases for CharacterRecognitionFinetuner."""
    
    def test_initialization(self, mock_settings, mock_text_processor, mock_vision_processor):
        """Test character recognition finetuner initialization."""
        finetuner = CharacterRecognitionFinetuner(mock_settings, mock_text_processor, mock_vision_processor)
        
        assert finetuner.settings == mock_settings
        assert finetuner.device is not None
        assert finetuner.num_characters == 0
        assert not finetuner.is_trained
    
    def test_lora_setup(self, mock_settings, mock_text_processor, mock_vision_processor):
        """Test LoRA model setup."""
        finetuner = CharacterRecognitionFinetuner(mock_settings, mock_text_processor, mock_vision_processor)
        
        from peft import LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        finetuner.setup_lora_model(lora_config, FineTuningConfig())
        
        assert finetuner.recognition_model is not None
        assert finetuner.optimizer is not None
        assert finetuner.loss_fn is not None
    
    def test_enhanced_embedding_without_model(self, mock_settings, mock_text_processor, mock_vision_processor):
        """Test enhanced embedding generation without initialized model."""
        finetuner = CharacterRecognitionFinetuner(mock_settings, mock_text_processor, mock_vision_processor)
        
        text_embedding = np.array([0.1] * 384)
        enhanced = finetuner.get_enhanced_embedding(text_embedding)
        
        # Should return original embedding when model not initialized
        assert np.array_equal(enhanced, text_embedding)
    
    def test_model_info(self, mock_settings, mock_text_processor, mock_vision_processor):
        """Test model information retrieval."""
        finetuner = CharacterRecognitionFinetuner(mock_settings, mock_text_processor, mock_vision_processor)
        
        info = finetuner.get_model_info()
        
        assert 'num_characters' in info
        assert 'character_vocab_size' in info
        assert 'is_trained' in info
        assert 'device' in info
        assert 'model_type' in info


class TestArtStyleClassifier:
    """Test cases for ArtStyleClassifier."""
    
    def test_initialization(self, mock_settings, mock_vision_processor):
        """Test art style classifier initialization."""
        classifier = ArtStyleClassifier(mock_settings, mock_vision_processor)
        
        assert classifier.settings == mock_settings
        assert classifier.device is not None
        assert classifier.num_styles == 0
        assert not classifier.is_trained
    
    def test_lora_setup(self, mock_settings, mock_vision_processor):
        """Test LoRA model setup."""
        classifier = ArtStyleClassifier(mock_settings, mock_vision_processor)
        
        from peft import LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, # Changed from IMAGE_CLASSIFICATION
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        classifier.setup_lora_model(lora_config, FineTuningConfig())
        
        assert classifier.classifier_model is not None
        assert classifier.optimizer is not None
        assert classifier.loss_fn is not None
    
    def test_enhanced_embedding_without_model(self, mock_settings, mock_vision_processor):
        """Test enhanced embedding generation without initialized model."""
        classifier = ArtStyleClassifier(mock_settings, mock_vision_processor)
        
        image_embedding = np.array([0.2] * 512)
        enhanced = classifier.get_enhanced_embedding(image_embedding)
        
        # Should return original embedding when model not initialized
        assert np.array_equal(enhanced, image_embedding)
    
    def test_era_from_year(self, mock_settings, mock_vision_processor):
        """Test era inference from year."""
        classifier = ArtStyleClassifier(mock_settings, mock_vision_processor)
        
        assert classifier._get_era_from_year(1975) == 'vintage'
        assert classifier._get_era_from_year(1990) == 'classic'
        assert classifier._get_era_from_year(2000) == 'digital'
        assert classifier._get_era_from_year(2010) == 'modern'
        assert classifier._get_era_from_year(2020) == 'contemporary'
        assert classifier._get_era_from_year(None) == 'unknown'


class TestGenreEnhancementFinetuner:
    """Test cases for GenreEnhancementFinetuner."""
    
    def test_initialization(self, mock_settings, mock_text_processor):
        """Test genre enhancement finetuner initialization."""
        enhancer = GenreEnhancementFinetuner(mock_settings, mock_text_processor)
        
        assert enhancer.settings == mock_settings
        assert enhancer.device is not None
        assert enhancer.num_genres == 0
        assert not enhancer.is_trained
    
    def test_lora_setup(self, mock_settings, mock_text_processor):
        """Test LoRA model setup."""
        enhancer = GenreEnhancementFinetuner(mock_settings, mock_text_processor)
        
        from peft import LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        enhancer.setup_lora_model(lora_config, FineTuningConfig())
        
        assert enhancer.enhancement_model is not None
        assert enhancer.optimizer is not None
        assert enhancer.loss_fn is not None
    
    def test_auxiliary_label_inference(self, mock_settings, mock_text_processor):
        """Test auxiliary label inference from sample data."""
        enhancer = GenreEnhancementFinetuner(mock_settings, mock_text_processor)
        
        # Create auxiliary vocabularies
        enhancer.theme_vocab = {'school': 0, 'military': 1, 'magic': 2}
        enhancer.target_vocab = {'shounen': 0, 'shoujo': 1, 'general': 2}
        enhancer.mood_vocab = {'funny': 0, 'dark': 1, 'romantic': 2, 'general': 3}
        
        # Mock sample
        class MockSample:
            def __init__(self, tags):
                self.tags = tags
        
        sample = MockSample(['action', 'shounen', 'comedy'])
        themes, target, mood = enhancer._infer_auxiliary_labels(sample)
        
        assert target == 'shounen'
        assert mood == 'funny'


class TestAnimeFineTuner:
    """Test cases for AnimeFineTuner (main orchestrator)."""
    
    @patch('src.vector.anime_fine_tuning.TextProcessor')
    @patch('src.vector.anime_fine_tuning.VisionProcessor')
    @patch('src.vector.anime_fine_tuning.CharacterRecognitionFinetuner')
    @patch('src.vector.anime_fine_tuning.ArtStyleClassifier')
    @patch('src.vector.anime_fine_tuning.GenreEnhancementFinetuner')
    def test_initialization(self, mock_genre, mock_art, mock_char, mock_vision, mock_text, mock_settings):
        """Test anime fine-tuner initialization."""
        finetuner = AnimeFineTuner(mock_settings)
        
        assert finetuner.settings == mock_settings
        assert finetuner.config is not None
        assert finetuner.text_processor is not None
        assert finetuner.vision_processor is not None
        assert finetuner.character_finetuner is not None
        assert finetuner.art_style_classifier is not None
        assert finetuner.genre_enhancer is not None
    
    @patch('src.vector.anime_fine_tuning.TextProcessor')
    @patch('src.vector.anime_fine_tuning.VisionProcessor')
    @patch('src.vector.anime_fine_tuning.CharacterRecognitionFinetuner')
    @patch('src.vector.anime_fine_tuning.ArtStyleClassifier')
    @patch('src.vector.anime_fine_tuning.GenreEnhancementFinetuner')
    def test_dataset_preparation(self, mock_genre, mock_art, mock_char, mock_vision, mock_text, mock_settings, sample_anime_data):
        """Test dataset preparation."""
        finetuner = AnimeFineTuner(mock_settings)
        
        # Create temporary data file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_anime_data, f)
            temp_path = f.name
        
        try:
            dataset = finetuner.prepare_dataset(temp_path)
            assert isinstance(dataset, AnimeDataset)
            assert len(dataset.samples) >= len(sample_anime_data)
        finally:
            Path(temp_path).unlink()
    
    @patch('src.vector.anime_fine_tuning.TextProcessor')
    @patch('src.vector.anime_fine_tuning.VisionProcessor')
    @patch('src.vector.anime_fine_tuning.CharacterRecognitionFinetuner')
    @patch('src.vector.anime_fine_tuning.ArtStyleClassifier')
    @patch('src.vector.anime_fine_tuning.GenreEnhancementFinetuner')
    def test_lora_config_creation(self, mock_genre, mock_art, mock_char, mock_vision, mock_text, mock_settings):
        """Test LoRA configuration creation."""
        finetuner = AnimeFineTuner(mock_settings)
        
        from peft import TaskType
        lora_config = finetuner.create_lora_config(TaskType.FEATURE_EXTRACTION)
        
        assert lora_config is not None
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 32
        assert lora_config.lora_dropout == 0.1
    
    @patch('src.vector.anime_fine_tuning.TextProcessor')
    @patch('src.vector.anime_fine_tuning.VisionProcessor')
    @patch('src.vector.anime_fine_tuning.CharacterRecognitionFinetuner')
    @patch('src.vector.anime_fine_tuning.ArtStyleClassifier')
    @patch('src.vector.anime_fine_tuning.GenreEnhancementFinetuner')
    def test_enhanced_embeddings(self, mock_genre, mock_art, mock_char, mock_vision, mock_text, mock_settings):
        """Test enhanced embeddings generation."""
        # Setup mocks
        mock_text.return_value.encode_text.return_value = np.array([0.1] * 384)
        mock_vision.return_value.encode_image.return_value = np.array([0.2] * 512)
        mock_char.return_value.get_enhanced_embedding.return_value = np.array([0.3] * 512)
        mock_art.return_value.get_enhanced_embedding.return_value = np.array([0.4] * 256)
        mock_genre.return_value.get_enhanced_embedding.return_value = np.array([0.5] * 384)
        
        finetuner = AnimeFineTuner(mock_settings)
        
        embeddings = finetuner.get_enhanced_embeddings(
            text="test anime query",
            image_data="fake_image_data"
        )
        
        assert 'character' in embeddings
        assert 'art_style' in embeddings
        assert 'genre' in embeddings
    
    @patch('src.vector.anime_fine_tuning.TextProcessor')
    @patch('src.vector.anime_fine_tuning.VisionProcessor')
    @patch('src.vector.anime_fine_tuning.CharacterRecognitionFinetuner')
    @patch('src.vector.anime_fine_tuning.ArtStyleClassifier')
    @patch('src.vector.anime_fine_tuning.GenreEnhancementFinetuner')
    def test_training_summary(self, mock_genre, mock_art, mock_char, mock_vision, mock_text, mock_settings):
        """Test training summary generation."""
        finetuner = AnimeFineTuner(mock_settings)
        finetuner.training_stats = {'total_loss': [0.5, 0.3, 0.2]}
        finetuner.best_model_path = Path("test/path")
        
        summary = finetuner.get_training_summary()
        
        assert 'config' in summary
        assert 'training_stats' in summary
        assert 'best_model_path' in summary
        assert 'timestamp' in summary


@pytest.mark.integration
class TestFineTuningIntegration:
    """Integration tests for the complete fine-tuning pipeline."""
    
    def test_config_validation(self):
        """Test fine-tuning configuration validation."""
        from src.config import Settings
        
        # Test valid configuration
        settings = Settings(
            enable_fine_tuning=True,
            fine_tuning_lora_r=16,
            fine_tuning_lora_alpha=64,
            fine_tuning_batch_size=8
        )
        
        assert settings.enable_fine_tuning is True
        assert settings.fine_tuning_lora_r == 16
        assert settings.fine_tuning_lora_alpha == 64
        assert settings.fine_tuning_batch_size == 8
    
    def test_invalid_config_validation(self):
        """Test invalid configuration validation."""
        from src.config import Settings
        
        # Test invalid LoRA rank (too high)
        with pytest.raises(Exception):
            Settings(fine_tuning_lora_r=100)
        
        # Test invalid learning rate (too high)
        with pytest.raises(Exception):
            Settings(fine_tuning_learning_rate=1.0)
    
    @patch('torch.cuda.is_available')
    def test_device_selection(self, mock_cuda):
        """Test device selection for training."""
        mock_cuda.return_value = False
        
        from src.vector.character_recognition import CharacterRecognitionFinetuner
        
        finetuner = CharacterRecognitionFinetuner(mock_settings, MagicMock(), MagicMock())
        assert str(finetuner.device) == 'cpu'
        
        mock_cuda.return_value = True
        finetuner = CharacterRecognitionFinetuner(mock_settings, MagicMock(), MagicMock())
        assert 'cuda' in str(finetuner.device)
    
    def test_data_sample_creation(self):
        """Test data sample creation with real data structure."""
        from src.vector.anime_dataset import DataSample
        
        sample = DataSample(
            anime_id="test_123",
            title="Test Anime",
            text="Test anime description",
            character_labels=["character1", "character2"],
            art_style_label="modern",
            genre_labels=["action", "adventure"]
        )
        
        assert sample.anime_id == "test_123"
        assert sample.title == "Test Anime"
        assert len(sample.character_labels) == 2
        assert sample.art_style_label == "modern"
        assert len(sample.genre_labels) == 2


class TestFineTuningErrorHandling:
    """Test error handling in fine-tuning system."""
    
    def test_missing_model_error(self, mock_settings, mock_text_processor, mock_vision_processor):
        """Test error when trying to train without initialized model."""
        finetuner = CharacterRecognitionFinetuner(mock_settings, mock_text_processor, mock_vision_processor)
        
        batch = {
            'text_embedding': torch.tensor([[0.1] * 384]),
            'image_embedding': torch.tensor([[0.2] * 512]),
            'character_labels': torch.tensor([[1.0, 0.0, 0.0]])
        }
        
        with pytest.raises(RuntimeError, match="Model not initialized"):
            finetuner.train_step(batch)
    
    def test_invalid_data_path(self, mock_settings):
        """Test error with invalid data path."""
        with patch('src.vector.anime_fine_tuning.TextProcessor'), \
             patch('src.vector.anime_fine_tuning.VisionProcessor'), \
             patch('src.vector.anime_fine_tuning.CharacterRecognitionFinetuner'), \
             patch('src.vector.anime_fine_tuning.ArtStyleClassifier'), \
             patch('src.vector.anime_fine_tuning.GenreEnhancementFinetuner'):
            
            finetuner = AnimeFineTuner(mock_settings)
            
            result = finetuner.prepare_dataset("nonexistent_file.json")
            assert result is None
    
    def test_empty_dataset_handling(self, mock_text_processor, mock_vision_processor):
        """Test handling of empty dataset."""
        config = FineTuningConfig()
        dataset = AnimeDataset(
            anime_data=[],
            text_processor=mock_text_processor,
            vision_processor=mock_vision_processor,
            config=config,
            augment_data=False
        )
        
        assert len(dataset.samples) == 0
        assert len(dataset.character_vocab) == 0
        assert len(dataset.art_style_vocab) == 0
        assert len(dataset.genre_vocab) == 0