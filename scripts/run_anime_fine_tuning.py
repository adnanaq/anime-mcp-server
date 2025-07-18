#!/usr/bin/env python3
"""Script for running anime-specific fine-tuning on the existing dataset.

This script orchestrates the complete fine-tuning pipeline for character recognition,
art style classification, and genre understanding enhancement.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.vector.anime_fine_tuning import AnimeFineTuner, FineTuningConfig
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_anime_data_for_finetuning(data_path: str, output_path: str):
    """Prepare anime data for fine-tuning from existing database.
    
    Args:
        data_path: Path to existing anime database JSON
        output_path: Path to save prepared fine-tuning data
    """
    logger.info(f"Preparing anime data from {data_path}")
    
    # Load existing anime database
    with open(data_path, 'r', encoding='utf-8') as f:
        anime_data = json.load(f)
    
    logger.info(f"Loaded {len(anime_data)} anime entries")
    
    # Filter and enhance data for fine-tuning
    enhanced_data = []
    for anime in anime_data:
        # Basic filtering
        if not anime.get('title') or not anime.get('type'):
            continue
        
        # Ensure required fields
        enhanced_anime = {
            'title': anime.get('title', ''),
            'type': anime.get('type', 'UNKNOWN'),
            'synopsis': anime.get('synopsis', ''),
            'tags': anime.get('tags', []),
            'studios': anime.get('studios', []),
            'sources': anime.get('sources', []),
            'picture': anime.get('picture'),
            'thumbnail': anime.get('thumbnail'),
            'animeSeason': anime.get('animeSeason', {}),
            'status': anime.get('status', 'UNKNOWN'),
            'episodes': anime.get('episodes', 0)
        }
        
        enhanced_data.append(enhanced_anime)
    
    # Save enhanced data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Prepared {len(enhanced_data)} entries for fine-tuning, saved to {output_path}")
    return enhanced_data


def run_fine_tuning(
    data_path: str,
    model_output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    use_gpu: bool = True,
    save_checkpoints: bool = True,
    eval_steps: int = 250
):
    """Run the complete fine-tuning pipeline.
    
    Args:
        data_path: Path to fine-tuning data JSON
        model_output_dir: Directory to save trained models
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_gpu: Whether to use GPU if available
        save_checkpoints: Whether to save model checkpoints
        eval_steps: Steps between evaluations
    """
    logger.info("Starting anime fine-tuning pipeline")
    
    # Setup settings
    settings = get_settings()
    settings.enable_fine_tuning = True
    settings.fine_tuning_num_epochs = num_epochs
    settings.fine_tuning_batch_size = batch_size
    settings.fine_tuning_learning_rate = learning_rate
    settings.fine_tuning_model_dir = model_output_dir
    
    # Check device availability
    if use_gpu and torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        logger.info("Using CPU for training")
    
    # Initialize fine-tuner
    finetuner = AnimeFineTuner(settings)
    
    # Update configuration
    finetuner.config.num_epochs = num_epochs
    finetuner.config.batch_size = batch_size
    finetuner.config.learning_rate = learning_rate
    finetuner.config.model_output_dir = model_output_dir
    finetuner.config.save_model = save_checkpoints
    
    try:
        # Prepare dataset
        logger.info("Preparing dataset...")
        dataset = finetuner.prepare_dataset(data_path)
        
        # Split dataset
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        logger.info(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Train models
        logger.info("Starting multi-task training...")
        training_stats = finetuner.train_multi_task(train_dataset)
        
        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = finetuner.evaluate_models(val_dataset)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = finetuner.evaluate_models(test_dataset)
        
        # Generate training summary
        summary = finetuner.get_training_summary()
        summary['validation_metrics'] = val_metrics
        summary['test_metrics'] = test_metrics
        
        # Save training summary
        summary_path = Path(model_output_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Best model saved to: {finetuner.best_model_path}")
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Print final metrics
        logger.info("Final Results:")
        logger.info(f"  Validation Accuracy: {val_metrics.get('overall_accuracy', 0):.4f}")
        logger.info(f"  Test Accuracy: {test_metrics.get('overall_accuracy', 0):.4f}")
        logger.info(f"  Character Recognition: {test_metrics.get('character_accuracy', 0):.4f}")
        logger.info(f"  Art Style Classification: {test_metrics.get('art_style_accuracy', 0):.4f}")
        logger.info(f"  Genre Understanding: {test_metrics.get('genre_accuracy', 0):.4f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise


def test_fine_tuned_models(model_path: str, test_queries: Optional[list] = None):
    """Test the fine-tuned models with sample queries.
    
    Args:
        model_path: Path to saved models
        test_queries: Optional list of test queries
    """
    logger.info(f"Testing fine-tuned models from {model_path}")
    
    # Setup settings and fine-tuner
    settings = get_settings()
    finetuner = AnimeFineTuner(settings)
    
    # Load fine-tuned models
    finetuner.load_finetuned_models(model_path)
    
    # Default test queries if none provided
    if test_queries is None:
        test_queries = [
            "action anime with superpowers",
            "romantic comedy school anime",
            "dark fantasy with magic",
            "mecha robots fighting",
            "slice of life daily activities"
        ]
    
    logger.info("Testing enhanced embeddings:")
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        
        # Get enhanced embeddings
        embeddings = finetuner.get_enhanced_embeddings(text=query)
        
        for task, embedding in embeddings.items():
            logger.info(f"  {task.capitalize()} embedding shape: {embedding.shape}")
        
        # Test character predictions
        if 'character' in embeddings:
            char_predictions = finetuner.character_finetuner.predict_characters(
                text_embedding=embeddings['character']
            )
            if char_predictions:
                logger.info(f"  Character predictions: {char_predictions[:3]}")
        
        # Test genre predictions
        if 'genre' in embeddings:
            genre_predictions = finetuner.genre_enhancer.predict_genres(
                text_embedding=embeddings['genre']
            )
            if genre_predictions:
                logger.info(f"  Genre predictions: {genre_predictions[:3]}")


def main():
    """Main function for running fine-tuning script."""
    parser = argparse.ArgumentParser(description="Run anime domain-specific fine-tuning")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for fine-tuning')
    prepare_parser.add_argument('--input', required=True, help='Path to anime database JSON')
    prepare_parser.add_argument('--output', required=True, help='Path to save prepared data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run fine-tuning')
    train_parser.add_argument('--data', required=True, help='Path to fine-tuning data JSON')
    train_parser.add_argument('--output', required=True, help='Directory to save models')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    train_parser.add_argument('--no-checkpoints', action='store_true', help='Disable checkpoint saving')
    train_parser.add_argument('--eval-steps', type=int, default=250, help='Steps between evaluations')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test fine-tuned models')
    test_parser.add_argument('--model-path', required=True, help='Path to saved models')
    test_parser.add_argument('--queries', nargs='+', help='Test queries')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_anime_data_for_finetuning(args.input, args.output)
    
    elif args.command == 'train':
        run_fine_tuning(
            data_path=args.data,
            model_output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_gpu=not args.no_gpu,
            save_checkpoints=not args.no_checkpoints,
            eval_steps=args.eval_steps
        )
    
    elif args.command == 'test':
        test_fine_tuned_models(args.model_path, args.queries)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()