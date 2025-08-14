#!/usr/bin/env python3
"""
Training script for the recommendation engine

Usage:
    python scripts/train.py --config configs/model_config.yaml
    python scripts/train.py --dataset movielens-25m --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from src.utils.config import Config, load_config, get_project_root
from src.utils.logger import get_logger, ExperimentTracker
from src.data.data_loader import prepare_data, create_data_loaders
from src.models.two_tower import TwoTowerModel
from src.models.training import TwoTowerTrainer, create_model
from src.models.embeddings import create_embeddings_cache
from src.retrieval.faiss_index import FAISSIndex
from src.models.lightgbm_ranker import CTRRanker, create_user_item_features


logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description="Train recommendation model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens-25m",
        help="Dataset to use (movielens-25m, movielens-20m, movielens-1m)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip content embeddings generation"
    )
    
    parser.add_argument(
        "--skip-faiss",
        action="store_true", 
        help="Skip FAISS index building"
    )
    
    parser.add_argument(
        "--skip-ctr",
        action="store_true",
        help="Skip CTR model training"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project name"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for logging"
    )
    
    return parser.parse_args()


def train_two_tower_model(config: Config, 
                         train_dataset, 
                         val_dataset, 
                         test_dataset,
                         output_dir: Path) -> TwoTowerModel:
    """Train the two-tower model"""
    
    logger.info("Training two-tower model...")
    
    # Create model
    num_users = len(train_dataset.user_encoder.classes_)
    num_items = len(train_dataset.item_encoder.classes_)
    
    model = create_model(config, num_users, num_items)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Create trainer
    trainer = TwoTowerTrainer(model, config, device=config.device)
    
    # Train model
    history = trainer.train(train_loader, val_loader, test_loader)
    
    # Save final model
    model_path = output_dir / "two_tower.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'num_users': num_users,
        'num_items': num_items,
        'user_encoder': train_dataset.user_encoder,
        'item_encoder': train_dataset.item_encoder,
        'history': history
    }, model_path)
    
    logger.info(f"Saved two-tower model to {model_path}")
    
    return model


def build_faiss_index(model: TwoTowerModel,
                     train_dataset,
                     config: Config,
                     output_dir: Path):
    """Build FAISS index for fast similarity search"""
    
    logger.info("Building FAISS index...")
    
    # Generate item embeddings
    model.eval()
    with torch.no_grad():
        all_item_embeddings = model.get_all_item_embeddings()
    
    # Get item IDs
    item_ids = list(range(len(train_dataset.item_encoder.classes_)))
    
    # Create FAISS index
    faiss_index = FAISSIndex(
        dimension=all_item_embeddings.shape[1],
        index_type=config.faiss.index_type,
        metric=config.faiss.metric,
        nlist=config.faiss.nlist
    )
    
    # Build index
    faiss_index.build_index(
        all_item_embeddings.numpy(), 
        item_ids
    )
    
    # Save index
    index_path = output_dir / "faiss_index"
    faiss_index.save(str(index_path))
    
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Benchmark search performance
    logger.info("Benchmarking FAISS index...")
    
    # Create random query vectors
    num_queries = 1000
    query_vectors = torch.randn(num_queries, all_item_embeddings.shape[1]).numpy()
    
    benchmark_results = faiss_index.benchmark_search(query_vectors)
    
    for config_name, metrics in benchmark_results.items():
        logger.info(f"  {config_name}: {metrics['latency_per_query_ms']:.2f}ms/query, {metrics['qps']:.1f} QPS")


def train_ctr_model(train_dataset, val_dataset, config: Config, output_dir: Path):
    """Train CTR re-ranking model"""
    
    logger.info("Training CTR model...")
    
    # Convert datasets to DataFrames
    train_interactions = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        train_interactions.append({
            'user_id': sample['user_id'].item(),
            'item_id': sample['item_id'].item(),
            'rating': sample['rating'].item()
        })
    
    train_df = pd.DataFrame(train_interactions)
    
    val_interactions = []
    for i in range(len(val_dataset)):
        sample = val_dataset[i] 
        val_interactions.append({
            'user_id': sample['user_id'].item(),
            'item_id': sample['item_id'].item(), 
            'rating': sample['rating'].item()
        })
    
    val_df = pd.DataFrame(val_interactions)
    
    # Create dummy user and item data
    unique_users = train_df['user_id'].unique()
    unique_items = train_df['item_id'].unique()
    
    users_df = pd.DataFrame({
        'user_id': unique_users,
        'age': np.random.randint(18, 65, len(unique_users)),
        'gender': np.random.choice(['M', 'F'], len(unique_users))
    })
    
    items_df = pd.DataFrame({
        'item_id': unique_items,
        'title': [f'Item {i}' for i in unique_items],
        'genres': [np.random.choice(['Action', 'Comedy', 'Drama'], size=2) for _ in unique_items]
    })
    
    # Create user and item features
    user_histories, item_stats = create_user_item_features(train_df, users_df, items_df)
    
    # Create CTR ranker
    ctr_ranker = CTRRanker(config)
    
    # Prepare features
    train_features = ctr_ranker.prepare_features(
        train_df, users_df, items_df, user_histories, item_stats
    )
    
    val_features = ctr_ranker.prepare_features(
        val_df, users_df, items_df, user_histories, item_stats
    )
    
    # Train model
    metrics = ctr_ranker.train(train_features, val_features)
    
    logger.info(f"CTR model training completed: {metrics}")
    
    # Save model
    model_path = output_dir / "ctr_ranker.txt"
    ctr_ranker.save_model(str(model_path))
    
    logger.info(f"Saved CTR model to {model_path}")
    
    # Log feature importance
    importance = ctr_ranker.get_feature_importance()
    logger.info("Top 10 important features:")
    for feature, score in list(importance.items())[:10]:
        logger.info(f"  {feature}: {score}")


def main():
    """Main training pipeline"""
    
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config.data.dataset = args.dataset
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize experiment tracker
    experiment_name = args.experiment_name or f"training_{config.data.dataset}"
    tracker = ExperimentTracker(experiment_name)
    
    # Log configuration
    config_dict = {
        'dataset': config.data.dataset,
        'model': {
            'embedding_dim': config.model.embedding_dim,
            'tower_dims': config.model.tower_dims,
            'dropout': config.model.dropout
        },
        'training': {
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate
        },
        'device': config.device
    }
    tracker.log_config(config_dict)
    
    try:
        # Step 1: Prepare data
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, test_dataset, user_encoder, item_encoder = prepare_data(config)
        
        tracker.log_metrics(0, {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'num_users': len(user_encoder.classes_),
            'num_items': len(item_encoder.classes_)
        })
        
        # Save encoders
        encoders_path = output_dir / "encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump({
                'user_encoder': user_encoder,
                'item_encoder': item_encoder
            }, f)
        
        # Step 2: Generate content embeddings (if not skipped)
        if not args.skip_embeddings:
            logger.info("Generating content embeddings...")
            # This would require the actual movie/user data loaded in prepare_data
            # For now, we'll skip this step in the demo
            logger.info("Content embeddings generation skipped in demo")
        
        # Step 3: Train two-tower model
        logger.info("Training two-tower model...")
        model = train_two_tower_model(config, train_dataset, val_dataset, test_dataset, output_dir)
        
        # Step 4: Build FAISS index (if not skipped)
        if not args.skip_faiss:
            build_faiss_index(model, train_dataset, config, output_dir)
        
        # Step 5: Train CTR model (if not skipped)
        if not args.skip_ctr:
            import pandas as pd
            import numpy as np
            train_ctr_model(train_dataset, val_dataset, config, output_dir)
        
        # Final metrics
        final_metrics = {
            'training_completed': True,
            'output_dir': str(output_dir),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
        
        tracker.log_completion(final_metrics)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Models saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
    
    main()