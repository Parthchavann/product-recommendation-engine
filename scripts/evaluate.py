#!/usr/bin/env python3
"""
Evaluation script for the recommendation engine

Usage:
    python scripts/evaluate.py --model-dir models --test-data data/processed/test.csv
    python scripts/evaluate.py --compare model1:models/v1 model2:models/v2
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pickle
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config, load_config
from src.utils.logger import get_logger
from src.data.data_loader import RecommendationDataset
from src.models.two_tower import TwoTowerModel
from src.retrieval.faiss_index import FAISSIndex, CandidateGenerator
from src.models.lightgbm_ranker import CTRRanker
from src.evaluation.metrics import EvaluationSuite
from src.evaluation.ab_testing import ABTestManager, simulate_ab_test_data


logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description="Evaluate recommendation model")
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing trained models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset CSV file"
    )
    
    parser.add_argument(
        "--compare",
        nargs='+',
        help="Compare multiple models (format: name:path)"
    )
    
    parser.add_argument(
        "--k-values",
        nargs='+',
        type=int,
        default=[5, 10, 20],
        help="K values for evaluation metrics"
    )
    
    parser.add_argument(
        "--num-test-users",
        type=int,
        default=1000,
        help="Number of test users to evaluate"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--run-ab-test-demo",
        action="store_true",
        help="Run A/B testing demonstration"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu)"
    )
    
    return parser.parse_args()


class ModelEvaluator:
    """Evaluate trained recommendation models"""
    
    def __init__(self, model_dir: Path, config: Config, device: str = "cuda"):
        self.model_dir = model_dir
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Model components
        self.two_tower_model = None
        self.faiss_index = None
        self.ctr_ranker = None
        self.candidate_generator = None
        
        # Data mappings
        self.user_encoder = None
        self.item_encoder = None
        
        # Evaluation suite
        self.eval_suite = EvaluationSuite()
        
        logger.info(f"Initialized ModelEvaluator on device: {self.device}")
    
    def load_models(self):
        """Load all trained models"""
        
        logger.info("Loading trained models...")
        
        # Load two-tower model
        model_path = self.model_dir / "two_tower.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model parameters from checkpoint
            num_users = checkpoint['num_users']
            num_items = checkpoint['num_items']
            config = checkpoint.get('config', self.config)
            
            # Create and load model
            self.two_tower_model = TwoTowerModel(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=config.model.embedding_dim,
                tower_dims=config.model.tower_dims,
                dropout=0.0  # No dropout for evaluation
            )
            
            self.two_tower_model.load_state_dict(checkpoint['model_state_dict'])
            self.two_tower_model.to(self.device)
            self.two_tower_model.eval()
            
            # Load encoders
            self.user_encoder = checkpoint.get('user_encoder')
            self.item_encoder = checkpoint.get('item_encoder')
            
            logger.info("Two-tower model loaded successfully")
        else:
            logger.warning(f"Two-tower model not found at {model_path}")
        
        # Load FAISS index
        index_path = self.model_dir / "faiss_index"
        if (index_path.with_suffix('.index')).exists():
            # Load metadata to get index configuration
            with open(f"{index_path}.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            self.faiss_index = FAISSIndex(
                dimension=metadata['dimension'],
                index_type=metadata['index_type'],
                metric=metadata['metric']
            )
            
            self.faiss_index.load(str(index_path))
            self.candidate_generator = CandidateGenerator(self.faiss_index)
            
            logger.info(f"FAISS index loaded: {self.faiss_index.index.ntotal} vectors")
        else:
            logger.warning(f"FAISS index not found at {index_path}")
        
        # Load CTR ranker
        ctr_path = self.model_dir / "ctr_ranker.txt"
        if ctr_path.exists():
            self.ctr_ranker = CTRRanker()
            self.ctr_ranker.load_model(str(ctr_path))
            
            logger.info("CTR ranker loaded successfully")
        else:
            logger.warning(f"CTR ranker not found at {ctr_path}")
    
    def generate_recommendations(self, 
                               user_ids: List[int], 
                               k: int = 10,
                               use_reranking: bool = True) -> Dict[int, List[int]]:
        """Generate recommendations for users"""
        
        if not self.two_tower_model or not self.candidate_generator:
            raise ValueError("Models not loaded properly")
        
        recommendations = {}
        
        logger.info(f"Generating recommendations for {len(user_ids)} users (k={k})")
        
        for user_id in user_ids:
            try:
                # Get user embedding
                user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    user_embedding = self.two_tower_model.get_user_embedding(user_tensor)
                    user_embedding_np = user_embedding.cpu().numpy()
                
                # Get candidates from FAISS
                candidates = self.candidate_generator.get_candidates(
                    user_embedding_np,
                    k=k * 3  # Over-fetch for re-ranking
                )
                
                # Extract item IDs
                candidate_items = [c['item_id'] for c in candidates]
                
                # Re-rank if CTR model is available and requested
                if use_reranking and self.ctr_ranker:
                    # Simplified re-ranking (would need proper feature preparation)
                    # For demo, just take top k from candidates
                    final_items = candidate_items[:k]
                else:
                    final_items = candidate_items[:k]
                
                recommendations[user_id] = final_items
                
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for user {user_id}: {e}")
                recommendations[user_id] = []
        
        return recommendations
    
    def load_test_data(self, test_data_path: str) -> Dict[int, List[int]]:
        """Load test data ground truth"""
        
        if not test_data_path or not Path(test_data_path).exists():
            logger.warning("Test data not found, generating synthetic ground truth")
            return self._generate_synthetic_ground_truth()
        
        # Load CSV data
        test_df = pd.read_csv(test_data_path)
        
        # Convert to ground truth format
        ground_truth = {}
        
        for user_id in test_df['user_id'].unique():
            user_items = test_df[
                (test_df['user_id'] == user_id) & 
                (test_df['rating'] >= 4.0)  # Positive interactions
            ]['item_id'].tolist()
            
            ground_truth[user_id] = user_items
        
        logger.info(f"Loaded ground truth for {len(ground_truth)} users")
        
        return ground_truth
    
    def _generate_synthetic_ground_truth(self) -> Dict[int, List[int]]:
        """Generate synthetic ground truth for demonstration"""
        
        logger.info("Generating synthetic ground truth data")
        
        ground_truth = {}
        
        # Generate for first 1000 users
        for user_id in range(1, 1001):
            # Each user likes 5-15 random items
            num_items = np.random.randint(5, 16)
            liked_items = np.random.choice(1000, size=num_items, replace=False).tolist()
            ground_truth[user_id] = liked_items
        
        return ground_truth
    
    def evaluate_model(self,
                      test_user_ids: List[int],
                      ground_truth: Dict[int, List[int]],
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        logger.info(f"Evaluating model on {len(test_user_ids)} users")
        
        results = {}
        
        for k in k_values:
            logger.info(f"Evaluating at k={k}")
            
            # Generate recommendations
            recommendations = self.generate_recommendations(test_user_ids, k=k)
            
            # Prepare data for evaluation
            predictions = []
            actual = []
            
            for user_id in test_user_ids:
                if user_id in recommendations and user_id in ground_truth:
                    predictions.append(recommendations[user_id])
                    actual.append(ground_truth[user_id])
            
            if not predictions:
                logger.warning(f"No valid predictions for k={k}")
                continue
            
            # Calculate metrics
            metrics = self.eval_suite.evaluate_recommendations(
                predictions=predictions,
                ground_truth=actual,
                k_values=[k]
            )
            
            # Store results
            for metric_name, value in metrics.items():
                results[metric_name] = value
        
        return results
    
    def compare_models(self, 
                      model_configs: List[Dict[str, str]],
                      test_user_ids: List[int],
                      ground_truth: Dict[int, List[int]]) -> pd.DataFrame:
        """Compare multiple models"""
        
        logger.info(f"Comparing {len(model_configs)} models")
        
        comparison_results = {}
        
        for model_config in model_configs:
            model_name = model_config['name']
            model_path = Path(model_config['path'])
            
            logger.info(f"Evaluating model: {model_name}")
            
            # Temporarily switch to this model's directory
            original_model_dir = self.model_dir
            self.model_dir = model_path
            
            try:
                # Load this model
                self.load_models()
                
                # Evaluate
                results = self.evaluate_model(test_user_ids, ground_truth, k_values=[10])
                comparison_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
            
            # Restore original model directory
            self.model_dir = original_model_dir
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        return comparison_df


def run_ab_test_demo():
    """Run A/B testing demonstration"""
    
    logger.info("Running A/B testing demonstration...")
    
    # Initialize A/B test manager
    ab_manager = ABTestManager(min_sample_size=500)
    
    # Create experiment
    experiment_id = "new_ranking_algorithm_v1"
    
    experiment = ab_manager.create_experiment(
        experiment_id=experiment_id,
        name="New Ranking Algorithm",
        description="Test improved CTR re-ranking with additional features",
        treatment_percentage=0.5,
        target_metrics=['ctr', 'conversion_rate', 'revenue_per_user'],
        start_date=datetime.now() - timedelta(days=14),
        end_date=datetime.now()
    )
    
    # Start experiment
    ab_manager.start_experiment(experiment_id)
    
    # Simulate data
    simulation_data = simulate_ab_test_data(
        ab_manager,
        experiment_id,
        num_users=5000,
        days=14
    )
    
    # Analyze results
    results = ab_manager.analyze_experiment(experiment_id)
    
    # Print results
    logger.info("=== A/B Test Results ===")
    logger.info(f"Experiment: {results['name']}")
    logger.info(f"Status: {results['status']}")
    logger.info(f"Sample sizes - Control: {results['sample_sizes']['control']}, Treatment: {results['sample_sizes']['treatment']}")
    
    logger.info("\n=== Metrics ===")
    for metric_name, metric_data in results['metrics'].items():
        logger.info(f"{metric_name}:")
        logger.info(f"  Control: {metric_data['control']:.4f}")
        logger.info(f"  Treatment: {metric_data['treatment']:.4f}")
        logger.info(f"  Lift: {metric_data['lift_percentage']:.2f}%")
    
    logger.info("\n=== Statistical Tests ===")
    for test_name, test_data in results['statistical_tests'].items():
        logger.info(f"{test_name}:")
        logger.info(f"  P-value: {test_data['p_value']:.4f}")
        logger.info(f"  Significant: {test_data['is_significant']}")
    
    logger.info(f"\n=== Recommendation ===")
    logger.info(f"Action: {results['recommendations']['action']}")
    logger.info(f"Confidence: {results['recommendations']['confidence']}")
    logger.info(f"Reason: {results['recommendations']['reason']}")
    
    return results


def main():
    """Main evaluation pipeline"""
    
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run A/B test demo if requested
    if args.run_ab_test_demo:
        from datetime import datetime, timedelta
        ab_results = run_ab_test_demo()
        
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(ab_results, f, indent=2, default=str)
            logger.info(f"A/B test results saved to {args.output_file}")
        
        return
    
    # Initialize evaluator
    model_dir = Path(args.model_dir)
    evaluator = ModelEvaluator(model_dir, config, device=args.device)
    
    try:
        # Load models
        evaluator.load_models()
        
        # Load test data
        ground_truth = evaluator.load_test_data(args.test_data)
        
        # Get test user IDs
        test_user_ids = list(ground_truth.keys())[:args.num_test_users]
        
        if args.compare:
            # Model comparison mode
            model_configs = []
            for model_spec in args.compare:
                name, path = model_spec.split(':')
                model_configs.append({'name': name, 'path': path})
            
            comparison_df = evaluator.compare_models(model_configs, test_user_ids, ground_truth)
            
            logger.info("=== Model Comparison Results ===")
            print(comparison_df)
            
            if args.output_file:
                comparison_df.to_json(args.output_file, indent=2)
                logger.info(f"Comparison results saved to {args.output_file}")
        
        else:
            # Single model evaluation
            results = evaluator.evaluate_model(test_user_ids, ground_truth, args.k_values)
            
            logger.info("=== Evaluation Results ===")
            for metric, value in results.items():
                logger.info(f"{metric}: {value:.4f}")
            
            if args.output_file:
                import json
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {args.output_file}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()