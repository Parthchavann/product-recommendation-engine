import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import ndcg_score, average_precision_score, roc_auc_score
import torch

from ..utils.logger import get_logger


logger = get_logger(__name__)


class RecommendationMetrics:
    """Comprehensive evaluation metrics for recommendation systems"""
    
    @staticmethod
    def precision_at_k(predicted: List[int], actual: List[int], k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            predicted: List of predicted item IDs (ranked)
            actual: List of ground truth item IDs
            k: Number of top items to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant = len(set(predicted_k) & set(actual))
        
        return relevant / k
    
    @staticmethod
    def recall_at_k(predicted: List[int], actual: List[int], k: int) -> float:
        """
        Calculate Recall@K
        
        Args:
            predicted: List of predicted item IDs (ranked)
            actual: List of ground truth item IDs
            k: Number of top items to consider
            
        Returns:
            Recall@K score
        """
        if len(actual) == 0:
            return 0.0
        
        predicted_k = predicted[:k]
        relevant = len(set(predicted_k) & set(actual))
        
        return relevant / len(actual)
    
    @staticmethod
    def f1_at_k(predicted: List[int], actual: List[int], k: int) -> float:
        """
        Calculate F1@K
        
        Args:
            predicted: List of predicted item IDs (ranked)
            actual: List of ground truth item IDs
            k: Number of top items to consider
            
        Returns:
            F1@K score
        """
        precision = RecommendationMetrics.precision_at_k(predicted, actual, k)
        recall = RecommendationMetrics.recall_at_k(predicted, actual, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(predicted_scores: np.ndarray, 
                  actual_relevance: np.ndarray, 
                  k: int) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            predicted_scores: Predicted relevance scores
            actual_relevance: True relevance scores (binary or graded)
            k: Number of top items to consider
            
        Returns:
            NDCG@K score
        """
        if len(predicted_scores) == 0 or len(actual_relevance) == 0:
            return 0.0
        
        try:
            return ndcg_score(
                actual_relevance.reshape(1, -1), 
                predicted_scores.reshape(1, -1), 
                k=k
            )
        except:
            return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(predicted: List[int], actual: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            predicted: List of predicted item IDs (ranked)
            actual: List of ground truth item IDs
            
        Returns:
            MRR score
        """
        for i, item in enumerate(predicted):
            if item in actual:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def average_precision(predicted: List[int], actual: List[int]) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            predicted: List of predicted item IDs (ranked)
            actual: List of ground truth item IDs
            
        Returns:
            Average Precision score
        """
        if len(actual) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0
        
        for i, item in enumerate(predicted):
            if item in actual:
                num_hits += 1
                score += num_hits / (i + 1.0)
        
        return score / len(actual)
    
    @staticmethod
    def coverage(recommendations: List[List[int]], catalog_size: int) -> float:
        """
        Calculate catalog coverage (what fraction of items are recommended)
        
        Args:
            recommendations: List of recommendation lists
            catalog_size: Total number of items in catalog
            
        Returns:
            Coverage score (0-1)
        """
        if catalog_size == 0:
            return 0.0
        
        recommended_items = set()
        for rec_list in recommendations:
            recommended_items.update(rec_list)
        
        return len(recommended_items) / catalog_size
    
    @staticmethod
    def diversity(recommendations: List[List[int]], 
                  item_features: Dict[int, np.ndarray]) -> float:
        """
        Calculate intra-list diversity (average pairwise distance within recommendations)
        
        Args:
            recommendations: List of recommendation lists
            item_features: Dictionary mapping item_id to feature vector
            
        Returns:
            Average diversity score
        """
        diversities = []
        
        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue
            
            distances = []
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item_i = rec_list[i]
                    item_j = rec_list[j]
                    
                    if item_i in item_features and item_j in item_features:
                        feat_i = item_features[item_i]
                        feat_j = item_features[item_j]
                        
                        # Cosine distance
                        similarity = np.dot(feat_i, feat_j) / (
                            np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8
                        )
                        distance = 1 - similarity
                        distances.append(distance)
            
            if distances:
                diversities.append(np.mean(distances))
        
        return np.mean(diversities) if diversities else 0.0
    
    @staticmethod
    def novelty(recommendations: List[List[int]], 
                item_popularity: Dict[int, float]) -> float:
        """
        Calculate novelty (preference for less popular items)
        
        Args:
            recommendations: List of recommendation lists
            item_popularity: Dictionary mapping item_id to popularity score
            
        Returns:
            Average novelty score
        """
        novelty_scores = []
        
        for rec_list in recommendations:
            if not rec_list:
                continue
            
            rec_novelties = []
            for item_id in rec_list:
                if item_id in item_popularity:
                    # Novelty = -log(popularity)
                    popularity = item_popularity[item_id]
                    novelty = -np.log(popularity + 1e-8)
                    rec_novelties.append(novelty)
            
            if rec_novelties:
                novelty_scores.append(np.mean(rec_novelties))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    @staticmethod
    def hit_rate(predicted: List[int], actual: List[int], k: int) -> float:
        """
        Calculate Hit Rate@K (binary measure of whether any relevant item is in top-k)
        
        Args:
            predicted: List of predicted item IDs (ranked)
            actual: List of ground truth item IDs
            k: Number of top items to consider
            
        Returns:
            Hit rate (0 or 1)
        """
        predicted_k = predicted[:k]
        return 1.0 if any(item in actual for item in predicted_k) else 0.0


class RankingMetrics:
    """Metrics for ranking/re-ranking evaluation"""
    
    @staticmethod
    def spearman_correlation(predicted_ranks: List[int], 
                           actual_ranks: List[int]) -> float:
        """
        Calculate Spearman rank correlation
        
        Args:
            predicted_ranks: Predicted item ranks
            actual_ranks: Actual item ranks
            
        Returns:
            Spearman correlation coefficient
        """
        from scipy.stats import spearmanr
        
        if len(predicted_ranks) != len(actual_ranks) or len(predicted_ranks) < 2:
            return 0.0
        
        correlation, _ = spearmanr(predicted_ranks, actual_ranks)
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def kendall_tau(predicted_ranks: List[int], 
                   actual_ranks: List[int]) -> float:
        """
        Calculate Kendall's Tau rank correlation
        
        Args:
            predicted_ranks: Predicted item ranks
            actual_ranks: Actual item ranks
            
        Returns:
            Kendall's Tau coefficient
        """
        from scipy.stats import kendalltau
        
        if len(predicted_ranks) != len(actual_ranks) or len(predicted_ranks) < 2:
            return 0.0
        
        correlation, _ = kendalltau(predicted_ranks, actual_ranks)
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at k
        
        Args:
            relevance_scores: List of relevance scores in ranked order
            k: Number of positions to consider
            
        Returns:
            DCG@k score
        """
        relevance_scores = relevance_scores[:k]
        
        if not relevance_scores:
            return 0.0
        
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 2)
        
        return dcg


class EvaluationSuite:
    """Complete evaluation suite for recommendation systems"""
    
    def __init__(self):
        self.metrics = RecommendationMetrics()
        self.ranking_metrics = RankingMetrics()
        self.logger = get_logger(__name__)
    
    def evaluate_recommendations(self,
                                predictions: List[List[int]],
                                ground_truth: List[List[int]],
                                k_values: List[int] = [5, 10, 20],
                                item_features: Optional[Dict[int, np.ndarray]] = None,
                                item_popularity: Optional[Dict[int, float]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of recommendations
        
        Args:
            predictions: List of predicted item lists for each user
            ground_truth: List of ground truth item lists for each user
            k_values: List of k values to evaluate
            item_features: Item features for diversity calculation
            item_popularity: Item popularity for novelty calculation
            
        Returns:
            Dictionary of metric scores
        """
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Number of predictions and ground truth must match")
        
        results = {}
        
        self.logger.info(f"Evaluating {len(predictions)} users with k_values={k_values}")
        
        # Calculate metrics for each k
        for k in k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            hit_rates = []
            mrr_scores = []
            ap_scores = []
            
            for pred, actual in zip(predictions, ground_truth):
                precision_scores.append(self.metrics.precision_at_k(pred, actual, k))
                recall_scores.append(self.metrics.recall_at_k(pred, actual, k))
                f1_scores.append(self.metrics.f1_at_k(pred, actual, k))
                hit_rates.append(self.metrics.hit_rate(pred, actual, k))
                mrr_scores.append(self.metrics.mean_reciprocal_rank(pred[:k], actual))
                ap_scores.append(self.metrics.average_precision(pred[:k], actual))
            
            # Average across users
            results[f'precision@{k}'] = np.mean(precision_scores)
            results[f'recall@{k}'] = np.mean(recall_scores)
            results[f'f1@{k}'] = np.mean(f1_scores)
            results[f'hit_rate@{k}'] = np.mean(hit_rates)
            results[f'mrr@{k}'] = np.mean(mrr_scores)
            results[f'map@{k}'] = np.mean(ap_scores)  # Mean Average Precision
        
        # Overall MRR (using full predictions)
        overall_mrr = []
        for pred, actual in zip(predictions, ground_truth):
            overall_mrr.append(self.metrics.mean_reciprocal_rank(pred, actual))
        results['mrr'] = np.mean(overall_mrr)
        
        # Coverage (catalog coverage)
        if item_features:
            catalog_size = len(item_features)
            results['coverage'] = self.metrics.coverage(predictions, catalog_size)
            
            # Diversity
            results['diversity'] = self.metrics.diversity(predictions, item_features)
        
        # Novelty
        if item_popularity:
            results['novelty'] = self.metrics.novelty(predictions, item_popularity)
        
        # Log results
        self.logger.info("Evaluation Results:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def evaluate_ranking(self,
                        predicted_scores: List[np.ndarray],
                        actual_scores: List[np.ndarray],
                        k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate ranking quality using NDCG and other ranking metrics
        
        Args:
            predicted_scores: List of predicted scores for each user
            actual_scores: List of actual relevance scores for each user
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary of ranking metric scores
        """
        
        results = {}
        
        # Calculate NDCG for each k
        for k in k_values:
            ndcg_scores = []
            
            for pred_scores, actual_scores_user in zip(predicted_scores, actual_scores):
                if len(pred_scores) > 0 and len(actual_scores_user) > 0:
                    ndcg = self.metrics.ndcg_at_k(pred_scores, actual_scores_user, k)
                    ndcg_scores.append(ndcg)
            
            results[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        return results
    
    def compare_models(self,
                      model_predictions: Dict[str, List[List[int]]],
                      ground_truth: List[List[int]],
                      k: int = 10) -> pd.DataFrame:
        """
        Compare multiple models side by side
        
        Args:
            model_predictions: Dictionary mapping model names to predictions
            ground_truth: Ground truth recommendations
            k: K value for evaluation
            
        Returns:
            DataFrame with comparison results
        """
        
        results = []
        
        for model_name, predictions in model_predictions.items():
            model_results = self.evaluate_recommendations(
                predictions, ground_truth, k_values=[k]
            )
            
            row = {'model': model_name}
            row.update(model_results)
            results.append(row)
        
        return pd.DataFrame(results)
    
    def statistical_significance_test(self,
                                    model1_predictions: List[List[int]],
                                    model2_predictions: List[List[int]],
                                    ground_truth: List[List[int]],
                                    metric: str = 'precision',
                                    k: int = 10) -> Dict[str, float]:
        """
        Test statistical significance between two models
        
        Args:
            model1_predictions: Predictions from model 1
            model2_predictions: Predictions from model 2
            ground_truth: Ground truth recommendations
            metric: Metric to test ('precision', 'recall', 'f1')
            k: K value for evaluation
            
        Returns:
            Statistical test results
        """
        
        from scipy.stats import ttest_rel
        
        # Calculate metric for both models
        if metric == 'precision':
            metric_func = self.metrics.precision_at_k
        elif metric == 'recall':
            metric_func = self.metrics.recall_at_k
        elif metric == 'f1':
            metric_func = self.metrics.f1_at_k
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        model1_scores = []
        model2_scores = []
        
        for pred1, pred2, actual in zip(model1_predictions, model2_predictions, ground_truth):
            model1_scores.append(metric_func(pred1, actual, k))
            model2_scores.append(metric_func(pred2, actual, k))
        
        # Paired t-test
        t_stat, p_value = ttest_rel(model1_scores, model2_scores)
        
        return {
            'metric': f'{metric}@{k}',
            'model1_mean': np.mean(model1_scores),
            'model2_mean': np.mean(model2_scores),
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'effect_size': (np.mean(model2_scores) - np.mean(model1_scores)) / np.std(model1_scores)
        }