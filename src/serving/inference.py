import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import asyncio
from pathlib import Path
import pickle
from datetime import datetime, timedelta

from ..models.two_tower import TwoTowerModel
from ..models.lightgbm_ranker import CTRRanker
from ..retrieval.faiss_index import FAISSIndex, CandidateGenerator
from ..models.embeddings import ContentEmbedder
from .cache import CacheManager
from ..utils.logger import get_logger


logger = get_logger(__name__)


class RecommendationInference:
    """High-level inference engine for recommendations"""
    
    def __init__(self, 
                 model_config: Dict[str, str],
                 cache_manager: CacheManager,
                 device: str = 'cuda'):
        
        self.model_config = model_config
        self.cache_manager = cache_manager
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Model components
        self.two_tower_model = None
        self.faiss_index = None
        self.ctr_ranker = None
        self.content_embedder = None
        self.candidate_generator = None
        
        # Data mappings
        self.user_encoder = None
        self.item_encoder = None
        self.user_features = {}
        self.item_features = {}
        self.item_metadata = {}
        
        # Status
        self.models_loaded = False
        self.loaded_at = None
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0,
            'candidate_generation_ms': 0,
            'reranking_ms': 0
        }
        
        logger.info(f"Initialized RecommendationInference on device: {self.device}")
    
    async def load_models(self):
        """Load all models and data"""
        
        logger.info("Loading recommendation models...")
        
        try:
            # Load two-tower model
            await self._load_two_tower_model()
            
            # Load FAISS index
            await self._load_faiss_index()
            
            # Load CTR ranker
            await self._load_ctr_ranker()
            
            # Load content embedder
            await self._load_content_embedder()
            
            # Load data mappings
            await self._load_data_mappings()
            
            # Initialize candidate generator
            if self.faiss_index:
                self.candidate_generator = CandidateGenerator(self.faiss_index)
            
            self.models_loaded = True
            self.loaded_at = datetime.now()
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
            raise
    
    async def _load_two_tower_model(self):
        """Load two-tower model"""
        
        model_path = self.model_config.get('two_tower_path', 'models/two_tower.pt')
        
        if not Path(model_path).exists():
            logger.warning(f"Two-tower model not found at {model_path}")
            return
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # Load from training checkpoint
                model_config = checkpoint.get('config')
                
                self.two_tower_model = TwoTowerModel(
                    num_users=checkpoint.get('num_users', 10000),
                    num_items=checkpoint.get('num_items', 10000),
                    embedding_dim=model_config.model.embedding_dim if model_config else 128,
                    tower_dims=model_config.model.tower_dims if model_config else [256, 128, 64],
                    dropout=0.0  # No dropout during inference
                )
                
                self.two_tower_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Load complete model
                self.two_tower_model = checkpoint
            
            self.two_tower_model.to(self.device)
            self.two_tower_model.eval()
            
            logger.info("Two-tower model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading two-tower model: {e}")
            self.two_tower_model = None
    
    async def _load_faiss_index(self):
        """Load FAISS index"""
        
        index_path = self.model_config.get('faiss_index_path', 'models/faiss_index')
        
        if not Path(f"{index_path}.index").exists():
            logger.warning(f"FAISS index not found at {index_path}")
            return
        
        try:
            # Load index configuration from metadata
            with open(f"{index_path}.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            self.faiss_index = FAISSIndex(
                dimension=metadata['dimension'],
                index_type=metadata['index_type'],
                metric=metadata['metric'],
                nlist=metadata.get('nlist', 100),
                nprobe=metadata.get('nprobe', 10)
            )
            
            self.faiss_index.load(index_path)
            
            logger.info(f"FAISS index loaded: {self.faiss_index.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.faiss_index = None
    
    async def _load_ctr_ranker(self):
        """Load CTR ranking model"""
        
        model_path = self.model_config.get('ctr_model_path', 'models/ctr_ranker.txt')
        
        if not Path(model_path).exists():
            logger.warning(f"CTR model not found at {model_path}")
            return
        
        try:
            self.ctr_ranker = CTRRanker()
            self.ctr_ranker.load_model(model_path)
            
            logger.info("CTR ranker loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading CTR ranker: {e}")
            self.ctr_ranker = None
    
    async def _load_content_embedder(self):
        """Load content embedder"""
        
        model_name = self.model_config.get('embedder_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        try:
            self.content_embedder = ContentEmbedder(
                model_name=model_name,
                device=self.device
            )
            
            logger.info(f"Content embedder loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading content embedder: {e}")
            self.content_embedder = None
    
    async def _load_data_mappings(self):
        """Load user/item encoders and feature mappings"""
        
        try:
            # Load encoders (if available)
            encoders_path = Path('models/encoders.pkl')
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    encoders = pickle.load(f)
                
                self.user_encoder = encoders.get('user_encoder')
                self.item_encoder = encoders.get('item_encoder')
            
            # Load feature mappings (if available)
            features_path = Path('models/features.pkl')
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    features = pickle.load(f)
                
                self.user_features = features.get('user_features', {})
                self.item_features = features.get('item_features', {})
            
            # Load item metadata (if available)
            metadata_path = Path('models/item_metadata.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.item_metadata = pickle.load(f)
            
            logger.info("Data mappings loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data mappings: {e}")
    
    async def get_recommendations(self,
                                user_id: int,
                                num_recommendations: int = 10,
                                filter_categories: Optional[List[str]] = None,
                                exclude_items: Optional[List[int]] = None,
                                context: Optional[Dict[str, Any]] = None,
                                use_reranking: bool = True,
                                explain: bool = False) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User ID
            num_recommendations: Number of recommendations
            filter_categories: Categories to filter by
            exclude_items: Items to exclude
            context: Contextual information
            use_reranking: Whether to use CTR re-ranking
            explain: Include explanations
            
        Returns:
            List of recommendation items
        """
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"rec:user:{user_id}:k:{num_recommendations}"
        if filter_categories:
            cache_key += f":cats:{'|'.join(sorted(filter_categories))}"
        if exclude_items:
            cache_key += f":exclude:{'|'.join(map(str, sorted(exclude_items)))}"
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self.stats['cache_hits'] += 1
            cached_result['_cached'] = True
            return cached_result.get('recommendations', [])
        
        try:
            # Step 1: Generate candidates
            candidates_start = datetime.now()
            candidates = await self._generate_candidates(
                user_id, 
                num_recommendations * 3,  # Over-fetch for re-ranking
                exclude_items or []
            )
            candidates_time = (datetime.now() - candidates_start).total_seconds() * 1000
            
            # Step 2: Apply filters
            if filter_categories:
                candidates = self._filter_by_categories(candidates, filter_categories)
            
            # Step 3: Re-rank with CTR model
            rerank_start = datetime.now()
            if use_reranking and self.ctr_ranker and candidates:
                candidates = await self._rerank_candidates(
                    candidates, user_id, context or {}
                )
            rerank_time = (datetime.now() - rerank_start).total_seconds() * 1000
            
            # Step 4: Format response
            recommendations = await self._format_recommendations(
                candidates[:num_recommendations],
                explain=explain
            )
            
            # Update stats
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats['total_requests'] += 1
            self.stats['avg_latency_ms'] = (
                (self.stats['avg_latency_ms'] * (self.stats['total_requests'] - 1) + total_time) /
                self.stats['total_requests']
            )
            self.stats['candidate_generation_ms'] = candidates_time
            self.stats['reranking_ms'] = rerank_time
            
            # Cache result
            cache_data = {
                'recommendations': recommendations,
                'generated_at': start_time.isoformat(),
                'latency_ms': total_time
            }
            self.cache_manager.set(cache_key, cache_data, ttl=3600)  # 1 hour TTL
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id} in {total_time:.2f}ms")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Return fallback recommendations
            return await self._get_fallback_recommendations(num_recommendations)
    
    async def _generate_candidates(self,
                                 user_id: int,
                                 num_candidates: int,
                                 exclude_items: List[int]) -> List[Dict[str, Any]]:
        """Generate candidate items for a user"""
        
        if not self.candidate_generator or not self.two_tower_model:
            # Fallback to popular items
            return await self._get_popular_items(num_candidates, exclude_items)
        
        try:
            # Get user embedding
            user_embedding = await self._get_user_embedding(user_id)
            
            if user_embedding is None:
                # New user - use average embedding or popular items
                return await self._get_popular_items(num_candidates, exclude_items)
            
            # Get candidates from FAISS
            candidates = self.candidate_generator.get_candidates(
                user_embedding,
                k=num_candidates,
                filters={'exclude_items': exclude_items}
            )
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return await self._get_popular_items(num_candidates, exclude_items)
    
    async def _get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding vector"""
        
        # Check cache first
        cached_embedding = self.cache_manager.get(f"user_emb:{user_id}")
        if cached_embedding is not None:
            return np.array(cached_embedding)
        
        if not self.two_tower_model:
            return None
        
        try:
            # Encode user ID if we have the encoder
            if self.user_encoder and user_id in self.user_encoder.classes_:
                encoded_user_id = self.user_encoder.transform([user_id])[0]
            else:
                # Use user_id directly (assuming it's in the right range)
                encoded_user_id = user_id
            
            # Get user features
            user_features_vec = self.user_features.get(user_id)
            if user_features_vec is not None:
                user_features_tensor = torch.tensor(user_features_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                user_features_tensor = None
            
            # Generate embedding
            with torch.no_grad():
                user_id_tensor = torch.tensor([encoded_user_id], dtype=torch.long).to(self.device)
                user_embedding = self.two_tower_model.get_user_embedding(user_id_tensor, user_features_tensor)
                user_embedding_np = user_embedding.cpu().numpy().flatten()
            
            # Cache the embedding
            self.cache_manager.set(f"user_emb:{user_id}", user_embedding_np.tolist(), ttl=7200)  # 2 hours
            
            return user_embedding_np
            
        except Exception as e:
            logger.error(f"Error getting user embedding for user {user_id}: {e}")
            return None
    
    async def _rerank_candidates(self,
                               candidates: List[Dict[str, Any]],
                               user_id: int,
                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-rank candidates using CTR model"""
        
        if not self.ctr_ranker or not candidates:
            return candidates
        
        try:
            # Extract candidate item IDs
            candidate_ids = [c['item_id'] for c in candidates]
            
            # Get user features
            user_features = self.user_features.get(user_id, {})
            
            # Get item features for all candidates
            item_features_map = {}
            for item_id in candidate_ids:
                item_features_map[item_id] = self.item_features.get(item_id, {})
            
            # Re-rank using CTR model
            reranked = self.ctr_ranker.rerank(
                candidate_ids,
                user_id,
                user_features,
                item_features_map,
                context,
                top_k=len(candidates)
            )
            
            # Convert back to candidate format
            reranked_candidates = []
            for item_id, ctr_score in reranked:
                # Find original candidate
                original_candidate = next(
                    (c for c in candidates if c['item_id'] == item_id), 
                    None
                )
                
                if original_candidate:
                    candidate = original_candidate.copy()
                    candidate['ctr_score'] = float(ctr_score)
                    candidate['final_score'] = float(ctr_score)  # Use CTR as final score
                    reranked_candidates.append(candidate)
            
            return reranked_candidates
            
        except Exception as e:
            logger.error(f"Error re-ranking candidates: {e}")
            return candidates
    
    def _filter_by_categories(self,
                             candidates: List[Dict[str, Any]],
                             categories: List[str]) -> List[Dict[str, Any]]:
        """Filter candidates by categories"""
        
        filtered = []
        
        for candidate in candidates:
            item_id = candidate['item_id']
            
            # Get item metadata
            item_meta = self.item_metadata.get(item_id, {})
            item_categories = item_meta.get('categories', [])
            
            # Check if item has any of the requested categories
            if any(cat in item_categories for cat in categories):
                filtered.append(candidate)
        
        return filtered
    
    async def _format_recommendations(self,
                                    candidates: List[Dict[str, Any]],
                                    explain: bool = False) -> List[Dict[str, Any]]:
        """Format candidates into final recommendation format"""
        
        recommendations = []
        
        for candidate in candidates:
            item_id = candidate['item_id']
            
            # Get item metadata
            item_meta = self.item_metadata.get(item_id, {})
            
            recommendation = {
                'item_id': item_id,
                'title': item_meta.get('title', f'Item {item_id}'),
                'score': candidate.get('final_score', candidate.get('similarity_score', 0.0)),
                'categories': item_meta.get('categories', []),
                'metadata': {
                    'similarity_score': candidate.get('similarity_score', 0.0),
                    'ctr_score': candidate.get('ctr_score'),
                    'source': candidate.get('source', 'unknown')
                }
            }
            
            # Add explanation if requested
            if explain:
                explanation = self._generate_explanation(candidate, item_meta)
                recommendation['explanation'] = explanation
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_explanation(self, 
                            candidate: Dict[str, Any], 
                            item_meta: Dict[str, Any]) -> str:
        """Generate explanation for a recommendation"""
        
        explanations = []
        
        # Similarity-based explanation
        sim_score = candidate.get('similarity_score', 0)
        if sim_score > 0.8:
            explanations.append("Highly similar to your preferences")
        elif sim_score > 0.6:
            explanations.append("Similar to items you've enjoyed")
        
        # Category-based explanation
        categories = item_meta.get('categories', [])
        if categories:
            explanations.append(f"Based on your interest in {', '.join(categories[:2])}")
        
        # CTR-based explanation
        ctr_score = candidate.get('ctr_score')
        if ctr_score and ctr_score > 0.5:
            explanations.append("Popular among users like you")
        
        # Default explanation
        if not explanations:
            explanations.append("Recommended for you")
        
        return "; ".join(explanations)
    
    async def _get_popular_items(self, 
                               num_items: int, 
                               exclude_items: List[int]) -> List[Dict[str, Any]]:
        """Get popular items as fallback"""
        
        # This is a simplified implementation
        # In production, you'd query a database for popular items
        
        popular_items = []
        item_id = 1
        
        while len(popular_items) < num_items and item_id <= 1000:
            if item_id not in exclude_items:
                popular_items.append({
                    'item_id': item_id,
                    'similarity_score': 0.5,
                    'source': 'popularity'
                })
            item_id += 1
        
        return popular_items
    
    async def _get_fallback_recommendations(self, num_items: int) -> List[Dict[str, Any]]:
        """Get fallback recommendations when main pipeline fails"""
        
        popular_items = await self._get_popular_items(num_items, [])
        return await self._format_recommendations(popular_items)
    
    async def get_similar_items(self, 
                              item_id: int, 
                              num_items: int = 10) -> List[Dict[str, Any]]:
        """Get items similar to a given item"""
        
        if not self.faiss_index or not self.two_tower_model:
            return []
        
        try:
            # Get item embedding
            if self.item_encoder and item_id in self.item_encoder.classes_:
                encoded_item_id = self.item_encoder.transform([item_id])[0]
            else:
                encoded_item_id = item_id
            
            # Get item features
            item_features_vec = self.item_features.get(item_id)
            if item_features_vec is not None:
                item_features_tensor = torch.tensor(item_features_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                item_features_tensor = None
            
            # Generate embedding
            with torch.no_grad():
                item_id_tensor = torch.tensor([encoded_item_id], dtype=torch.long).to(self.device)
                item_embedding = self.two_tower_model.get_item_embedding(item_id_tensor, item_features_tensor)
                item_embedding_np = item_embedding.cpu().numpy()
            
            # Search for similar items
            similar_ids, similarities = self.faiss_index.search(item_embedding_np, k=num_items + 1)
            
            # Remove the query item itself and format results
            similar_items = []
            for sim_item_id, similarity in zip(similar_ids[0], similarities[0]):
                if sim_item_id != item_id:
                    item_meta = self.item_metadata.get(sim_item_id, {})
                    similar_items.append({
                        'item_id': sim_item_id,
                        'title': item_meta.get('title', f'Item {sim_item_id}'),
                        'similarity': float(similarity),
                        'categories': item_meta.get('categories', [])
                    })
            
            return similar_items[:num_items]
            
        except Exception as e:
            logger.error(f"Error getting similar items for {item_id}: {e}")
            return []
    
    async def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user profile information"""
        
        try:
            # Get user features
            user_features = self.user_features.get(user_id, {})
            
            # Get user embedding
            user_embedding = await self._get_user_embedding(user_id)
            
            profile = {
                'user_id': user_id,
                'features': user_features,
                'has_embedding': user_embedding is not None,
                'profile_completeness': len(user_features) / 10.0  # Assuming 10 expected features
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile for {user_id}: {e}")
            return None
    
    async def get_trending_items(self,
                               num_items: int = 20,
                               category: Optional[str] = None,
                               time_window: str = "24h") -> List[Dict[str, Any]]:
        """Get trending items"""
        
        # Simplified implementation - in production, this would query recent interaction data
        trending = await self._get_popular_items(num_items, [])
        return await self._format_recommendations(trending)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics"""
        
        return {
            'models_loaded': self.models_loaded,
            'loaded_at': self.loaded_at.isoformat() if self.loaded_at else None,
            'device': self.device,
            **self.stats
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        return {
            'two_tower_model': self.two_tower_model is not None,
            'faiss_index': {
                'loaded': self.faiss_index is not None,
                'num_vectors': self.faiss_index.index.ntotal if self.faiss_index else 0
            },
            'ctr_ranker': self.ctr_ranker is not None,
            'content_embedder': self.content_embedder is not None,
            'user_features_count': len(self.user_features),
            'item_features_count': len(self.item_features),
            'item_metadata_count': len(self.item_metadata)
        }