import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union, Tuple
import pickle
from pathlib import Path
from tqdm import tqdm
import gc

from ..utils.logger import get_logger
from ..utils.config import get_data_dir


logger = get_logger(__name__)


class ContentEmbedder:
    """Generate BERT embeddings for item metadata"""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: str = 'cuda',
                 max_length: int = 512,
                 batch_size: int = 32):
        
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        logger.info(f"Loaded embedding model: {model_name} on {self.device}")
        logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode text descriptions to embeddings
        
        Args:
            texts: List of text descriptions
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        # Encode in batches to manage memory
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     desc="Encoding texts", disable=not show_progress):
            
            batch_texts = texts[i:i + self.batch_size]
            
            # Clean and truncate texts
            cleaned_texts = [self._clean_text(text) for text in batch_texts]
            
            # Encode batch
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    cleaned_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalize
                )
            
            all_embeddings.append(batch_embeddings)
            
            # Clear cache periodically
            if i % (self.batch_size * 10) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (rough character limit)
        max_chars = self.max_length * 4  # Rough estimate for tokenization
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text
    
    def create_item_embeddings(self, 
                              movies_df: pd.DataFrame,
                              cache_path: Optional[str] = None) -> Dict[int, np.ndarray]:
        """
        Create embeddings for all items
        
        Args:
            movies_df: DataFrame with movie information
            cache_path: Path to cache embeddings
            
        Returns:
            Dictionary mapping item_id to embedding
        """
        
        # Check cache
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Creating item embeddings from scratch...")
        
        # Combine title, description, and categories for rich text representation
        texts = []
        item_ids = []
        
        for _, row in movies_df.iterrows():
            # Create comprehensive text description
            text_parts = []
            
            # Title
            if pd.notna(row.get('title')):
                text_parts.append(f"Title: {row['title']}")
            
            # Genres
            if pd.notna(row.get('genres')) and row['genres'] != '(no genres listed)':
                genres = row['genres'].replace('|', ', ')
                text_parts.append(f"Genres: {genres}")
            
            # Description (if available)
            if 'description' in row and pd.notna(row['description']):
                text_parts.append(f"Description: {row['description']}")
            
            # Combine all parts
            combined_text = '. '.join(text_parts)
            
            if not combined_text.strip():
                combined_text = "No information available"
            
            texts.append(combined_text)
            item_ids.append(row['item_id'])
        
        # Generate embeddings
        embeddings = self.encode_texts(texts)
        
        # Create mapping
        item_embeddings = {
            item_id: embedding 
            for item_id, embedding in zip(item_ids, embeddings)
        }
        
        # Cache results
        if cache_path:
            cache_dir = Path(cache_path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(item_embeddings, f)
            
            logger.info(f"Cached embeddings to {cache_path}")
        
        logger.info(f"Created embeddings for {len(item_embeddings)} items")
        
        return item_embeddings
    
    def create_user_embeddings(self, 
                              users_df: pd.DataFrame,
                              interactions_df: pd.DataFrame,
                              movies_df: pd.DataFrame,
                              cache_path: Optional[str] = None) -> Dict[int, np.ndarray]:
        """
        Create user embeddings based on interaction history
        
        Args:
            users_df: User demographics
            interactions_df: User-item interactions
            movies_df: Item metadata
            cache_path: Path to cache embeddings
            
        Returns:
            Dictionary mapping user_id to embedding
        """
        
        # Check cache
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached user embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Creating user embeddings from interaction history...")
        
        # Get item embeddings first
        item_cache_path = cache_path.replace('user_', 'item_') if cache_path else None
        item_embeddings = self.create_item_embeddings(movies_df, item_cache_path)
        
        user_embeddings = {}
        
        # Create user profiles based on interactions
        for user_id in tqdm(users_df['user_id'], desc="Creating user embeddings"):
            
            # Get user's interactions
            user_interactions = interactions_df[
                (interactions_df['user_id'] == user_id) & 
                (interactions_df['rating'] >= 4.0)  # Only positive interactions
            ]
            
            if len(user_interactions) == 0:
                # No positive interactions - use zero embedding
                embedding_dim = next(iter(item_embeddings.values())).shape[0]
                user_embeddings[user_id] = np.zeros(embedding_dim)
                continue
            
            # Get embeddings for interacted items
            item_embs = []
            weights = []
            
            for _, interaction in user_interactions.iterrows():
                item_id = interaction['item_id']
                if item_id in item_embeddings:
                    item_embs.append(item_embeddings[item_id])
                    # Weight by rating (higher ratings get more weight)
                    weights.append(interaction['rating'])
            
            if item_embs:
                # Weighted average of item embeddings
                item_embs = np.array(item_embs)
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                
                user_embedding = np.average(item_embs, axis=0, weights=weights)
                
                # L2 normalize
                norm = np.linalg.norm(user_embedding)
                if norm > 0:
                    user_embedding = user_embedding / norm
                
                user_embeddings[user_id] = user_embedding
            else:
                # Fallback to zero embedding
                embedding_dim = next(iter(item_embeddings.values())).shape[0]
                user_embeddings[user_id] = np.zeros(embedding_dim)
        
        # Cache results
        if cache_path:
            cache_dir = Path(cache_path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(user_embeddings, f)
            
            logger.info(f"Cached user embeddings to {cache_path}")
        
        logger.info(f"Created embeddings for {len(user_embeddings)} users")
        
        return user_embeddings
    
    def find_similar_items(self, 
                          query_embedding: np.ndarray,
                          item_embeddings: Dict[int, np.ndarray],
                          k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar items to a query embedding
        
        Args:
            query_embedding: Query embedding vector
            item_embeddings: Dictionary of item embeddings
            k: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        
        similarities = []
        
        for item_id, item_emb in item_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, item_emb)
            similarities.append((item_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def compute_item_similarity_matrix(self, 
                                     item_embeddings: Dict[int, np.ndarray],
                                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all items
        
        Args:
            item_embeddings: Dictionary of item embeddings
            save_path: Path to save similarity matrix
            
        Returns:
            Similarity matrix [num_items, num_items]
        """
        
        logger.info("Computing item similarity matrix...")
        
        item_ids = sorted(list(item_embeddings.keys()))
        embeddings_matrix = np.array([item_embeddings[item_id] for item_id in item_ids])
        
        # Compute cosine similarity matrix
        # embeddings are already normalized, so dot product = cosine similarity
        similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
        
        if save_path:
            np.save(save_path, similarity_matrix)
            
            # Also save item_id mapping
            mapping_path = save_path.replace('.npy', '_mapping.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(item_ids, f)
            
            logger.info(f"Saved similarity matrix to {save_path}")
        
        return similarity_matrix
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()


class HybridEmbedder:
    """Combines content embeddings with collaborative features"""
    
    def __init__(self, 
                 content_embedder: ContentEmbedder,
                 embedding_dim: int = 128):
        
        self.content_embedder = content_embedder
        self.embedding_dim = embedding_dim
        
        # Dimension reduction layer (if content embeddings are larger)
        content_dim = content_embedder.get_embedding_dimension()
        
        if content_dim > embedding_dim:
            self.projection = torch.nn.Linear(content_dim, embedding_dim)
            logger.info(f"Created projection layer: {content_dim} -> {embedding_dim}")
        else:
            self.projection = None
    
    def create_hybrid_item_features(self, 
                                   movies_df: pd.DataFrame,
                                   interactions_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Create hybrid item features combining content and collaborative signals
        
        Args:
            movies_df: Movies metadata
            interactions_df: User-item interactions
            
        Returns:
            Dictionary mapping item_id to hybrid features
        """
        
        # Get content embeddings
        content_embeddings = self.content_embedder.create_item_embeddings(movies_df)
        
        # Compute collaborative features
        item_stats = interactions_df.groupby('item_id').agg({
            'rating': ['mean', 'std', 'count'],
            'user_id': 'nunique'
        }).fillna(0)
        
        item_stats.columns = ['avg_rating', 'rating_std', 'num_ratings', 'num_users']
        
        # Normalize collaborative features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        collab_features_normalized = scaler.fit_transform(item_stats.values)
        
        # Combine features
        hybrid_features = {}
        
        for i, (item_id, content_emb) in enumerate(content_embeddings.items()):
            
            # Project content embedding if needed
            if self.projection is not None:
                with torch.no_grad():
                    content_emb_torch = torch.tensor(content_emb, dtype=torch.float32).unsqueeze(0)
                    projected = self.projection(content_emb_torch).squeeze(0).numpy()
                    content_part = projected
            else:
                content_part = content_emb
            
            # Get collaborative features
            if item_id in item_stats.index:
                idx = list(item_stats.index).index(item_id)
                collab_part = collab_features_normalized[idx]
            else:
                collab_part = np.zeros(collab_features_normalized.shape[1])
            
            # Combine (you can experiment with different combination strategies)
            # Simple concatenation
            hybrid_feature = np.concatenate([content_part, collab_part])
            
            hybrid_features[item_id] = hybrid_feature
        
        logger.info(f"Created hybrid features with dimension {hybrid_features[list(hybrid_features.keys())[0]].shape[0]}")
        
        return hybrid_features


def create_embeddings_cache(config, 
                           ratings_df: pd.DataFrame,
                           movies_df: pd.DataFrame,
                           users_df: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Create and cache content embeddings
    
    Args:
        config: Configuration object
        ratings_df: Ratings data
        movies_df: Movies data  
        users_df: Users data
        
    Returns:
        Tuple of (item_embeddings, user_embeddings)
    """
    
    # Setup paths
    embeddings_dir = get_data_dir() / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    item_cache_path = embeddings_dir / "item_embeddings.pkl"
    user_cache_path = embeddings_dir / "user_embeddings.pkl"
    
    # Create embedder
    embedder = ContentEmbedder(
        model_name=config.content.model_name,
        batch_size=config.content.batch_size,
        max_length=config.content.max_length
    )
    
    # Create item embeddings
    item_embeddings = embedder.create_item_embeddings(
        movies_df, 
        str(item_cache_path)
    )
    
    # Create user embeddings
    user_embeddings = embedder.create_user_embeddings(
        users_df, 
        ratings_df, 
        movies_df,
        str(user_cache_path)
    )
    
    logger.info("Content embeddings created and cached successfully!")
    
    return item_embeddings, user_embeddings