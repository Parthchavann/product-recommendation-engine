import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

from ..utils.logger import get_logger


logger = get_logger(__name__)


class TowerMLP(nn.Module):
    """Multi-layer perceptron tower for two-tower architecture"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.2,
                 batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Skip batch norm and activation for output layer
            if i < len(dims) - 2:
                # Batch normalization
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                # Activation
                if activation.lower() == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation.lower() == 'gelu':
                    self.layers.append(nn.GELU())
                elif activation.lower() == 'leaky_relu':
                    self.layers.append(nn.LeakyReLU(0.1))
                
                # Dropout
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TwoTowerModel(nn.Module):
    """Two-tower architecture for candidate generation"""
    
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 128,
                 tower_dims: List[int] = [256, 128, 64],
                 user_feature_dim: int = 10,
                 item_feature_dim: int = 20,
                 dropout: float = 0.2,
                 temperature: float = 0.07,
                 use_bias: bool = False):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.output_dim = tower_dims[-1]
        
        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_users, 
            embedding_dim,
            sparse=False  # Set to True for large embeddings
        )
        
        self.item_embedding = nn.Embedding(
            num_items, 
            embedding_dim,
            sparse=False
        )
        
        # User tower
        user_input_dim = embedding_dim + user_feature_dim
        self.user_tower = TowerMLP(
            user_input_dim,
            tower_dims[:-1],
            tower_dims[-1],
            dropout=dropout
        )
        
        # Item tower
        item_input_dim = embedding_dim + item_feature_dim
        self.item_tower = TowerMLP(
            item_input_dim,
            tower_dims[:-1],
            tower_dims[-1],
            dropout=dropout
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Optional bias terms
        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.tensor(0.0))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        
        # Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Initialize biases to zero
        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, 
                user_ids: torch.Tensor,
                item_ids: torch.Tensor,
                user_features: Optional[torch.Tensor] = None,
                item_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size] or [batch_size, num_items]
            user_features: User features [batch_size, user_feature_dim]
            item_features: Item features [batch_size, item_feature_dim] or [batch_size, num_items, item_feature_dim]
            
        Returns:
            Tuple of (similarity_scores, user_vectors, item_vectors)
        """
        
        # Get user representations
        user_vectors = self.get_user_embedding(user_ids, user_features)
        
        # Handle different item tensor shapes
        if item_ids.dim() == 1:
            # Single item per user
            item_vectors = self.get_item_embedding(item_ids, item_features)
            
            # Compute dot product similarity
            similarities = torch.sum(user_vectors * item_vectors, dim=-1)
            
        else:
            # Multiple items per user (for batch negative sampling)
            batch_size, num_items = item_ids.shape
            
            # Flatten for embedding lookup
            item_ids_flat = item_ids.view(-1)
            
            if item_features is not None:
                item_features_flat = item_features.view(-1, item_features.shape[-1])
            else:
                item_features_flat = None
            
            # Get item embeddings
            item_vectors_flat = self.get_item_embedding(item_ids_flat, item_features_flat)
            item_vectors = item_vectors_flat.view(batch_size, num_items, -1)
            
            # Compute similarities: [batch_size, num_items]
            user_vectors_expanded = user_vectors.unsqueeze(1)  # [batch_size, 1, output_dim]
            similarities = torch.sum(user_vectors_expanded * item_vectors, dim=-1)
        
        # Add bias terms if enabled
        if self.use_bias:
            user_bias = self.user_bias(user_ids).squeeze(-1)
            
            if item_ids.dim() == 1:
                item_bias = self.item_bias(item_ids).squeeze(-1)
            else:
                item_bias = self.item_bias(item_ids).squeeze(-1)
            
            similarities = similarities + user_bias.unsqueeze(-1) + item_bias + self.global_bias
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        return similarities, user_vectors, item_vectors
    
    def get_user_embedding(self, 
                          user_ids: torch.Tensor,
                          user_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get user embeddings"""
        
        # User embedding lookup
        user_emb = self.user_embedding(user_ids)
        
        # Concatenate with features if provided
        if user_features is not None:
            user_input = torch.cat([user_emb, user_features], dim=-1)
        else:
            # Pad with zeros if no features
            zero_features = torch.zeros(
                user_emb.size(0), 
                self.item_tower.layers[0].in_features - self.embedding_dim,
                device=user_emb.device
            )
            user_input = torch.cat([user_emb, zero_features], dim=-1)
        
        # Pass through user tower
        user_vector = self.user_tower(user_input)
        
        # L2 normalize
        user_vector = F.normalize(user_vector, p=2, dim=-1)
        
        return user_vector
    
    def get_item_embedding(self, 
                          item_ids: torch.Tensor,
                          item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get item embeddings"""
        
        # Item embedding lookup
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate with features if provided
        if item_features is not None:
            item_input = torch.cat([item_emb, item_features], dim=-1)
        else:
            # Pad with zeros if no features
            zero_features = torch.zeros(
                item_emb.size(0),
                self.item_tower.layers[0].in_features - self.embedding_dim,
                device=item_emb.device
            )
            item_input = torch.cat([item_emb, zero_features], dim=-1)
        
        # Pass through item tower
        item_vector = self.item_tower(item_input)
        
        # L2 normalize
        item_vector = F.normalize(item_vector, p=2, dim=-1)
        
        return item_vector
    
    def predict(self, 
               user_ids: torch.Tensor,
               item_ids: torch.Tensor,
               user_features: Optional[torch.Tensor] = None,
               item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Make predictions (for inference)
        
        Returns:
            Similarity scores
        """
        
        with torch.no_grad():
            similarities, _, _ = self.forward(
                user_ids, item_ids, user_features, item_features
            )
        
        return torch.sigmoid(similarities)
    
    def get_all_item_embeddings(self, 
                               item_features_dict: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Get embeddings for all items (for FAISS indexing)
        
        Args:
            item_features_dict: Dictionary mapping item_id to features
            
        Returns:
            Item embeddings [num_items, output_dim]
        """
        
        device = next(self.parameters()).device
        all_item_ids = torch.arange(self.num_items, device=device)
        
        # Prepare features
        if item_features_dict is not None:
            features_list = []
            for i in range(self.num_items):
                if i in item_features_dict:
                    features_list.append(item_features_dict[i])
                else:
                    # Default zero features
                    default_features = torch.zeros(
                        self.item_tower.layers[0].in_features - self.embedding_dim,
                        device=device
                    )
                    features_list.append(default_features)
            
            all_item_features = torch.stack(features_list)
        else:
            all_item_features = None
        
        # Get embeddings in batches to avoid memory issues
        batch_size = 1000
        all_embeddings = []
        
        for i in range(0, self.num_items, batch_size):
            end_idx = min(i + batch_size, self.num_items)
            batch_ids = all_item_ids[i:end_idx]
            
            if all_item_features is not None:
                batch_features = all_item_features[i:end_idx]
            else:
                batch_features = None
            
            with torch.no_grad():
                batch_embeddings = self.get_item_embedding(batch_ids, batch_features)
                all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_embeddings(self, 
                       save_path: str,
                       user_features_dict: Optional[Dict[int, torch.Tensor]] = None,
                       item_features_dict: Optional[Dict[int, torch.Tensor]] = None):
        """Save user and item embeddings for serving"""
        
        logger.info(f"Saving embeddings to {save_path}")
        
        # Get all embeddings
        device = next(self.parameters()).device
        
        # User embeddings
        all_user_ids = torch.arange(self.num_users, device=device)
        if user_features_dict is not None:
            user_features_list = []
            for i in range(self.num_users):
                if i in user_features_dict:
                    user_features_list.append(user_features_dict[i])
                else:
                    default_features = torch.zeros(
                        self.user_tower.layers[0].in_features - self.embedding_dim,
                        device=device
                    )
                    user_features_list.append(default_features)
            all_user_features = torch.stack(user_features_list)
        else:
            all_user_features = None
        
        with torch.no_grad():
            user_embeddings = self.get_user_embedding(all_user_ids, all_user_features)
            item_embeddings = self.get_all_item_embeddings(item_features_dict)
        
        # Save to file
        torch.save({
            'user_embeddings': user_embeddings.cpu(),
            'item_embeddings': item_embeddings.cpu(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.output_dim
        }, save_path)
        
        logger.info(f"Saved embeddings: {user_embeddings.shape}, {item_embeddings.shape}")


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning in two-tower model"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, 
                user_vectors: torch.Tensor,
                item_vectors: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute InfoNCE loss
        
        Args:
            user_vectors: User embeddings [batch_size, dim]
            item_vectors: Item embeddings [batch_size, dim] or [batch_size, num_items, dim]
            labels: Ground truth labels (optional)
            
        Returns:
            Loss value
        """
        
        if item_vectors.dim() == 2:
            # Standard contrastive learning (batch negatives)
            # Compute similarity matrix
            similarities = torch.matmul(user_vectors, item_vectors.T) / self.temperature
            
            # Labels are diagonal (positive pairs)
            batch_size = user_vectors.size(0)
            labels = torch.arange(batch_size, device=user_vectors.device)
            
        else:
            # In-batch negatives with multiple items per user
            batch_size, num_items = item_vectors.shape[:2]
            
            # Reshape for matrix multiplication
            user_vectors_expanded = user_vectors.unsqueeze(1)  # [B, 1, D]
            
            # Compute similarities
            similarities = torch.sum(
                user_vectors_expanded * item_vectors, dim=-1
            ) / self.temperature  # [B, num_items]
            
            # First item is positive, rest are negatives
            if labels is None:
                labels = torch.zeros(batch_size, dtype=torch.long, device=user_vectors.device)
        
        return self.cross_entropy(similarities, labels)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                positive_scores: torch.Tensor,
                negative_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute BPR loss
        
        Args:
            positive_scores: Scores for positive items
            negative_scores: Scores for negative items
            
        Returns:
            BPR loss
        """
        
        # BPR assumes positive items should have higher scores than negative items
        difference = positive_scores - negative_scores
        loss = -torch.log(torch.sigmoid(difference)).mean()
        
        return loss