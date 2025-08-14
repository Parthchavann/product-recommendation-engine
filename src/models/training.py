import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import wandb
from pathlib import Path

from .two_tower import TwoTowerModel, InfoNCELoss, BPRLoss
from ..utils.config import Config
from ..utils.logger import get_logger, ExperimentTracker
from ..evaluation.metrics import RecommendationMetrics


logger = get_logger(__name__)


class TwoTowerTrainer:
    """Trainer for Two-Tower recommendation model"""
    
    def __init__(self, 
                 model: TwoTowerModel,
                 config: Config,
                 device: str = 'cuda'):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        if hasattr(config.training, 'loss_type') and config.training.loss_type == 'bpr':
            self.criterion = BPRLoss()
        else:
            self.criterion = InfoNCELoss(temperature=config.model.temperature)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics = RecommendationMetrics()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Experiment tracking
        self.use_wandb = hasattr(config.logging, 'wandb_project')
        if self.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        
        optimizer_name = getattr(self.config.training, 'optimizer', 'adamw').lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        
        scheduler_name = getattr(self.config.training, 'scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        
        wandb.init(
            project=self.config.logging.wandb_project,
            name=f"{self.config.model.name}_embedding{self.config.model.embedding_dim}",
            config={
                'model': {
                    'embedding_dim': self.config.model.embedding_dim,
                    'tower_dims': self.config.model.tower_dims,
                    'dropout': self.config.model.dropout,
                    'temperature': self.config.model.temperature,
                },
                'training': {
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.training.learning_rate,
                    'num_epochs': self.config.training.num_epochs,
                    'weight_decay': self.config.training.weight_decay,
                }
            }
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            similarities, user_vectors, item_vectors = self.model(
                batch['user_id'],
                batch['item_id'],
                batch.get('user_features'),
                batch.get('item_features')
            )
            
            # Compute loss
            if isinstance(self.criterion, InfoNCELoss):
                loss = self.criterion(user_vectors, item_vectors)
            else:  # BPR loss
                # For BPR, assume positive items have rating > 0.5
                positive_mask = batch['rating'] > 0.5
                negative_mask = ~positive_mask
                
                if positive_mask.any() and negative_mask.any():
                    positive_scores = similarities[positive_mask]
                    negative_scores = similarities[negative_mask]
                    
                    # Randomly pair positives with negatives
                    min_size = min(len(positive_scores), len(negative_scores))
                    pos_indices = torch.randperm(len(positive_scores))[:min_size]
                    neg_indices = torch.randperm(len(negative_scores))[:min_size]
                    
                    loss = self.criterion(
                        positive_scores[pos_indices],
                        negative_scores[neg_indices]
                    )
                else:
                    # Fallback to MSE if we don't have both positives and negatives
                    loss = nn.MSELoss()(similarities, batch['rating'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config.training, 'gradient_clip') and self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}"
            })
        
        return {'train_loss': total_loss / num_batches}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                similarities, user_vectors, item_vectors = self.model(
                    batch['user_id'],
                    batch['item_id'],
                    batch.get('user_features'),
                    batch.get('item_features')
                )
                
                # Compute loss
                if isinstance(self.criterion, InfoNCELoss):
                    loss = self.criterion(user_vectors, item_vectors)
                else:
                    positive_mask = batch['rating'] > 0.5
                    negative_mask = ~positive_mask
                    
                    if positive_mask.any() and negative_mask.any():
                        positive_scores = similarities[positive_mask]
                        negative_scores = similarities[negative_mask]
                        
                        min_size = min(len(positive_scores), len(negative_scores))
                        pos_indices = torch.randperm(len(positive_scores))[:min_size]
                        neg_indices = torch.randperm(len(negative_scores))[:min_size]
                        
                        loss = self.criterion(
                            positive_scores[pos_indices],
                            negative_scores[neg_indices]
                        )
                    else:
                        loss = nn.MSELoss()(similarities, batch['rating'])
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions for metrics
                predictions = torch.sigmoid(similarities).cpu().numpy()
                targets = batch['rating'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # AUC (if we have both positive and negative samples)
        if len(np.unique(all_targets)) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_targets, all_predictions)
        else:
            auc = 0.5
        
        # Accuracy (threshold at 0.5)
        predictions_binary = (all_predictions > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == all_targets)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_auc': auc,
            'val_accuracy': accuracy
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              test_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader (optional)
            
        Returns:
            Training history
        """
        
        logger.info("Starting training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_accuracy': []
        }
        
        for epoch in range(self.config.training.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            for key, value in train_metrics.items():
                history[key].append(value)
            
            for key, value in val_metrics.items():
                history[key].append(value)
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Val AUC: {val_metrics['val_auc']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'learning_rate': current_lr,
                    **train_metrics,
                    **val_metrics
                })
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Model checkpointing
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                
                logger.info(f"New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if hasattr(self.config.training, 'early_stopping_patience'):
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
            
            # Save periodic checkpoints
            if hasattr(self.config.logging, 'save_model_every'):
                if (epoch + 1) % self.config.logging.save_model_every == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', epoch, val_metrics)
        
        # Final test evaluation
        if test_loader is not None:
            logger.info("\nEvaluating on test set...")
            test_metrics = self.validate_epoch(test_loader)
            logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
            logger.info(f"Test AUC: {test_metrics['val_auc']:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['val_accuracy']:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'test_loss': test_metrics['val_loss'],
                    'test_auc': test_metrics['val_auc'],
                    'test_accuracy': test_metrics['val_accuracy']
                })
        
        logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()
        
        return history
    
    def save_checkpoint(self, 
                       filename: str, 
                       epoch: int, 
                       metrics: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filename)
        logger.info(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str) -> Dict:
        """Load model checkpoint"""
        
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint: {filename}")
        logger.info(f"Epoch: {checkpoint['epoch']}, Best Val Loss: {self.best_val_loss:.4f}")
        
        return checkpoint


def create_model(config: Config, 
                num_users: int, 
                num_items: int) -> TwoTowerModel:
    """Create two-tower model from config"""
    
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config.model.embedding_dim,
        tower_dims=config.model.tower_dims,
        dropout=config.model.dropout,
        temperature=config.model.temperature,
        user_feature_dim=10,  # From data preprocessing
        item_feature_dim=20   # From data preprocessing
    )
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model