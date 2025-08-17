import asyncio
import threading
import time
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import torch

from ..utils.logger import get_logger
from ..utils.config import Config
from .training import ModelTrainer
from ..retrieval.faiss_index import FAISSIndex


logger = get_logger(__name__)


@dataclass
class ModelUpdateConfig:
    """Configuration for model updates"""
    update_frequency: str = "daily"  # hourly, daily, weekly
    trigger_threshold: float = 0.05  # Metric degradation threshold
    validation_samples: int = 1000
    backup_versions: int = 3
    auto_rollback: bool = True
    min_training_data: int = 10000


class ModelUpdateManager:
    """Manages real-time model updates and hot-swapping"""
    
    def __init__(self, 
                 config: ModelUpdateConfig,
                 models_dir: Path,
                 data_dir: Path,
                 inference_engine=None):
        self.config = config
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.inference_engine = inference_engine
        
        # Update tracking
        self.update_history = []
        self.performance_metrics = {}
        self.is_updating = False
        self.last_update = None
        
        # Background update thread
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.update_callbacks: List[Callable] = []
        
        logger.info(f"Initialized ModelUpdateManager with frequency: {config.update_frequency}")
    
    def start_auto_updates(self):
        """Start automatic model updates"""
        if self.update_thread and self.update_thread.is_alive():
            logger.warning("Auto-updates already running")
            return
        
        self.stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        logger.info("Started automatic model updates")
    
    def stop_auto_updates(self):
        """Stop automatic model updates"""
        if not self.update_thread:
            return
        
        self.stop_event.set()
        self.update_thread.join(timeout=30)
        
        logger.info("Stopped automatic model updates")
    
    def add_update_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback to be called when models are updated"""
        self.update_callbacks.append(callback)
    
    def _update_loop(self):
        """Main update loop running in background thread"""
        while not self.stop_event.is_set():
            try:
                # Calculate next update time
                next_update = self._calculate_next_update()
                sleep_time = (next_update - datetime.now()).total_seconds()
                
                if sleep_time > 0:
                    if self.stop_event.wait(min(sleep_time, 3600)):  # Check every hour max
                        break
                
                # Check if update is needed
                if self._should_update():
                    asyncio.run(self.trigger_update())
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _calculate_next_update(self) -> datetime:
        """Calculate when the next update should occur"""
        now = datetime.now()
        
        if self.config.update_frequency == "hourly":
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif self.config.update_frequency == "daily":
            return now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif self.config.update_frequency == "weekly":
            days_ahead = 6 - now.weekday()  # Sunday
            return (now + timedelta(days=days_ahead)).replace(hour=2, minute=0, second=0, microsecond=0)
        else:
            return now + timedelta(hours=24)  # Default to daily
    
    def _should_update(self) -> bool:
        """Check if an update should be triggered"""
        # Check if enough time has passed
        if self.last_update:
            time_since_update = datetime.now() - self.last_update
            min_interval = timedelta(hours=1)
            
            if time_since_update < min_interval:
                return False
        
        # Check if currently updating
        if self.is_updating:
            return False
        
        # Check data availability
        if not self._has_sufficient_data():
            logger.info("Insufficient training data for update")
            return False
        
        # Check performance degradation
        if self._performance_degraded():
            logger.info("Performance degradation detected, triggering update")
            return True
        
        # Check scheduled update
        if self._is_scheduled_update_time():
            logger.info("Scheduled update time reached")
            return True
        
        return False
    
    def _has_sufficient_data(self) -> bool:
        """Check if there's enough new data for training"""
        try:
            # Check for new interaction data
            interaction_files = list(self.data_dir.glob("interactions_*.csv"))
            if not interaction_files:
                return False
            
            # Count recent interactions
            recent_data_count = 0
            cutoff_time = datetime.now() - timedelta(days=1)
            
            for file_path in interaction_files:
                try:
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time > cutoff_time:
                        # Rough estimate - in production, you'd parse the file
                        recent_data_count += 1000  # Assume 1000 interactions per file
                except:
                    continue
            
            return recent_data_count >= self.config.min_training_data
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return False
    
    def _performance_degraded(self) -> bool:
        """Check if model performance has degraded"""
        if not self.performance_metrics:
            return False
        
        current_metrics = self.performance_metrics.get('current', {})
        baseline_metrics = self.performance_metrics.get('baseline', {})
        
        if not current_metrics or not baseline_metrics:
            return False
        
        # Check key metrics
        key_metrics = ['ndcg_10', 'mrr', 'recall_20']
        
        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                current_value = current_metrics[metric]
                baseline_value = baseline_metrics[metric]
                
                # Check if degradation exceeds threshold
                if baseline_value > 0:
                    degradation = (baseline_value - current_value) / baseline_value
                    if degradation > self.config.trigger_threshold:
                        logger.warning(f"Performance degradation detected: {metric} dropped by {degradation:.2%}")
                        return True
        
        return False
    
    def _is_scheduled_update_time(self) -> bool:
        """Check if it's time for a scheduled update"""
        if not self.last_update:
            return True
        
        now = datetime.now()
        time_since_update = now - self.last_update
        
        if self.config.update_frequency == "hourly":
            return time_since_update >= timedelta(hours=1)
        elif self.config.update_frequency == "daily":
            return time_since_update >= timedelta(days=1)
        elif self.config.update_frequency == "weekly":
            return time_since_update >= timedelta(weeks=1)
        
        return False
    
    async def trigger_update(self, force: bool = False) -> Dict[str, Any]:
        """Trigger model update"""
        if self.is_updating and not force:
            return {"status": "already_updating"}
        
        self.is_updating = True
        update_start = datetime.now()
        
        try:
            logger.info("Starting model update process...")
            
            # Create backup of current models
            backup_path = self._create_model_backup()
            
            # Prepare training data
            training_data = await self._prepare_training_data()
            
            # Train new models
            new_models = await self._train_models(training_data)
            
            # Validate new models
            validation_results = await self._validate_models(new_models, training_data)
            
            # Check if new models are better
            if validation_results['should_deploy']:
                # Deploy new models
                deployment_result = await self._deploy_models(new_models)
                
                # Update performance tracking
                self._update_performance_metrics(validation_results['metrics'])
                
                # Notify callbacks
                self._notify_update_callbacks("models_updated", {
                    "validation_results": validation_results,
                    "deployment_result": deployment_result
                })
                
                self.last_update = update_start
                
                result = {
                    "status": "success",
                    "update_time": update_start.isoformat(),
                    "validation_results": validation_results,
                    "deployment_result": deployment_result
                }
                
                logger.info(f"Model update completed successfully in {(datetime.now() - update_start).total_seconds():.1f}s")
                
            else:
                # Rollback if needed
                if self.config.auto_rollback:
                    await self._restore_from_backup(backup_path)
                
                result = {
                    "status": "skipped",
                    "reason": "New models did not improve performance",
                    "validation_results": validation_results
                }
                
                logger.info("Model update skipped - new models did not improve performance")
            
            # Record update history
            self.update_history.append({
                "timestamp": update_start,
                "result": result,
                "duration_seconds": (datetime.now() - update_start).total_seconds()
            })
            
            # Keep only recent history
            if len(self.update_history) > 50:
                self.update_history = self.update_history[-50:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error during model update: {e}")
            
            # Attempt rollback
            if self.config.auto_rollback and 'backup_path' in locals():
                try:
                    await self._restore_from_backup(backup_path)
                    logger.info("Successfully rolled back to previous models")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback models: {rollback_error}")
            
            return {
                "status": "error",
                "error": str(e),
                "update_time": update_start.isoformat()
            }
            
        finally:
            self.is_updating = False
    
    def _create_model_backup(self) -> Path:
        """Create backup of current models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.models_dir / "backups" / f"backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy current model files
        for model_file in self.models_dir.glob("*.pkl"):
            shutil.copy2(model_file, backup_dir)
        
        for model_file in self.models_dir.glob("*.pt"):
            shutil.copy2(model_file, backup_dir)
        
        # Clean up old backups
        self._cleanup_old_backups()
        
        logger.info(f"Created model backup at {backup_dir}")
        return backup_dir
    
    def _cleanup_old_backups(self):
        """Remove old backup directories"""
        backup_dirs = sorted(
            [d for d in (self.models_dir / "backups").glob("backup_*") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep only the most recent backups
        for old_backup in backup_dirs[self.config.backup_versions:]:
            try:
                shutil.rmtree(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")
    
    async def _prepare_training_data(self) -> Dict[str, Any]:
        """Prepare data for model training"""
        # This would involve:
        # 1. Loading recent interaction data
        # 2. Preprocessing and feature engineering
        # 3. Creating train/validation splits
        # 4. Data quality checks
        
        # Placeholder implementation
        return {
            "interactions": [],
            "users": [],
            "items": [],
            "train_split": 0.8,
            "data_quality_score": 0.95
        }
    
    async def _train_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train new model versions"""
        try:
            # Initialize trainer
            trainer = ModelTrainer(
                config=self._get_training_config(),
                output_dir=self.models_dir / "temp_training"
            )
            
            # Train models
            training_results = await asyncio.to_thread(
                trainer.train_all_models,
                training_data
            )
            
            return {
                "two_tower_model": training_results.get("two_tower"),
                "lightgbm_model": training_results.get("lightgbm"),
                "faiss_index": training_results.get("faiss_index"),
                "training_metrics": training_results.get("metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    async def _validate_models(self, new_models: Dict[str, Any], training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate new models against current models"""
        try:
            # Load validation data
            validation_data = training_data.get('validation_data', [])
            
            if not validation_data:
                # Use a subset of training data for validation
                validation_data = training_data.get('interactions', [])[:self.config.validation_samples]
            
            # Run validation tests
            current_metrics = await self._evaluate_current_models(validation_data)
            new_metrics = await self._evaluate_new_models(new_models, validation_data)
            
            # Compare performance
            improvement_score = self._calculate_improvement_score(current_metrics, new_metrics)
            
            should_deploy = improvement_score > 0.01  # Deploy if >1% improvement
            
            return {
                "should_deploy": should_deploy,
                "improvement_score": improvement_score,
                "current_metrics": current_metrics,
                "new_metrics": new_metrics,
                "metrics": new_metrics
            }
            
        except Exception as e:
            logger.error(f"Error validating models: {e}")
            return {
                "should_deploy": False,
                "error": str(e)
            }
    
    async def _evaluate_current_models(self, validation_data: List) -> Dict[str, float]:
        """Evaluate current production models"""
        # Placeholder - would evaluate current models on validation data
        return {
            "ndcg_10": 0.35,
            "mrr": 0.25,
            "recall_20": 0.18
        }
    
    async def _evaluate_new_models(self, new_models: Dict[str, Any], validation_data: List) -> Dict[str, float]:
        """Evaluate new models on validation data"""
        # Placeholder - would evaluate new models on validation data
        return {
            "ndcg_10": 0.37,
            "mrr": 0.26,
            "recall_20": 0.19
        }
    
    def _calculate_improvement_score(self, current_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> float:
        """Calculate overall improvement score"""
        improvements = []
        
        for metric in ['ndcg_10', 'mrr', 'recall_20']:
            if metric in current_metrics and metric in new_metrics:
                current_val = current_metrics[metric]
                new_val = new_metrics[metric]
                
                if current_val > 0:
                    improvement = (new_val - current_val) / current_val
                    improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    async def _deploy_models(self, new_models: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy new models to production"""
        try:
            deployment_results = {}
            
            # Save new models to production location
            if "two_tower_model" in new_models:
                model_path = self.models_dir / "two_tower_model.pt"
                torch.save(new_models["two_tower_model"], model_path)
                deployment_results["two_tower"] = "deployed"
            
            if "lightgbm_model" in new_models:
                model_path = self.models_dir / "lightgbm_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(new_models["lightgbm_model"], f)
                deployment_results["lightgbm"] = "deployed"
            
            if "faiss_index" in new_models:
                index_path = self.models_dir / "faiss_index"
                new_models["faiss_index"].save(str(index_path))
                deployment_results["faiss_index"] = "deployed"
            
            # Hot-swap models in inference engine
            if self.inference_engine:
                await self.inference_engine.reload_models()
                deployment_results["inference_engine"] = "reloaded"
            
            return deployment_results
            
        except Exception as e:
            logger.error(f"Error deploying models: {e}")
            raise
    
    async def _restore_from_backup(self, backup_path: Path):
        """Restore models from backup"""
        try:
            # Copy backup files to production location
            for backup_file in backup_path.glob("*"):
                if backup_file.is_file():
                    production_file = self.models_dir / backup_file.name
                    shutil.copy2(backup_file, production_file)
            
            # Reload models in inference engine
            if self.inference_engine:
                await self.inference_engine.reload_models()
            
            logger.info(f"Successfully restored models from backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get configuration for model training"""
        return {
            "batch_size": 512,
            "epochs": 10,
            "learning_rate": 0.001,
            "early_stopping": True,
            "validation_split": 0.2
        }
    
    def _update_performance_metrics(self, new_metrics: Dict[str, float]):
        """Update performance tracking"""
        # Move current to baseline if this is first update
        if 'current' in self.performance_metrics and 'baseline' not in self.performance_metrics:
            self.performance_metrics['baseline'] = self.performance_metrics['current'].copy()
        
        # Update current metrics
        self.performance_metrics['current'] = new_metrics.copy()
        
        # Calculate trends
        if 'baseline' in self.performance_metrics:
            trends = {}
            for metric, value in new_metrics.items():
                if metric in self.performance_metrics['baseline']:
                    baseline_value = self.performance_metrics['baseline'][metric]
                    if baseline_value > 0:
                        trends[metric] = (value - baseline_value) / baseline_value
            
            self.performance_metrics['trends'] = trends
    
    def _notify_update_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify registered callbacks about updates"""
        for callback in self.update_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status and history"""
        return {
            "is_updating": self.is_updating,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "next_scheduled_update": self._calculate_next_update().isoformat(),
            "update_frequency": self.config.update_frequency,
            "performance_metrics": self.performance_metrics,
            "recent_updates": self.update_history[-10:],  # Last 10 updates
            "auto_updates_enabled": self.update_thread is not None and self.update_thread.is_alive()
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration dynamically"""
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
        
        # Restart auto-updates if frequency changed
        if 'update_frequency' in new_config and self.update_thread:
            self.stop_auto_updates()
            self.start_auto_updates()


class HotSwapManager:
    """Manages hot-swapping of models without downtime"""
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.swap_lock = asyncio.Lock()
        
    async def hot_swap_model(self, model_type: str, new_model_path: Path) -> bool:
        """Perform hot swap of a specific model"""
        async with self.swap_lock:
            try:
                if model_type == "two_tower":
                    await self._swap_two_tower_model(new_model_path)
                elif model_type == "lightgbm":
                    await self._swap_lightgbm_model(new_model_path)
                elif model_type == "faiss_index":
                    await self._swap_faiss_index(new_model_path)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                logger.info(f"Successfully hot-swapped {model_type} model")
                return True
                
            except Exception as e:
                logger.error(f"Failed to hot-swap {model_type} model: {e}")
                return False
    
    async def _swap_two_tower_model(self, model_path: Path):
        """Hot swap the two-tower model"""
        new_model = torch.load(model_path, map_location='cpu')
        self.inference_engine.two_tower_model = new_model
    
    async def _swap_lightgbm_model(self, model_path: Path):
        """Hot swap the LightGBM model"""
        with open(model_path, 'rb') as f:
            new_model = pickle.load(f)
        self.inference_engine.lightgbm_model = new_model
    
    async def _swap_faiss_index(self, index_path: Path):
        """Hot swap the FAISS index"""
        new_index = FAISSIndex(
            dimension=self.inference_engine.faiss_index.dimension,
            index_type=self.inference_engine.faiss_index.index_type
        )
        new_index.load(str(index_path))
        self.inference_engine.faiss_index = new_index