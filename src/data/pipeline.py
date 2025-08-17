import asyncio
import schedule
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import pickle
import json

from ..utils.logger import get_logger
from ..utils.config import Config
from .data_loader import RecommendationDataset
from ..models.training import ModelTrainer


logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline"""
    source_type: str = "file"  # file, database, api
    source_path: str = "data/raw"
    processed_path: str = "data/processed"
    models_path: str = "models"
    
    # Scheduling
    ingestion_frequency: str = "hourly"  # hourly, daily, weekly
    training_frequency: str = "daily"
    
    # Data quality
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 10
    max_age_days: int = 90
    
    # Processing
    batch_size: int = 10000
    parallel_workers: int = 4
    
    # Validation
    test_split: float = 0.2
    validation_split: float = 0.1


class DataPipelineManager:
    """Manages automated data ingestion, processing, and model training"""
    
    def __init__(self, config: PipelineConfig, model_trainer: Optional[ModelTrainer] = None):
        self.config = config
        self.model_trainer = model_trainer
        
        # Pipeline state
        self.is_running = False
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.pipeline_stats = {
            'last_ingestion': None,
            'last_training': None,
            'records_processed': 0,
            'models_trained': 0,
            'errors': []
        }
        
        # Callbacks
        self.ingestion_callbacks: List[Callable] = []
        self.training_callbacks: List[Callable] = []
        
        # Create directories
        Path(self.config.source_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.processed_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.models_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataPipelineManager with config: {config}")
    
    def start_pipeline(self):
        """Start the automated data pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Schedule tasks
        self._schedule_tasks()
        
        # Start background thread
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            daemon=True
        )
        self.pipeline_thread.start()
        
        logger.info("Started automated data pipeline")
    
    def stop_pipeline(self):
        """Stop the automated data pipeline"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=30)
        
        # Clear scheduled tasks
        schedule.clear()
        
        logger.info("Stopped automated data pipeline")
    
    def _schedule_tasks(self):
        """Schedule pipeline tasks based on configuration"""
        # Schedule data ingestion
        if self.config.ingestion_frequency == "hourly":
            schedule.every().hour.do(self._run_ingestion)
        elif self.config.ingestion_frequency == "daily":
            schedule.every().day.at("01:00").do(self._run_ingestion)
        elif self.config.ingestion_frequency == "weekly":
            schedule.every().sunday.at("01:00").do(self._run_ingestion)
        
        # Schedule model training
        if self.config.training_frequency == "daily":
            schedule.every().day.at("03:00").do(self._run_training)
        elif self.config.training_frequency == "weekly":
            schedule.every().sunday.at("03:00").do(self._run_training)
        
        logger.info(f"Scheduled tasks: ingestion={self.config.ingestion_frequency}, training={self.config.training_frequency}")
    
    def _pipeline_loop(self):
        """Main pipeline loop"""
        while not self.stop_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in pipeline loop: {e}")
                self.pipeline_stats['errors'].append({
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'component': 'pipeline_loop'
                })
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _run_ingestion(self):
        """Run data ingestion task"""
        try:
            logger.info("Starting data ingestion...")
            
            # Ingest new data
            ingested_data = self.ingest_data()
            
            # Process the data
            processed_data = self.process_data(ingested_data)
            
            # Validate data quality
            validation_results = self.validate_data_quality(processed_data)
            
            # Save processed data
            self.save_processed_data(processed_data)
            
            # Update statistics
            self.pipeline_stats['last_ingestion'] = datetime.now()
            self.pipeline_stats['records_processed'] += len(processed_data.get('interactions', []))
            
            # Notify callbacks
            for callback in self.ingestion_callbacks:
                try:
                    callback('ingestion_complete', {
                        'processed_records': len(processed_data.get('interactions', [])),
                        'validation_results': validation_results
                    })
                except Exception as e:
                    logger.error(f"Error in ingestion callback: {e}")
            
            logger.info(f"Data ingestion completed: {len(processed_data.get('interactions', []))} records processed")
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            self.pipeline_stats['errors'].append({
                'timestamp': datetime.now(),
                'error': str(e),
                'component': 'ingestion'
            })
    
    def _run_training(self):
        """Run model training task"""
        try:
            logger.info("Starting model training...")
            
            if not self.model_trainer:
                logger.warning("No model trainer configured, skipping training")
                return
            
            # Load processed data
            training_data = self.load_training_data()
            
            if not training_data or len(training_data.get('interactions', [])) < 1000:
                logger.warning("Insufficient training data, skipping training")
                return
            
            # Train models
            training_results = await asyncio.run(self._train_models_async(training_data))
            
            # Validate model performance
            validation_results = self.validate_model_performance(training_results)
            
            # Deploy models if validation passes
            if validation_results.get('should_deploy', False):
                self.deploy_models(training_results)
            
            # Update statistics
            self.pipeline_stats['last_training'] = datetime.now()
            self.pipeline_stats['models_trained'] += 1
            
            # Notify callbacks
            for callback in self.training_callbacks:
                try:
                    callback('training_complete', {
                        'training_results': training_results,
                        'validation_results': validation_results
                    })
                except Exception as e:
                    logger.error(f"Error in training callback: {e}")
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            self.pipeline_stats['errors'].append({
                'timestamp': datetime.now(),
                'error': str(e),
                'component': 'training'
            })
    
    async def _train_models_async(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models asynchronously"""
        return await asyncio.to_thread(
            self.model_trainer.train_all_models,
            training_data
        )
    
    def ingest_data(self) -> Dict[str, Any]:
        """Ingest new data from configured source"""
        if self.config.source_type == "file":
            return self._ingest_from_files()
        elif self.config.source_type == "database":
            return self._ingest_from_database()
        elif self.config.source_type == "api":
            return self._ingest_from_api()
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
    
    def _ingest_from_files(self) -> Dict[str, Any]:
        """Ingest data from file system"""
        source_path = Path(self.config.source_path)
        
        ingested_data = {
            'interactions': [],
            'users': [],
            'items': [],
            'metadata': {
                'ingestion_time': datetime.now(),
                'source': 'files'
            }
        }
        
        # Look for new interaction files
        interaction_files = list(source_path.glob("interactions_*.csv"))
        cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour
        
        for file_path in interaction_files:
            try:
                # Check if file is new enough
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mod_time > cutoff_time:
                    df = pd.read_csv(file_path)
                    ingested_data['interactions'].extend(df.to_dict('records'))
                    logger.debug(f"Ingested {len(df)} interactions from {file_path}")
            except Exception as e:
                logger.error(f"Error ingesting file {file_path}: {e}")
        
        # Look for user and item metadata files
        user_files = list(source_path.glob("users_*.csv"))
        item_files = list(source_path.glob("items_*.csv"))
        
        for file_path in user_files[-1:]:  # Latest user file
            try:
                df = pd.read_csv(file_path)
                ingested_data['users'].extend(df.to_dict('records'))
            except Exception as e:
                logger.error(f"Error ingesting user file {file_path}: {e}")
        
        for file_path in item_files[-1:]:  # Latest item file
            try:
                df = pd.read_csv(file_path)
                ingested_data['items'].extend(df.to_dict('records'))
            except Exception as e:
                logger.error(f"Error ingesting item file {file_path}: {e}")
        
        return ingested_data
    
    def _ingest_from_database(self) -> Dict[str, Any]:
        """Ingest data from database"""
        # Placeholder for database ingestion
        # In production, this would connect to your data warehouse
        logger.warning("Database ingestion not implemented")
        return {'interactions': [], 'users': [], 'items': []}
    
    def _ingest_from_api(self) -> Dict[str, Any]:
        """Ingest data from API"""
        # Placeholder for API ingestion
        # In production, this would call your event tracking APIs
        logger.warning("API ingestion not implemented")
        return {'interactions': [], 'users': [], 'items': []}
    
    def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean ingested data"""
        logger.info("Processing ingested data...")
        
        processed_data = {
            'interactions': [],
            'users': [],
            'items': [],
            'metadata': raw_data.get('metadata', {})
        }
        
        # Process interactions
        interactions_df = pd.DataFrame(raw_data.get('interactions', []))
        if not interactions_df.empty:
            # Data cleaning
            interactions_df = self._clean_interactions(interactions_df)
            
            # Feature engineering
            interactions_df = self._engineer_features(interactions_df)
            
            # Quality filtering
            interactions_df = self._filter_quality_interactions(interactions_df)
            
            processed_data['interactions'] = interactions_df.to_dict('records')
        
        # Process users
        users_df = pd.DataFrame(raw_data.get('users', []))
        if not users_df.empty:
            users_df = self._clean_users(users_df)
            processed_data['users'] = users_df.to_dict('records')
        
        # Process items
        items_df = pd.DataFrame(raw_data.get('items', []))
        if not items_df.empty:
            items_df = self._clean_items(items_df)
            processed_data['items'] = items_df.to_dict('records')
        
        processed_data['metadata']['processing_time'] = datetime.now()
        processed_data['metadata']['processed_interactions'] = len(processed_data['interactions'])
        
        logger.info(f"Data processing completed: {len(processed_data['interactions'])} interactions processed")
        
        return processed_data
    
    def _clean_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean interaction data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])
        
        # Convert timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove old interactions
        cutoff_date = datetime.now() - timedelta(days=self.config.max_age_days)
        if 'timestamp' in df.columns:
            df = df[df['timestamp'] > cutoff_date]
        
        # Remove invalid user/item IDs
        df = df.dropna(subset=['user_id', 'item_id'])
        df = df[df['user_id'] > 0]
        df = df[df['item_id'] > 0]
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to interactions"""
        if df.empty:
            return df
        
        # Add time-based features
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Add user interaction counts
        user_counts = df.groupby('user_id').size()
        df['user_interaction_count'] = df['user_id'].map(user_counts)
        
        # Add item interaction counts
        item_counts = df.groupby('item_id').size()
        df['item_interaction_count'] = df['item_id'].map(item_counts)
        
        # Add recency features
        if 'timestamp' in df.columns:
            now = datetime.now()
            df['days_since_interaction'] = (now - df['timestamp']).dt.days
        
        return df
    
    def _filter_quality_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter interactions based on quality criteria"""
        if df.empty:
            return df
        
        # Filter users with minimum interactions
        user_counts = df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= self.config.min_interactions_per_user].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items with minimum interactions
        item_counts = df.groupby('item_id').size()
        valid_items = item_counts[item_counts >= self.config.min_interactions_per_item].index
        df = df[df['item_id'].isin(valid_items)]
        
        return df
    
    def _clean_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean user data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id'])
        
        # Remove invalid user IDs
        df = df.dropna(subset=['user_id'])
        df = df[df['user_id'] > 0]
        
        return df
    
    def _clean_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean item data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['item_id'])
        
        # Remove invalid item IDs
        df = df.dropna(subset=['item_id'])
        df = df[df['item_id'] > 0]
        
        return df
    
    def validate_data_quality(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality"""
        validation_results = {
            'passed': True,
            'issues': [],
            'metrics': {},
            'timestamp': datetime.now()
        }
        
        interactions = processed_data.get('interactions', [])
        
        # Check minimum data volume
        if len(interactions) < 100:
            validation_results['passed'] = False
            validation_results['issues'].append(f"Insufficient interactions: {len(interactions)} < 100")
        
        if interactions:
            df = pd.DataFrame(interactions)
            
            # Check user coverage
            unique_users = df['user_id'].nunique()
            validation_results['metrics']['unique_users'] = unique_users
            
            if unique_users < 10:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Insufficient users: {unique_users} < 10")
            
            # Check item coverage
            unique_items = df['item_id'].nunique()
            validation_results['metrics']['unique_items'] = unique_items
            
            if unique_items < 10:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Insufficient items: {unique_items} < 10")
            
            # Check data freshness
            if 'timestamp' in df.columns:
                latest_timestamp = pd.to_datetime(df['timestamp']).max()
                hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600
                validation_results['metrics']['data_age_hours'] = hours_old
                
                if hours_old > 24:
                    validation_results['issues'].append(f"Data is {hours_old:.1f} hours old")
        
        return validation_results
    
    def save_processed_data(self, processed_data: Dict[str, Any]):
        """Save processed data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_path = Path(self.config.processed_path)
        
        # Save interactions
        if processed_data.get('interactions'):
            df = pd.DataFrame(processed_data['interactions'])
            file_path = processed_path / f"interactions_processed_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            logger.debug(f"Saved {len(df)} processed interactions to {file_path}")
        
        # Save users
        if processed_data.get('users'):
            df = pd.DataFrame(processed_data['users'])
            file_path = processed_path / f"users_processed_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            logger.debug(f"Saved {len(df)} processed users to {file_path}")
        
        # Save items
        if processed_data.get('items'):
            df = pd.DataFrame(processed_data['items'])
            file_path = processed_path / f"items_processed_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            logger.debug(f"Saved {len(df)} processed items to {file_path}")
        
        # Save metadata
        metadata_path = processed_path / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(processed_data.get('metadata', {}), f, indent=2, default=str)
    
    def load_training_data(self) -> Dict[str, Any]:
        """Load processed data for training"""
        processed_path = Path(self.config.processed_path)
        
        # Find latest processed files
        interaction_files = sorted(processed_path.glob("interactions_processed_*.csv"))
        user_files = sorted(processed_path.glob("users_processed_*.csv"))
        item_files = sorted(processed_path.glob("items_processed_*.csv"))
        
        training_data = {
            'interactions': [],
            'users': [],
            'items': []
        }
        
        # Load latest interactions
        if interaction_files:
            df = pd.read_csv(interaction_files[-1])
            training_data['interactions'] = df.to_dict('records')
        
        # Load latest users
        if user_files:
            df = pd.read_csv(user_files[-1])
            training_data['users'] = df.to_dict('records')
        
        # Load latest items
        if item_files:
            df = pd.read_csv(item_files[-1])
            training_data['items'] = df.to_dict('records')
        
        return training_data
    
    def validate_model_performance(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trained model performance"""
        validation_results = {
            'should_deploy': False,
            'metrics': training_results.get('validation_metrics', {}),
            'issues': [],
            'timestamp': datetime.now()
        }
        
        # Check if validation metrics meet thresholds
        metrics = validation_results['metrics']
        
        # Minimum performance thresholds
        min_ndcg = 0.1
        min_recall = 0.05
        
        if metrics.get('ndcg_10', 0) >= min_ndcg:
            validation_results['should_deploy'] = True
        else:
            validation_results['issues'].append(f"NDCG@10 {metrics.get('ndcg_10', 0):.3f} < {min_ndcg}")
        
        if metrics.get('recall_20', 0) >= min_recall:
            validation_results['should_deploy'] = validation_results['should_deploy'] and True
        else:
            validation_results['should_deploy'] = False
            validation_results['issues'].append(f"Recall@20 {metrics.get('recall_20', 0):.3f} < {min_recall}")
        
        return validation_results
    
    def deploy_models(self, training_results: Dict[str, Any]):
        """Deploy trained models to production"""
        models_path = Path(self.config.models_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models with timestamp
        if 'two_tower_model' in training_results:
            model_path = models_path / f"two_tower_model_{timestamp}.pt"
            # Save model logic here
            logger.info(f"Deployed two-tower model to {model_path}")
        
        if 'lightgbm_model' in training_results:
            model_path = models_path / f"lightgbm_model_{timestamp}.pkl"
            # Save model logic here
            logger.info(f"Deployed LightGBM model to {model_path}")
        
        if 'faiss_index' in training_results:
            index_path = models_path / f"faiss_index_{timestamp}"
            # Save index logic here
            logger.info(f"Deployed FAISS index to {index_path}")
        
        # Update symlinks to latest models
        self._update_production_symlinks(timestamp)
    
    def _update_production_symlinks(self, timestamp: str):
        """Update symlinks to point to latest models"""
        models_path = Path(self.config.models_path)
        
        # Create symlinks to latest models
        symlinks = [
            ('two_tower_model.pt', f'two_tower_model_{timestamp}.pt'),
            ('lightgbm_model.pkl', f'lightgbm_model_{timestamp}.pkl'),
            ('faiss_index', f'faiss_index_{timestamp}')
        ]
        
        for symlink_name, target_name in symlinks:
            symlink_path = models_path / symlink_name
            target_path = models_path / target_name
            
            if target_path.exists():
                if symlink_path.exists() or symlink_path.is_symlink():
                    symlink_path.unlink()
                symlink_path.symlink_to(target_name)
                logger.debug(f"Updated symlink: {symlink_name} -> {target_name}")
    
    def add_ingestion_callback(self, callback: Callable):
        """Add callback for ingestion events"""
        self.ingestion_callbacks.append(callback)
    
    def add_training_callback(self, callback: Callable):
        """Add callback for training events"""
        self.training_callbacks.append(callback)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'is_running': self.is_running,
            'config': {
                'ingestion_frequency': self.config.ingestion_frequency,
                'training_frequency': self.config.training_frequency,
                'source_type': self.config.source_type
            },
            'statistics': self.pipeline_stats.copy(),
            'next_ingestion': self._get_next_scheduled_time('ingestion'),
            'next_training': self._get_next_scheduled_time('training')
        }
    
    def _get_next_scheduled_time(self, task_type: str) -> Optional[str]:
        """Get next scheduled time for task type"""
        try:
            jobs = schedule.jobs
            task_jobs = [job for job in jobs if task_type in str(job.job_func)]
            if task_jobs:
                next_run = min(job.next_run for job in task_jobs)
                return next_run.isoformat() if next_run else None
        except:
            pass
        return None
    
    def trigger_ingestion(self):
        """Manually trigger data ingestion"""
        logger.info("Manually triggering data ingestion...")
        self._run_ingestion()
    
    def trigger_training(self):
        """Manually trigger model training"""
        logger.info("Manually triggering model training...")
        self._run_training()


class DataQualityMonitor:
    """Monitor data quality and alert on issues"""
    
    def __init__(self, pipeline_manager: DataPipelineManager):
        self.pipeline_manager = pipeline_manager
        self.quality_history = []
        self.alert_thresholds = {
            'min_daily_interactions': 1000,
            'max_missing_data_percentage': 0.1,
            'min_unique_users': 50,
            'min_unique_items': 100
        }
    
    def check_data_quality(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data quality check"""
        quality_report = {
            'timestamp': datetime.now(),
            'overall_score': 1.0,
            'issues': [],
            'metrics': {},
            'alerts': []
        }
        
        interactions = processed_data.get('interactions', [])
        
        if interactions:
            df = pd.DataFrame(interactions)
            
            # Check data volume
            daily_interactions = len(df)
            quality_report['metrics']['daily_interactions'] = daily_interactions
            
            if daily_interactions < self.alert_thresholds['min_daily_interactions']:
                quality_report['alerts'].append({
                    'severity': 'warning',
                    'message': f"Low interaction volume: {daily_interactions}"
                })
                quality_report['overall_score'] *= 0.8
            
            # Check for missing data
            missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
            quality_report['metrics']['missing_data_percentage'] = missing_percentage
            
            if missing_percentage > self.alert_thresholds['max_missing_data_percentage']:
                quality_report['alerts'].append({
                    'severity': 'error',
                    'message': f"High missing data: {missing_percentage:.1%}"
                })
                quality_report['overall_score'] *= 0.6
            
            # Check user diversity
            unique_users = df['user_id'].nunique()
            quality_report['metrics']['unique_users'] = unique_users
            
            if unique_users < self.alert_thresholds['min_unique_users']:
                quality_report['alerts'].append({
                    'severity': 'warning',
                    'message': f"Low user diversity: {unique_users}"
                })
                quality_report['overall_score'] *= 0.9
            
            # Check item diversity
            unique_items = df['item_id'].nunique()
            quality_report['metrics']['unique_items'] = unique_items
            
            if unique_items < self.alert_thresholds['min_unique_items']:
                quality_report['alerts'].append({
                    'severity': 'warning',
                    'message': f"Low item diversity: {unique_items}"
                })
                quality_report['overall_score'] *= 0.9
        
        # Store in history
        self.quality_history.append(quality_report)
        
        # Keep only recent history
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]
        
        return quality_report
    
    def get_quality_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get data quality trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [
            report for report in self.quality_history
            if report['timestamp'] > cutoff_date
        ]
        
        if not recent_reports:
            return {'error': 'No recent quality reports'}
        
        # Calculate trends
        scores = [report['overall_score'] for report in recent_reports]
        
        trends = {
            'period_days': days,
            'reports_count': len(recent_reports),
            'average_score': np.mean(scores),
            'score_trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'declining',
            'alert_counts': {
                'error': sum(1 for report in recent_reports for alert in report['alerts'] if alert['severity'] == 'error'),
                'warning': sum(1 for report in recent_reports for alert in report['alerts'] if alert['severity'] == 'warning')
            }
        }
        
        return trends