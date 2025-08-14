import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from ..utils.logger import get_logger
from ..utils.config import Config


logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


class CTRRanker:
    """LightGBM-based CTR prediction for re-ranking recommendations"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config
        self.model = None
        self.feature_names = []
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = None
        
        # Model parameters
        if config and hasattr(config, 'ctr'):
            self.params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': config.ctr.boosting_type,
                'num_leaves': config.ctr.num_leaves,
                'learning_rate': config.ctr.learning_rate,
                'feature_fraction': config.ctr.feature_fraction,
                'bagging_fraction': config.ctr.bagging_fraction,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': 4,
                'force_row_wise': True
            }
            self.num_boost_round = config.ctr.num_boost_round
        else:
            # Default parameters
            self.params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'num_threads': 4,
                'force_row_wise': True
            }
            self.num_boost_round = 100
        
        logger.info("Initialized CTR Ranker with LightGBM")
    
    def prepare_features(self,
                        interactions_df: pd.DataFrame,
                        users_df: pd.DataFrame,
                        items_df: pd.DataFrame,
                        user_histories: Dict[int, Dict],
                        item_stats: Dict[int, Dict],
                        contextual_features: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepare comprehensive features for CTR prediction
        
        Args:
            interactions_df: User-item interactions
            users_df: User demographics  
            items_df: Item metadata
            user_histories: User historical features
            item_stats: Item statistical features
            contextual_features: Additional contextual features
            
        Returns:
            Feature DataFrame ready for training
        """
        
        logger.info("Preparing CTR features...")
        
        feature_data = []
        
        for _, interaction in interactions_df.iterrows():
            user_id = interaction['user_id']
            item_id = interaction['item_id']
            
            feature_row = {
                'user_id': user_id,
                'item_id': item_id,
                'target': interaction.get('clicked', interaction.get('rating', 0) >= 4.0)
            }
            
            # User features
            user_info = users_df[users_df['user_id'] == user_id]
            if len(user_info) > 0:
                user_row = user_info.iloc[0]
                feature_row.update(self._extract_user_features(user_row))
            
            # User history features
            if user_id in user_histories:
                feature_row.update(self._extract_user_history_features(user_histories[user_id]))
            
            # Item features
            item_info = items_df[items_df['item_id'] == item_id]
            if len(item_info) > 0:
                item_row = item_info.iloc[0]
                feature_row.update(self._extract_item_features(item_row))
            
            # Item statistics
            if item_id in item_stats:
                feature_row.update(self._extract_item_stats(item_stats[item_id]))
            
            # Contextual features
            if contextual_features and user_id in contextual_features:
                feature_row.update(self._extract_contextual_features(contextual_features[user_id]))
            
            # Cross features
            feature_row.update(self._extract_cross_features(
                user_histories.get(user_id, {}), 
                item_stats.get(item_id, {}),
                user_info.iloc[0] if len(user_info) > 0 else {},
                item_info.iloc[0] if len(item_info) > 0 else {}
            ))
            
            feature_data.append(feature_row)
        
        features_df = pd.DataFrame(feature_data)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Prepared {len(features_df)} samples with {len(features_df.columns)-3} features")
        
        return features_df
    
    def _extract_user_features(self, user_row: pd.Series) -> Dict:
        """Extract user demographic features"""
        
        features = {}
        
        # Age (normalized)
        if 'age' in user_row:
            features['user_age'] = user_row['age'] / 100.0
            features['user_age_bucket'] = int(user_row['age'] // 10)
        
        # Gender
        if 'gender' in user_row:
            features['user_gender_M'] = 1.0 if user_row['gender'] == 'M' else 0.0
            features['user_gender_F'] = 1.0 if user_row['gender'] == 'F' else 0.0
        
        # Occupation (if available)
        if 'occupation' in user_row:
            features['user_occupation'] = user_row['occupation']
        
        return features
    
    def _extract_user_history_features(self, user_history: Dict) -> Dict:
        """Extract user historical behavior features"""
        
        features = {}
        
        # Basic interaction statistics
        features['user_total_interactions'] = user_history.get('total_interactions', 0)
        features['user_positive_interactions'] = user_history.get('positive_interactions', 0)
        features['user_avg_rating'] = user_history.get('avg_rating', 0)
        features['user_rating_std'] = user_history.get('rating_std', 0)
        
        # Temporal features
        features['user_days_since_last_interaction'] = user_history.get('days_since_last', 0)
        features['user_avg_session_length'] = user_history.get('avg_session_length', 0)
        features['user_interactions_per_day'] = user_history.get('interactions_per_day', 0)
        
        # Diversity and exploration
        features['user_unique_categories'] = user_history.get('unique_categories', 0)
        features['user_category_diversity'] = user_history.get('category_diversity', 0)
        features['user_exploration_rate'] = user_history.get('exploration_rate', 0)
        
        # Engagement patterns
        features['user_avg_hour'] = user_history.get('avg_hour', 12) / 24.0
        features['user_weekend_ratio'] = user_history.get('weekend_ratio', 0)
        features['user_mobile_ratio'] = user_history.get('mobile_ratio', 0)
        
        # CTR and conversion
        features['user_historical_ctr'] = user_history.get('ctr', 0)
        features['user_bounce_rate'] = user_history.get('bounce_rate', 0)
        
        return features
    
    def _extract_item_features(self, item_row: pd.Series) -> Dict:
        """Extract item metadata features"""
        
        features = {}
        
        # Basic item info
        if 'title' in item_row:
            features['item_title_length'] = len(str(item_row['title']))
        
        # Genres (one-hot encoding for major genres)
        if 'genres' in item_row and pd.notna(item_row['genres']):
            genres = str(item_row['genres']).split('|')
            
            # Major genres
            major_genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Horror', 'Sci-Fi']
            for genre in major_genres:
                features[f'item_genre_{genre}'] = 1.0 if genre in genres else 0.0
            
            features['item_num_genres'] = len(genres)
        
        # Release year (if available)
        if 'year' in item_row and pd.notna(item_row['year']):
            current_year = 2023
            features['item_age_years'] = current_year - int(item_row['year'])
            features['item_is_recent'] = 1.0 if features['item_age_years'] <= 2 else 0.0
        
        return features
    
    def _extract_item_stats(self, item_stats: Dict) -> Dict:
        """Extract item statistical features"""
        
        features = {}
        
        # Popularity metrics
        features['item_total_interactions'] = item_stats.get('total_interactions', 0)
        features['item_unique_users'] = item_stats.get('unique_users', 0)
        features['item_avg_rating'] = item_stats.get('avg_rating', 0)
        features['item_rating_std'] = item_stats.get('rating_std', 0)
        
        # Popularity percentile
        features['item_popularity_percentile'] = item_stats.get('popularity_percentile', 0)
        features['item_is_popular'] = 1.0 if features['item_popularity_percentile'] >= 0.8 else 0.0
        
        # CTR metrics
        features['item_ctr'] = item_stats.get('ctr', 0)
        features['item_conversion_rate'] = item_stats.get('conversion_rate', 0)
        
        # Temporal patterns
        features['item_days_since_last_interaction'] = item_stats.get('days_since_last', 0)
        features['item_interaction_velocity'] = item_stats.get('interaction_velocity', 0)
        
        return features
    
    def _extract_contextual_features(self, contextual: Dict) -> Dict:
        """Extract contextual features"""
        
        features = {}
        
        # Time features
        features['context_hour'] = contextual.get('hour', 12) / 24.0
        features['context_day_of_week'] = contextual.get('day_of_week', 0) / 7.0
        features['context_is_weekend'] = contextual.get('is_weekend', 0)
        
        # Device and platform
        device_type = contextual.get('device_type', 'desktop')
        features['context_device_mobile'] = 1.0 if device_type == 'mobile' else 0.0
        features['context_device_tablet'] = 1.0 if device_type == 'tablet' else 0.0
        features['context_device_desktop'] = 1.0 if device_type == 'desktop' else 0.0
        
        # Session context
        features['context_session_position'] = contextual.get('session_position', 0)
        features['context_session_length'] = contextual.get('session_length', 0)
        
        return features
    
    def _extract_cross_features(self,
                               user_history: Dict,
                               item_stats: Dict, 
                               user_info: Dict,
                               item_info: Dict) -> Dict:
        """Extract cross/interaction features"""
        
        features = {}
        
        # User-item affinity
        user_avg_rating = user_history.get('avg_rating', 0)
        item_avg_rating = item_stats.get('avg_rating', 0)
        features['user_item_rating_affinity'] = user_avg_rating * item_avg_rating
        
        # Popularity vs user exploration
        item_popularity = item_stats.get('popularity_percentile', 0)
        user_exploration = user_history.get('exploration_rate', 0)
        features['popularity_exploration_cross'] = item_popularity * (1 - user_exploration)
        
        # User activity vs item popularity
        user_activity = user_history.get('interactions_per_day', 0)
        features['activity_popularity_cross'] = np.log1p(user_activity) * item_popularity
        
        # Age vs content
        if 'age' in user_info and not pd.isna(user_info.get('age')):
            user_age = user_info['age']
            
            # Age-appropriate content (example heuristic)
            if 'genres' in item_info and pd.notna(item_info['genres']):
                genres = str(item_info['genres']).split('|')
                young_adult_genres = ['Action', 'Comedy', 'Romance']
                features['age_content_match'] = 1.0 if (
                    user_age <= 30 and any(g in young_adult_genres for g in genres)
                ) else 0.0
        
        # CTR similarity
        user_ctr = user_history.get('ctr', 0)
        item_ctr = item_stats.get('ctr', 0)
        features['user_item_ctr_similarity'] = abs(user_ctr - item_ctr)
        
        return features
    
    def train(self,
              train_df: pd.DataFrame,
              val_df: Optional[pd.DataFrame] = None,
              categorical_features: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train the CTR prediction model
        
        Args:
            train_df: Training data with features and target
            val_df: Validation data (optional)
            categorical_features: List of categorical feature names
            
        Returns:
            Training metrics
        """
        
        logger.info("Training CTR model...")
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in ['user_id', 'item_id', 'target']]
        X_train = train_df[feature_cols]
        y_train = train_df['target'].astype(int)
        
        self.feature_names = feature_cols
        
        # Process categorical features
        if categorical_features:
            cat_indices = []
            for cat_feat in categorical_features:
                if cat_feat in feature_cols:
                    cat_indices.append(feature_cols.index(cat_feat))
        else:
            cat_indices = None
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=feature_cols,
            categorical_feature=cat_indices
        )
        
        # Validation data
        valid_sets = [train_data]
        valid_names = ['train']
        
        if val_df is not None:
            X_val = val_df[feature_cols]
            y_val = val_df['target'].astype(int)
            
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=feature_cols,
                categorical_feature=cat_indices
            )
            
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=False),
                lgb.log_evaluation(period=0)  # Disable verbose logging
            ]
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(
            self.model.feature_name(),
            self.model.feature_importance()
        ))
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_auc = self._calculate_auc(y_train, train_pred)
        train_logloss = self._calculate_logloss(y_train, train_pred)
        
        metrics = {
            'train_auc': train_auc,
            'train_logloss': train_logloss
        }
        
        if val_df is not None:
            val_pred = self.model.predict(X_val)
            val_auc = self._calculate_auc(y_val, val_pred)
            val_logloss = self._calculate_logloss(y_val, val_pred)
            
            metrics.update({
                'val_auc': val_auc,
                'val_logloss': val_logloss
            })
        
        logger.info(f"Training completed. Train AUC: {train_auc:.4f}")
        if val_df is not None:
            logger.info(f"Validation AUC: {val_auc:.4f}")
        
        return metrics
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict CTR scores
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            Array of CTR predictions
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Select feature columns
        X = features_df[self.feature_names]
        
        # Predict
        predictions = self.model.predict(X)
        
        return predictions
    
    def rerank(self,
               candidates: List[int],
               user_id: int,
               user_features: Dict,
               item_features_map: Dict[int, Dict],
               context: Dict,
               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Re-rank candidates based on CTR prediction
        
        Args:
            candidates: List of candidate item IDs
            user_id: User ID
            user_features: User feature dictionary
            item_features_map: Mapping from item_id to features
            context: Contextual features
            top_k: Number of items to return
            
        Returns:
            List of (item_id, ctr_score) tuples, ranked by CTR
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not candidates:
            return []
        
        # Prepare features for all candidates
        feature_rows = []
        
        for item_id in candidates:
            feature_row = {'user_id': user_id, 'item_id': item_id}
            
            # User features
            feature_row.update(self._extract_user_features(pd.Series(user_features)))
            feature_row.update(self._extract_user_history_features(user_features))
            
            # Item features
            item_features = item_features_map.get(item_id, {})
            feature_row.update(self._extract_item_features(pd.Series(item_features)))
            feature_row.update(self._extract_item_stats(item_features))
            
            # Context features
            feature_row.update(self._extract_contextual_features(context))
            
            # Cross features
            feature_row.update(self._extract_cross_features(
                user_features, item_features, user_features, item_features
            ))
            
            feature_rows.append(feature_row)
        
        # Create DataFrame
        features_df = pd.DataFrame(feature_rows).fillna(0)
        
        # Predict CTR scores
        ctr_scores = self.predict(features_df)
        
        # Create ranked list
        ranked_items = list(zip(candidates, ctr_scores))
        ranked_items.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_items[:top_k]
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, float]:
        """
        Get top feature importance scores
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            Dictionary of feature importance scores
        """
        
        if self.feature_importance is None:
            return {}
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_k])
    
    def save_model(self, model_path: str):
        """Save trained model to disk"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'params': self.params,
            'feature_importance': self.feature_importance
        }
        
        metadata_path = path.with_suffix('.metadata')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Saved CTR model to {path}")
    
    def load_model(self, model_path: str):
        """Load trained model from disk"""
        
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(path))
        
        # Load metadata
        metadata_path = path.with_suffix('.metadata')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata.get('feature_names', [])
            self.params = metadata.get('params', {})
            self.feature_importance = metadata.get('feature_importance', {})
        
        logger.info(f"Loaded CTR model from {path}")
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUC score"""
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    
    def _calculate_logloss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate log loss"""
        try:
            from sklearn.metrics import log_loss
            return log_loss(y_true, y_pred)
        except:
            return 1.0


def create_user_item_features(interactions_df: pd.DataFrame,
                             users_df: pd.DataFrame,
                             items_df: pd.DataFrame,
                             window_days: int = 30) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Create user history and item statistics for CTR feature engineering
    
    Args:
        interactions_df: User-item interactions
        users_df: User demographics
        items_df: Item metadata  
        window_days: Time window for recent interactions
        
    Returns:
        Tuple of (user_histories, item_stats)
    """
    
    logger.info("Creating user and item features for CTR model...")
    
    # User histories
    user_histories = {}
    
    for user_id in interactions_df['user_id'].unique():
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        # Basic statistics
        total_interactions = len(user_interactions)
        positive_interactions = len(user_interactions[user_interactions['rating'] >= 4.0])
        avg_rating = user_interactions['rating'].mean()
        rating_std = user_interactions['rating'].std() if len(user_interactions) > 1 else 0
        
        # Temporal features
        if 'timestamp' in user_interactions.columns:
            last_interaction = user_interactions['timestamp'].max()
            days_since_last = (pd.Timestamp.now() - pd.to_datetime(last_interaction)).days
        else:
            days_since_last = 0
        
        # Diversity metrics
        unique_items = user_interactions['item_id'].nunique()
        
        # CTR calculation
        ctr = positive_interactions / total_interactions if total_interactions > 0 else 0
        
        user_histories[user_id] = {
            'total_interactions': total_interactions,
            'positive_interactions': positive_interactions,
            'avg_rating': avg_rating,
            'rating_std': rating_std,
            'days_since_last': days_since_last,
            'unique_items': unique_items,
            'ctr': ctr,
            'interactions_per_day': total_interactions / max(1, days_since_last),
            'exploration_rate': unique_items / max(1, total_interactions)
        }
    
    # Item statistics
    item_stats = {}
    
    for item_id in interactions_df['item_id'].unique():
        item_interactions = interactions_df[interactions_df['item_id'] == item_id]
        
        # Basic statistics
        total_interactions = len(item_interactions)
        unique_users = item_interactions['user_id'].nunique()
        avg_rating = item_interactions['rating'].mean()
        rating_std = item_interactions['rating'].std() if len(item_interactions) > 1 else 0
        
        # Popularity percentile
        popularity_percentile = (total_interactions - interactions_df.groupby('item_id').size().min()) / \
                              (interactions_df.groupby('item_id').size().max() - interactions_df.groupby('item_id').size().min())
        
        # CTR
        positive_interactions = len(item_interactions[item_interactions['rating'] >= 4.0])
        ctr = positive_interactions / total_interactions if total_interactions > 0 else 0
        
        item_stats[item_id] = {
            'total_interactions': total_interactions,
            'unique_users': unique_users,
            'avg_rating': avg_rating,
            'rating_std': rating_std,
            'popularity_percentile': popularity_percentile,
            'ctr': ctr,
            'conversion_rate': ctr  # Assuming CTR = conversion rate for simplicity
        }
    
    logger.info(f"Created features for {len(user_histories)} users and {len(item_stats)} items")
    
    return user_histories, item_stats