import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import os
import zipfile
import requests
from pathlib import Path
import logging

from ..utils.config import Config, get_data_dir
from ..utils.logger import get_logger


logger = get_logger(__name__)


class RecommendationDataset(Dataset):
    """Custom dataset for recommendation engine training"""
    
    def __init__(self, 
                 interactions_df: pd.DataFrame,
                 user_features: Optional[Dict] = None,
                 item_features: Optional[Dict] = None,
                 user_encoder: Optional[LabelEncoder] = None,
                 item_encoder: Optional[LabelEncoder] = None):
        
        self.interactions = interactions_df.copy()
        self.user_features = user_features or {}
        self.item_features = item_features or {}
        
        # Encode categorical features
        if user_encoder is None:
            self.user_encoder = LabelEncoder()
            self.interactions['user_id_encoded'] = self.user_encoder.fit_transform(
                self.interactions['user_id']
            )
        else:
            self.user_encoder = user_encoder
            self.interactions['user_id_encoded'] = self.user_encoder.transform(
                self.interactions['user_id']
            )
        
        if item_encoder is None:
            self.item_encoder = LabelEncoder()
            self.interactions['item_id_encoded'] = self.item_encoder.fit_transform(
                self.interactions['item_id']
            )
        else:
            self.item_encoder = item_encoder
            self.interactions['item_id_encoded'] = self.item_encoder.transform(
                self.interactions['item_id']
            )
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        
        # Get user and item features (default to zeros if not available)
        user_feat = self.user_features.get(row['user_id'], np.zeros(10))
        item_feat = self.item_features.get(row['item_id'], np.zeros(20))
        
        return {
            'user_id': torch.tensor(row['user_id_encoded'], dtype=torch.long),
            'item_id': torch.tensor(row['item_id_encoded'], dtype=torch.long),
            'rating': torch.tensor(row['rating'], dtype=torch.float),
            'user_features': torch.tensor(user_feat, dtype=torch.float),
            'item_features': torch.tensor(item_feat, dtype=torch.float)
        }


def download_movielens_data(data_dir: Path, version: str = "25m") -> Path:
    """
    Download MovieLens dataset
    
    Args:
        data_dir: Directory to save data
        version: Version of dataset (25m, 20m, 1m)
        
    Returns:
        Path to extracted data directory
    """
    
    urls = {
        "25m": "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "20m": "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    }
    
    if version not in urls:
        raise ValueError(f"Version {version} not supported. Choose from {list(urls.keys())}")
    
    url = urls[version]
    filename = f"ml-{version}.zip"
    filepath = data_dir / "raw" / filename
    extract_dir = data_dir / "raw" / f"ml-{version}"
    
    # Create directories
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Download if not exists
    if not filepath.exists():
        logger.info(f"Downloading MovieLens {version} dataset...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {filename}")
    
    # Extract if not exists
    if not extract_dir.exists():
        logger.info(f"Extracting {filename}...")
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(data_dir / "raw")
        
        logger.info(f"Extracted to {extract_dir}")
    
    return extract_dir


def load_movielens_data(data_dir: Path, version: str = "25m") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens dataset
    
    Args:
        data_dir: Data directory
        version: Dataset version
        
    Returns:
        Tuple of (ratings, movies, users) DataFrames
    """
    
    # Download/extract data
    extract_dir = download_movielens_data(data_dir, version)
    
    # Load ratings
    ratings_file = extract_dir / "ratings.csv"
    ratings = pd.read_csv(ratings_file)
    
    # Load movies
    movies_file = extract_dir / "movies.csv"
    movies = pd.read_csv(movies_file)
    
    # Load users (if available)
    users_file = extract_dir / "users.csv"
    if users_file.exists():
        users = pd.read_csv(users_file)
    else:
        # Create dummy user data
        unique_users = ratings['userId'].unique()
        users = pd.DataFrame({
            'userId': unique_users,
            'age': np.random.randint(18, 65, len(unique_users)),
            'gender': np.random.choice(['M', 'F'], len(unique_users)),
            'occupation': np.random.randint(0, 20, len(unique_users))
        })
    
    # Standardize column names
    ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    movies.columns = ['item_id', 'title', 'genres']
    users.columns = ['user_id'] + list(users.columns[1:])
    
    logger.info(f"Loaded MovieLens {version}: {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
    
    return ratings, movies, users


def create_negative_samples(df: pd.DataFrame, 
                          num_negatives: int = 4,
                          seed: int = 42) -> pd.DataFrame:
    """
    Generate negative samples for training
    
    Args:
        df: Positive interactions DataFrame
        num_negatives: Number of negative samples per positive
        seed: Random seed
        
    Returns:
        DataFrame with negative samples
    """
    
    np.random.seed(seed)
    all_items = df['item_id'].unique()
    negative_samples = []
    
    logger.info(f"Generating negative samples with ratio {num_negatives}:1")
    
    for user_id in df['user_id'].unique():
        user_items = set(df[df['user_id'] == user_id]['item_id'].values)
        available_items = list(set(all_items) - user_items)
        
        if len(available_items) == 0:
            continue
        
        num_user_positives = len(user_items)
        num_to_sample = min(
            num_negatives * num_user_positives, 
            len(available_items)
        )
        
        sampled_items = np.random.choice(
            available_items, 
            size=num_to_sample,
            replace=False
        )
        
        for item in sampled_items:
            negative_samples.append({
                'user_id': user_id,
                'item_id': item,
                'rating': 0.0,
                'timestamp': 0
            })
    
    negative_df = pd.DataFrame(negative_samples)
    logger.info(f"Generated {len(negative_df)} negative samples")
    
    return negative_df


def filter_data(ratings_df: pd.DataFrame,
               min_user_interactions: int = 5,
               min_item_interactions: int = 10) -> pd.DataFrame:
    """
    Filter data to remove sparse users and items
    
    Args:
        ratings_df: Ratings DataFrame
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        
    Returns:
        Filtered DataFrame
    """
    
    logger.info(f"Before filtering: {len(ratings_df)} interactions")
    
    # Iterative filtering
    prev_size = 0
    current_size = len(ratings_df)
    
    while prev_size != current_size:
        prev_size = current_size
        
        # Filter users
        user_counts = ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
        
        # Filter items
        item_counts = ratings_df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        ratings_df = ratings_df[ratings_df['item_id'].isin(valid_items)]
        
        current_size = len(ratings_df)
        logger.info(f"Filtering iteration: {current_size} interactions remaining")
    
    logger.info(f"After filtering: {len(ratings_df)} interactions")
    
    return ratings_df


def create_user_features(users_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Create user feature vectors
    
    Args:
        users_df: Users DataFrame
        
    Returns:
        Dictionary mapping user_id to feature vector
    """
    
    user_features = {}
    
    for _, row in users_df.iterrows():
        features = []
        
        # Age (normalized)
        features.append(row.get('age', 30) / 100.0)
        
        # Gender (one-hot)
        gender = row.get('gender', 'M')
        features.extend([1.0 if gender == 'M' else 0.0, 1.0 if gender == 'F' else 0.0])
        
        # Occupation (normalized)
        features.append(row.get('occupation', 0) / 20.0)
        
        # Pad to 10 dimensions
        while len(features) < 10:
            features.append(0.0)
        
        user_features[row['user_id']] = np.array(features[:10])
    
    return user_features


def create_item_features(movies_df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Create item feature vectors from genres
    
    Args:
        movies_df: Movies DataFrame
        
    Returns:
        Dictionary mapping item_id to feature vector
    """
    
    # Get all unique genres
    all_genres = set()
    for genres_str in movies_df['genres'].fillna(''):
        if genres_str and genres_str != '(no genres listed)':
            all_genres.update(genres_str.split('|'))
    
    all_genres = sorted(list(all_genres))
    genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
    
    item_features = {}
    
    for _, row in movies_df.iterrows():
        # Genre features (multi-hot encoding)
        features = np.zeros(len(all_genres))
        
        genres_str = row.get('genres', '')
        if genres_str and genres_str != '(no genres listed)':
            genres = genres_str.split('|')
            for genre in genres:
                if genre in genre_to_idx:
                    features[genre_to_idx[genre]] = 1.0
        
        # Pad or truncate to 20 dimensions
        if len(features) > 20:
            features = features[:20]
        else:
            padded_features = np.zeros(20)
            padded_features[:len(features)] = features
            features = padded_features
        
        item_features[row['item_id']] = features
    
    return item_features


def prepare_data(config: Config) -> Tuple[Dataset, Dataset, Dataset, LabelEncoder, LabelEncoder]:
    """
    Prepare training, validation, and test datasets
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, user_encoder, item_encoder)
    """
    
    data_dir = get_data_dir()
    
    # Load raw data
    if config.data.dataset.startswith('movielens'):
        version = config.data.dataset.split('-')[-1]
        ratings, movies, users = load_movielens_data(data_dir, version)
    else:
        raise ValueError(f"Dataset {config.data.dataset} not supported")
    
    # Filter data
    ratings = filter_data(
        ratings, 
        config.data.min_user_interactions,
        config.data.min_item_interactions
    )
    
    # Convert ratings to binary (like/dislike)
    ratings['rating'] = (ratings['rating'] >= 4.0).astype(float)
    positive_ratings = ratings[ratings['rating'] == 1.0]
    
    # Generate negative samples
    negative_ratings = create_negative_samples(
        positive_ratings, 
        config.data.negative_sampling_ratio
    )
    
    # Combine positive and negative samples
    all_interactions = pd.concat([positive_ratings, negative_ratings], ignore_index=True)
    all_interactions = all_interactions.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create feature dictionaries
    user_features = create_user_features(users)
    item_features = create_item_features(movies)
    
    # Train/val/test split
    train_ratio = config.data.train_ratio
    val_ratio = config.data.val_ratio
    
    n = len(all_interactions)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    train_df = all_interactions[:train_idx]
    val_df = all_interactions[train_idx:val_idx]
    test_df = all_interactions[val_idx:]
    
    # Create encoders from training data
    user_encoder = LabelEncoder()
    user_encoder.fit(train_df['user_id'])
    
    item_encoder = LabelEncoder()
    item_encoder.fit(train_df['item_id'])
    
    # Filter val/test to only include known users/items
    known_users = set(user_encoder.classes_)
    known_items = set(item_encoder.classes_)
    
    val_df = val_df[
        val_df['user_id'].isin(known_users) & 
        val_df['item_id'].isin(known_items)
    ]
    
    test_df = test_df[
        test_df['user_id'].isin(known_users) & 
        test_df['item_id'].isin(known_items)
    ]
    
    # Create datasets
    train_dataset = RecommendationDataset(
        train_df, user_features, item_features, user_encoder, item_encoder
    )
    
    val_dataset = RecommendationDataset(
        val_df, user_features, item_features, user_encoder, item_encoder
    )
    
    test_dataset = RecommendationDataset(
        test_df, user_features, item_features, user_encoder, item_encoder
    )
    
    logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    logger.info(f"Users: {len(user_encoder.classes_)}, Items: {len(item_encoder.classes_)}")
    
    return train_dataset, val_dataset, test_dataset, user_encoder, item_encoder


def create_data_loaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Dataset,
                       config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        config: Configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader