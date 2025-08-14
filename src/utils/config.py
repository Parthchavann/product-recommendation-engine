import yaml
import os
from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    name: str = "two_tower_v1"
    embedding_dim: int = 128
    tower_dims: list = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    temperature: float = 0.07


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    dataset: str = "movielens-25m"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    negative_sampling_ratio: int = 4
    min_user_interactions: int = 5
    min_item_interactions: int = 10


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10


@dataclass
class ContentConfig:
    """Content embedding configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 32


@dataclass
class FAISSConfig:
    """FAISS configuration"""
    index_type: str = "IVF"
    dimension: int = 64
    nlist: int = 100
    metric: str = "inner_product"


@dataclass
class CTRConfig:
    """CTR model configuration"""
    model_type: str = "lightgbm"
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    num_boost_round: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration"""
    wandb_project: str = "recommendation-engine"
    log_interval: int = 100
    save_model_every: int = 5


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    ctr: CTRConfig = field(default_factory=CTRConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: str = "cuda"
    num_workers: int = 4


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Using default configuration.")
        return Config()
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Create config objects from dictionary
    config = Config()
    
    if 'model' in config_dict:
        config.model = ModelConfig(**config_dict['model'])
    
    if 'data' in config_dict:
        config.data = DataConfig(**config_dict['data'])
    
    if 'training' in config_dict:
        config.training = TrainingConfig(**config_dict['training'])
    
    if 'content' in config_dict:
        config.content = ContentConfig(**config_dict['content'])
    
    if 'faiss' in config_dict:
        config.faiss = FAISSConfig(**config_dict['faiss'])
    
    if 'ctr' in config_dict:
        config.ctr = CTRConfig(**config_dict['ctr'])
    
    if 'logging' in config_dict:
        config.logging = LoggingConfig(**config_dict['logging'])
    
    if 'device' in config_dict:
        config.device = config_dict['device']
    
    if 'num_workers' in config_dict:
        config.num_workers = config_dict['num_workers']
    
    return config


def get_project_root() -> Path:
    """Get the root directory of the project"""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory path"""
    return get_project_root() / "data"


def get_model_dir() -> Path:
    """Get the models directory path"""
    return get_project_root() / "models"


def get_config_dir() -> Path:
    """Get the configs directory path"""
    return get_project_root() / "configs"


def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)


def save_config(config: Config, config_path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'model': {
            'name': config.model.name,
            'embedding_dim': config.model.embedding_dim,
            'tower_dims': config.model.tower_dims,
            'dropout': config.model.dropout,
            'temperature': config.model.temperature,
        },
        'data': {
            'dataset': config.data.dataset,
            'train_ratio': config.data.train_ratio,
            'val_ratio': config.data.val_ratio,
            'test_ratio': config.data.test_ratio,
            'negative_sampling_ratio': config.data.negative_sampling_ratio,
            'min_user_interactions': config.data.min_user_interactions,
            'min_item_interactions': config.data.min_item_interactions,
        },
        'training': {
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'scheduler': config.training.scheduler,
            'gradient_clip': config.training.gradient_clip,
            'early_stopping_patience': config.training.early_stopping_patience,
        },
        'device': config.device,
        'num_workers': config.num_workers,
    }
    
    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)