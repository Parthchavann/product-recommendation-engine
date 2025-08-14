import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger
    """
    
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logger.setLevel(numeric_level)
    
    # Default format
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs
        
    Returns:
        Configured experiment logger
    """
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{experiment_name}_{timestamp}.log"
    
    return setup_logger(
        name=experiment_name,
        log_file=str(log_file),
        level="INFO"
    )


class ExperimentTracker:
    """Simple experiment tracking utility"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.logger = get_experiment_logger(experiment_name, log_dir)
        self.start_time = datetime.now()
        
        self.logger.info(f"Starting experiment: {experiment_name}")
    
    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, step: int, metrics: dict):
        """Log metrics for a given step"""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metric_str}")
    
    def log_model_save(self, model_path: str, metrics: dict):
        """Log model save event"""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Saved model: {model_path} | {metric_str}")
    
    def log_completion(self, final_metrics: dict):
        """Log experiment completion"""
        duration = datetime.now() - self.start_time
        
        self.logger.info(f"Experiment {self.experiment_name} completed!")
        self.logger.info(f"Total duration: {duration}")
        self.logger.info("Final metrics:")
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")


# Global logger instance
_logger = None


def get_logger(name: str = "recommendation-engine") -> logging.Logger:
    """Get or create global logger instance"""
    global _logger
    
    if _logger is None:
        _logger = setup_logger(name, level="INFO")
    
    return _logger