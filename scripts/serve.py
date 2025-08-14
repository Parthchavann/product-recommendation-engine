#!/usr/bin/env python3
"""
Serving script for the recommendation engine API

Usage:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8080 --workers 4
"""

import argparse
import sys
from pathlib import Path
import uvicorn
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.logger import get_logger
from src.utils.config import get_config_dir


logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description="Serve recommendation API")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to bind to"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/serving_config.yaml",
        help="Serving configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level"
    )
    
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable access logging"
    )
    
    return parser.parse_args()


def load_serving_config(config_path: str) -> dict:
    """Load serving configuration"""
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded serving config from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def main():
    """Main serving function"""
    
    args = parse_args()
    
    # Load serving configuration
    serving_config = load_serving_config(args.config)
    
    # Override with command line arguments
    api_config = serving_config.get('api', {})
    
    host = args.host or api_config.get('host', '0.0.0.0')
    port = args.port or api_config.get('port', 8000)
    workers = args.workers or api_config.get('workers', 1)
    
    logger.info(f"Starting recommendation API server...")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Log level: {args.log_level}")
    
    # Uvicorn configuration
    uvicorn_config = {
        "app": "src.serving.api:app",
        "host": host,
        "port": port,
        "log_level": args.log_level,
        "access_log": args.access_log,
    }
    
    # Add worker configuration for production
    if not args.reload:
        uvicorn_config["workers"] = workers
    
    # Add reload for development
    if args.reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]
    
    try:
        # Start server
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()