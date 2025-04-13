"""
Amazon Product Data Loading Module

This module handles loading the Amazon product dataset with proper error handling,
caching, and configuration management.
"""

import os
import logging
import pandas as pd
from typing import Optional, Dict, Any, Union
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amazon_chatbot.data_loader')

# Default configuration
DEFAULT_CONFIG = {
    'data_path': 'data/amazon_products.csv',
    'cache_enabled': True,
    'cache_ttl': 3600,  # Time to live in seconds (1 hour)
    'chunk_size': None  # For large files, set to a value like 100000
}

# Cache storage
_data_cache = {}

def load_config(config_path: str = 'config/data_config.json') -> Dict[str, Any]:
    """
    Load configuration from a JSON file or use defaults if file doesn't exist.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return {**DEFAULT_CONFIG, **config}  # Merge with defaults
        else:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return DEFAULT_CONFIG

def load_amazon_data(
    file_path: Optional[str] = None,
    use_cache: bool = True,
    config_path: str = 'config/data_config.json'
) -> Optional[pd.DataFrame]:
    """
    Load Amazon product data from CSV file with caching support.
    
    Args:
        file_path: Path to the CSV file (overrides config)
        use_cache: Whether to use cached data if available
        config_path: Path to the configuration file
        
    Returns:
        Pandas DataFrame containing the Amazon product data or None if loading fails
    """
    # Load configuration
    config = load_config(config_path)
    
    # Use provided file path or default from config
    data_path = file_path or config['data_path']
    cache_enabled = config['cache_enabled'] and use_cache
    chunk_size = config['chunk_size']
    
    # Check if data is in cache and cache is enabled
    cache_key = f"data_{data_path}"
    if cache_enabled and cache_key in _data_cache:
        cache_entry = _data_cache[cache_key]
        # Check if cache is still valid
        if time.time() - cache_entry['timestamp'] < config['cache_ttl']:
            logger.info(f"Using cached data from {data_path}")
            return cache_entry['data']
        else:
            logger.info("Cache expired, reloading data")
    
    # Load the data
    try:
        start_time = time.time()
        
        if not os.path.exists(data_path):
            logger.error(f"File not found: {data_path}")
            return None
        
        # Check file size to determine if chunking is needed
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # Size in MB
        
        if chunk_size and file_size > 100:  # For files larger than 100MB
            logger.info(f"Loading large file ({file_size:.2f} MB) using chunking")
            chunks = []
            for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
        else:
            data = pd.read_csv(data_path)
        
        load_time = time.time() - start_time
        logger.info(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns in {load_time:.2f} seconds")
        
        # Store in cache if caching is enabled
        if cache_enabled:
            _data_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            logger.info("Data stored in cache")
        
        return data
    
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {data_path}")
        return None
    except pd.errors.ParserError:
        logger.error(f"Error parsing CSV file: {data_path}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        return None

def clear_cache():
    """Clear the data cache"""
    global _data_cache
    _data_cache = {}
    logger.info("Data cache cleared")

# Simple data validation function
def validate_amazon_data(data: pd.DataFrame) -> bool:
    """
    Validate the loaded Amazon product data.
    
    Args:
        data: Pandas DataFrame to validate
        
    Returns:
        Boolean indicating whether the data is valid
    """
    if data is None:
        return False
    
    # Check for required columns
    required_columns = ['asin', 'title', 'price']
    if not all(col in data.columns for col in required_columns):
        logger.error(f"Missing required columns. Required: {required_columns}")
        return False
    
    # Check for minimum data size
    if len(data) == 0:
        logger.error("Dataset is empty")
        return False
    
    return True

# Test the function
if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('config', exist_ok=True)
    
    # If config doesn't exist, create a default one
    if not os.path.exists('config/data_config.json'):
        with open('config/data_config.json', 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    
    # Load and validate data
    data = load_amazon_data()
    if data is not None and validate_amazon_data(data):
        print("Data validation successful!")
        print(f"Sample data:\n{data.head()}")