"""
Amazon Product Retrieval Module

This module handles retrieving relevant products from the FAISS index
based on user queries, with support for filtering and ranking.
"""

import os
import logging
import numpy as np
import pandas as pd
import faiss
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amazon_chatbot.product_retriever')

# Cache for model and index to avoid reloading
_cache = {
    'model': None,
    'index': None,
    'data': None,
    'last_load_time': 0
}

# Default configuration
DEFAULT_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'index_file': 'embeddings/faiss_index',
    'data_file': 'data/amazon_products_processed.csv',
    'cache_ttl': 3600,  # Time to live in seconds (1 hour)
    'top_k': 50,  # Retrieve more candidates initially for filtering
    'rerank_method': 'hybrid',  # Options: 'distance', 'reviews', 'price', 'hybrid'
    'nprobe': 10,  # Number of clusters to probe during search
    'use_gpu': False,  # Whether to use GPU for embedding generation
    'price_weight': 0.2,  # Weight for price in hybrid ranking
    'rating_weight': 0.3,  # Weight for ratings in hybrid ranking
    'relevance_weight': 0.5,  # Weight for semantic relevance in hybrid ranking
}

def load_config(config_path: str = 'config/retrieval_config.json') -> Dict[str, Any]:
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

def setup_model(model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False) -> SentenceTransformer:
    """
    Set up the SentenceTransformer model with appropriate configuration.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        use_gpu: Whether to use GPU for embedding generation
        
    Returns:
        SentenceTransformer model
    """
    start_time = time.time()
    
    # Check if model is already cached
    if _cache['model'] is not None and _cache['model'].__class__.__name__ == 'SentenceTransformer':
        logger.info("Using cached SentenceTransformer model")
        return _cache['model']
    
    try:
        # Initialize the model with appropriate device
        device = 'cuda' if use_gpu else 'cpu'
        model = SentenceTransformer(model_name, device=device)
        
        # Cache the model
        _cache['model'] = model
        
        logger.info(f"Model '{model_name}' loaded in {time.time() - start_time:.2f} seconds")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_faiss_index(
    index_file: str = 'embeddings/faiss_index',
    force_reload: bool = False
) -> faiss.Index:
    """
    Load the FAISS index with caching for better performance.
    
    Args:
        index_file: Path to the FAISS index file
        force_reload: Whether to force reloading the index
        
    Returns:
        FAISS index
    """
    # Check if index is already cached
    if not force_reload and _cache['index'] is not None and time.time() - _cache['last_load_time'] < 3600:
        logger.info("Using cached FAISS index")
        return _cache['index']
    
    start_time = time.time()
    
    try:
        # Load the index
        index = faiss.read_index(index_file)
        
        # Load metadata if available
        metadata_file = f"{os.path.splitext(index_file)[0]}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Set nprobe parameter if available in IVF indices
            if hasattr(index, 'nprobe') and 'config' in metadata and 'nprobe' in metadata['config']:
                index.nprobe = metadata['config']['nprobe']
                logger.info(f"Set nprobe={index.nprobe} from metadata")
        
        # Cache the index
        _cache['index'] = index
        _cache['last_load_time'] = time.time()
        
        logger.info(f"FAISS index loaded from {index_file} in {time.time() - start_time:.2f} seconds")
        logger.info(f"Index contains {index.ntotal} vectors")
        
        return index
    
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        raise

def load_product_data(
    data_file: str = 'data/amazon_products_processed.csv',
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Load the product data with caching for better performance.
    
    Args:
        data_file: Path to the product data file
        force_reload: Whether to force reloading the data
        
    Returns:
        DataFrame containing product data
    """
    # Check if data is already cached
    if not force_reload and _cache['data'] is not None and time.time() - _cache['last_load_time'] < 3600:
        logger.info("Using cached product data")
        return _cache['data']
    
    start_time = time.time()
    
    try:
        # Load the data
        data = pd.read_csv(data_file)
        
        # Cache the data
        _cache['data'] = data
        
        logger.info(f"Product data loaded from {data_file} in {time.time() - start_time:.2f} seconds")
        logger.info(f"Loaded {len(data)} products with {len(data.columns)} attributes")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        raise

def preprocess_query(query: str) -> str:
    """
    Preprocess the user query for better matching.
    
    Args:
        query: User query string
        
    Returns:
        Preprocessed query string
    """
    # Convert to lowercase
    query = query.lower()
    
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    # Extract price range if present
    price_pattern = r'\$(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)'
    price_matches = re.findall(price_pattern, query)
    
    # Remove price range from query if found
    if price_matches:
        query = re.sub(price_pattern, '', query).strip()
    
    return query

def extract_filters(query: str) -> Dict[str, Any]:
    """
    Extract filtering criteria from the user query.
    
    Args:
        query: User query string
        
    Returns:
        Dictionary of filter criteria
    """
    filters = {}
    
    # Extract price range
    price_pattern = r'\$(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)'
    price_matches = re.findall(price_pattern, query)
    
    if price_matches:
        min_price = float(price_matches[0][0])
        max_price = float(price_matches[0][1])
        filters['price_range'] = (min_price, max_price)
    
    # Extract brand preferences (if mentioned)
    brand_pattern = r'brand:?\s*([a-zA-Z0-9_\s]+)'
    brand_matches = re.findall(brand_pattern, query.lower())
    
    if brand_matches:
        filters['brand'] = brand_matches[0].strip()
    
    # Extract rating threshold
    rating_pattern = r'(\d+(?:\.\d+)?)\+?\s*stars?'
    rating_matches = re.findall(rating_pattern, query.lower())
    
    if rating_matches:
        filters['min_rating'] = float(rating_matches[0])
    
    return filters

def retrieve_similar_products(
    query: str,
    index: Optional[faiss.Index] = None,
    data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    return_distances: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], np.ndarray]]:
    """
    Retrieve similar products based on the user query.
    
    Args:
        query: User query string
        index: FAISS index (if None, will be loaded)
        data: Product data DataFrame (if None, will be loaded)
        config: Configuration dictionary (if None, default config will be used)
        top_k: Number of products to retrieve (overrides config)
        filters: Dictionary of filtering criteria (overrides extraction from query)
        return_distances: Whether to return distance scores
        
    Returns:
        List of dictionaries containing product information
        If return_distances is True, also returns numpy array of distances
    """
    start_time = time.time()
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Set top_k if not provided
    if top_k is None:
        top_k = config['top_k']
    
    # Load resources if not provided
    if index is None:
        index = load_faiss_index(config['index_file'])
    
    if data is None:
        data = load_product_data(config['data_file'])
    
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    # Extract filters if not provided
    if filters is None:
        filters = extract_filters(query)
        logger.info(f"Extracted filters: {filters}")
    
    # Load the model and generate query embedding
    model = setup_model(config['model_name'], config['use_gpu'])
    query_embedding = model.encode([processed_query], convert_to_numpy=True)
    
    # Set search parameters
    if hasattr(index, 'nprobe'):
        # Adjust search parameters for IVF indices
        original_nprobe = index.nprobe
        index.nprobe = config['nprobe']
        logger.info(f"Set nprobe={index.nprobe} for search")
    
    # Get initial larger pool of candidates for filtering
    search_k = min(top_k * 3, index.ntotal) if hasattr(index, 'ntotal') else top_k * 3
    distances, indices = index.search(query_embedding, search_k)
    
    # Reset search parameters
    if hasattr(index, 'nprobe'):
        index.nprobe = original_nprobe
    
    # Convert results to product data
    candidate_indices = indices[0]
    candidate_distances = distances[0]
    candidate_products = data.iloc[candidate_indices].copy()
    
    # Make sure candidate_products has a 'distance' column
    candidate_products['distance'] = candidate_distances
    
    # Apply filters
    if 'price_range' in filters and 'price' in candidate_products.columns:
        min_price, max_price = filters['price_range']
        candidate_products = candidate_products[
            (candidate_products['price'] >= min_price) & 
            (candidate_products['price'] <= max_price)
        ]
    
    if 'brand' in filters and 'brand' in candidate_products.columns:
        brand = filters['brand'].lower()
        candidate_products = candidate_products[
            candidate_products['brand'].str.lower().str.contains(brand)
        ]
    
    if 'min_rating' in filters and 'stars' in candidate_products.columns:
        min_rating = filters['min_rating']
        candidate_products = candidate_products[candidate_products['stars'] >= min_rating]
    
    # Rank products based on selected method
    if config['rerank_method'] == 'distance':
        candidate_products = candidate_products.sort_values('distance')
    elif config['rerank_method'] == 'reviews' and 'reviews' in candidate_products.columns:
        candidate_products = candidate_products.sort_values('reviews', ascending=False)
    elif config['rerank_method'] == 'price' and 'price' in candidate_products.columns:
        candidate_products = candidate_products.sort_values('price')
    elif config['rerank_method'] == 'hybrid':
        # Hybrid ranking that considers relevance, ratings, and price
        if 'stars' in candidate_products.columns and 'price' in candidate_products.columns:
            # Normalize scores (higher is better for all metrics)
            candidate_products['relevance_score'] = 1 - (candidate_products['distance'] / candidate_products['distance'].max())
            
            if candidate_products['stars'].max() > candidate_products['stars'].min():
                candidate_products['rating_score'] = (candidate_products['stars'] - candidate_products['stars'].min()) / (candidate_products['stars'].max() - candidate_products['stars'].min())
            else:
                candidate_products['rating_score'] = 1.0
            
            if candidate_products['price'].max() > candidate_products['price'].min():
                candidate_products['price_score'] = 1 - ((candidate_products['price'] - candidate_products['price'].min()) / (candidate_products['price'].max() - candidate_products['price'].min()))
            else:
                candidate_products['price_score'] = 1.0
            
            # Compute weighted hybrid score
            candidate_products['hybrid_score'] = (
                config['relevance_weight'] * candidate_products['relevance_score'] +
                config['rating_weight'] * candidate_products['rating_score'] +
                config['price_weight'] * candidate_products['price_score']
            )
            
            candidate_products = candidate_products.sort_values('hybrid_score', ascending=False)
        else:
            # Fall back to distance-based ranking if required columns are missing
            candidate_products = candidate_products.sort_values('distance')
    
    # Limit to top_k
    result_products = candidate_products.head(top_k)
    
    # Convert to list of dictionaries for better API compatibility
    product_list = []
    for _, product in result_products.iterrows():
        product_dict = product.to_dict()
        
        # Format price as string with $ if available
        if 'price' in product_dict:
            product_dict['price_formatted'] = f"${product_dict['price']:.2f}"
        
        # Format ratings if available
        if 'stars' in product_dict:
            product_dict['stars_formatted'] = f"{product_dict['stars']:.1f} ★"
        
        product_list.append(product_dict)
    
    logger.info(f"Retrieved {len(product_list)} products for query '{query}' in {time.time() - start_time:.2f} seconds")
    
    if return_distances:
        return product_list, candidate_distances[:len(product_list)]
    else:
        return product_list

def format_product_results(products: List[Dict[str, Any]], include_details: bool = True) -> List[str]:
    """
    Format product results as human-readable strings.
    
    Args:
        products: List of product dictionaries
        include_details: Whether to include detailed information
        
    Returns:
        List of formatted product strings
    """
    formatted_results = []
    
    for i, product in enumerate(products, 1):
        if include_details:
            # Detailed format
            product_str = f"{i}. {product.get('title', 'Unknown Product')}"
            
            # Add price if available
            if 'price_formatted' in product:
                product_str += f" - {product.get('price_formatted', '')}"
            elif 'price' in product:
                product_str += f" - ${product.get('price', ''):.2f}"
            
            # Add rating if available
            if 'stars_formatted' in product:
                product_str += f" - {product.get('stars_formatted', '')}"
            elif 'stars' in product:
                product_str += f" - {product.get('stars', '')} ★"
            
            # Add review count if available
            if 'reviews' in product:
                product_str += f" ({product.get('reviews', '')} reviews)"
            
        else:
            # Simple format
            product_str = f"{i}. {product.get('title', 'Unknown Product')}"
        
        formatted_results.append(product_str)
    
    return formatted_results

def search_products(
    query: str,
    top_k: int = 5,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    min_rating: Optional[float] = None,
    include_details: bool = True,
    config_path: str = 'config/retrieval_config.json'
) -> List[str]:
    """
    High-level search function for products.
    
    Args:
        query: User query string
        top_k: Number of products to retrieve
        price_min: Minimum price filter
        price_max: Maximum price filter
        min_rating: Minimum rating filter
        include_details: Whether to include detailed information in results
        config_path: Path to the configuration file
        
    Returns:
        List of formatted product strings
    """
    # Load configuration
    config = load_config(config_path)
    
    # Prepare filters
    filters = {}
    if price_min is not None and price_max is not None:
        filters['price_range'] = (price_min, price_max)
    if min_rating is not None:
        filters['min_rating'] = min_rating
    
    # Retrieve products
    products = retrieve_similar_products(
        query=query,
        config=config,
        top_k=top_k,
        filters=filters
    )
    
    # Format results
    formatted_results = format_product_results(products, include_details)
    
    return formatted_results

# Test the function
if __name__ == '__main__':
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Create default config if it doesn't exist
    config_path = 'config/retrieval_config.json'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    
    # Example user queries
    test_queries = [
        'affordable smartphones',
        'wireless headphones under $50',
        'high-quality kitchen knife set with 4+ stars',
        'laptop with long battery life'
    ]
    
    print("\nTesting product retrieval with different queries:\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            results = search_products(query, top_k=3)
            for result in results:
                print(f"  {result}")
            print()
        except Exception as e:
            print(f"  Error: {str(e)}\n")