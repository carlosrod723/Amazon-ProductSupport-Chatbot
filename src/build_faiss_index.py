"""
Amazon Product FAISS Index Building Module

This module handles building optimized FAISS indices for fast similarity search
of Amazon product embeddings, with special optimizations for Apple Silicon.
"""

import os
import logging
import numpy as np
import faiss
import json
import time
from typing import Optional, Dict, Any, Tuple, List
import platform
import psutil
import multiprocessing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amazon_chatbot.faiss_builder')

# Default configuration
DEFAULT_CONFIG = {
    'index_type': 'IVF_HNSW',  # Options: 'Flat', 'IVF', 'HNSW', 'IVF_HNSW', 'PQ'
    'nlist': 100,              # Number of clusters for IVF indices
    'M': 32,                   # Number of connections for HNSW
    'efConstruction': 40,      # Exploration factor for HNSW building
    'nprobe': 10,              # Number of clusters to probe during search 
    'm_per_core': 2,           # Split dimensions across cores for multi-threading
    'use_gpu': True,           # Whether to use GPU for index building if available
    'pq_m': 8,                 # Number of subquantizers for PQ compression (for large datasets)
    'hnsw_ef': 128,            # Exploration factor for HNSW search
    'metric': 'L2',            # Distance metric: 'L2' or 'IP' (inner product)
}

def load_config(config_path: str = 'config/faiss_config.json') -> Dict[str, Any]:
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

def detect_hardware() -> Dict[str, Any]:
    """
    Detect hardware capabilities to optimize FAISS index building.
    
    Returns:
        Dictionary containing hardware information
    """
    hw_info = {
        'platform': platform.system(),
        'processor': platform.processor(),
        'cpu_count': multiprocessing.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024 ** 3),
        'gpu_available': False,
        'apple_silicon': False,
    }
    
    # Check for Apple Silicon
    if hw_info['platform'] == 'Darwin' and 'arm' in hw_info['processor'].lower():
        hw_info['apple_silicon'] = True
        logger.info("Detected Apple Silicon processor")
    
    # Check for GPU
    try:
        # Try to use FAISS GPU if available
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            hw_info['gpu_available'] = True
            hw_info['gpu_count'] = faiss.get_num_gpus()
            logger.info(f"Detected {hw_info['gpu_count']} GPUs available for FAISS")
        else:
            logger.info("No GPUs available for FAISS. Using CPU only.")
    except:
        logger.info("Could not detect GPU availability. Using CPU only.")
    
    # Log hardware info
    logger.info(f"Hardware detection complete: {hw_info}")
    return hw_info

def select_optimal_index_type(
    embeddings: np.ndarray,
    hw_info: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """
    Select the optimal FAISS index type based on data and hardware.
    
    Args:
        embeddings: Array of embeddings
        hw_info: Hardware information dictionary
        config: Configuration dictionary
        
    Returns:
        Recommended index type
    """
    # Get dataset size
    num_vectors = embeddings.shape[0]
    dimension = embeddings.shape[1]
    
    # For very small datasets, use Flat index
    if num_vectors < 10000:
        return 'Flat'
    
    # For Apple Silicon with large memory, use IVF_HNSW
    if hw_info['apple_silicon'] and hw_info['memory_gb'] > 64:
        return 'IVF_HNSW'
    
    # For large datasets on machines with less memory, use PQ
    if num_vectors > 1000000 and hw_info['memory_gb'] < 32:
        return 'PQ'
    
    # For medium datasets with GPU, use IVF
    if hw_info['gpu_available'] and num_vectors < 1000000:
        return 'IVF'
    
    # For medium to large datasets on powerful CPUs, use HNSW
    if hw_info['cpu_count'] >= 8 and hw_info['memory_gb'] > 16:
        return 'HNSW'
    
    # Default to IVF as a balanced option
    return 'IVF'

def create_faiss_index(
    dimension: int,
    index_type: str,
    config: Dict[str, Any],
    hw_info: Dict[str, Any]
) -> faiss.Index:
    """
    Create a FAISS index with the specified parameters.
    
    Args:
        dimension: Dimensionality of the embeddings
        index_type: Type of FAISS index to create
        config: Configuration dictionary
        hw_info: Hardware information dictionary
        
    Returns:
        FAISS index
    """
    metric = faiss.METRIC_L2 if config['metric'] == 'L2' else faiss.METRIC_INNER_PRODUCT
    
    if index_type == 'Flat':
        index = faiss.IndexFlatL2(dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dimension)
        logger.info(f"Created Flat index with {dimension} dimensions")
    
    elif index_type == 'IVF':
        # For IVF, we need a quantizer (usually a flat index)
        quantizer = faiss.IndexFlatL2(dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, config['nlist'], metric)
        index.nprobe = config['nprobe']
        logger.info(f"Created IVF index with {dimension} dimensions, {config['nlist']} clusters, nprobe={config['nprobe']}")
    
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dimension, config['M'], metric)
        index.hnsw.efConstruction = config['efConstruction']
        index.hnsw.efSearch = config['hnsw_ef']
        logger.info(f"Created HNSW index with {dimension} dimensions, M={config['M']}, efConstruction={config['efConstruction']}")
    
    elif index_type == 'IVF_HNSW':
        # Create a hierarchical index: HNSW for coarse quantization + IVF for fine search
        quantizer = faiss.IndexHNSWFlat(dimension, config['M'])
        quantizer.hnsw.efConstruction = config['efConstruction']
        quantizer.hnsw.efSearch = config['hnsw_ef']
        index = faiss.IndexIVFFlat(quantizer, dimension, config['nlist'], metric)
        index.nprobe = config['nprobe']
        logger.info(f"Created IVF_HNSW index with {dimension} dimensions, {config['nlist']} clusters")
    
    elif index_type == 'PQ':
        # Product Quantization for memory efficiency
        # For PQ, m (num of subquantizers) must divide the dimension
        m = min(config['pq_m'], dimension)
        while dimension % m != 0:
            m -= 1
        
        index = faiss.IndexPQ(dimension, m, 8, metric)  # 8 bits per subquantizer
        logger.info(f"Created PQ index with {dimension} dimensions, {m} subquantizers")
    
    else:
        logger.warning(f"Unknown index type: {index_type}. Falling back to Flat index.")
        index = faiss.IndexFlatL2(dimension) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dimension)
    
    # Enable multi-threading if available
    faiss.omp_set_num_threads(min(hw_info['cpu_count'], 8))
    
    return index

def optimize_for_hardware(
    index: faiss.Index,
    hw_info: Dict[str, Any],
    config: Dict[str, Any]
) -> faiss.Index:
    """
    Optimize the FAISS index for the available hardware.
    
    Args:
        index: FAISS index to optimize
        hw_info: Hardware information dictionary
        config: Configuration dictionary
        
    Returns:
        Optimized FAISS index
    """
    # For Apple Silicon, use the native CPU implementation
    if hw_info['apple_silicon']:
        # Use multiple CPU cores effectively
        faiss.omp_set_num_threads(min(hw_info['cpu_count'], 12))
        logger.info(f"Optimized for Apple Silicon with {min(hw_info['cpu_count'], 12)} threads")
    
    # If GPU is available and requested, use it
    elif hw_info['gpu_available'] and config['use_gpu']:
        try:
            # Convert to GPU index
            gpu_resources = faiss.StandardGpuResources()
            gpu_options = faiss.GpuClonerOptions()
            gpu_options.useFloat16 = True  # Use FP16 for better performance
            
            index = faiss.index_cpu_to_gpu(gpu_resources, 0, index, gpu_options)
            logger.info("Index converted to GPU implementation")
        except Exception as e:
            logger.warning(f"Failed to convert index to GPU: {str(e)}")
    
    return index

def build_faiss_index(
    embeddings: np.ndarray,
    index_file: str = 'embeddings/faiss_index',
    config_path: str = 'config/faiss_config.json',
    index_type: Optional[str] = None,
    train_sample_size: Optional[int] = None
) -> Tuple[faiss.Index, Dict[str, Any]]:
    """
    Build an optimized FAISS index from embeddings.
    
    Args:
        embeddings: NumPy array of embeddings
        index_file: Path to save the FAISS index
        config_path: Path to the configuration file
        index_type: Type of index to build (overrides config)
        train_sample_size: Number of vectors to use for training
        
    Returns:
        Tuple containing:
        - FAISS index
        - Dictionary with index building statistics
    """
    if embeddings is None or len(embeddings) == 0:
        logger.error("Cannot build index from empty embeddings")
        return None, {"error": "Empty embeddings provided"}
    
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Override index type if specified
    if index_type:
        config['index_type'] = index_type
    
    # Detect hardware
    hw_info = detect_hardware()
    
    # Get embedding dimensions
    num_vectors, dimension = embeddings.shape
    logger.info(f"Building index for {num_vectors} vectors with {dimension} dimensions")
    
    # Select optimal index type if not explicitly set
    if index_type is None:
        recommended_type = select_optimal_index_type(embeddings, hw_info, config)
        if recommended_type != config['index_type']:
            logger.info(f"Recommended index type: {recommended_type} (configured: {config['index_type']})")
            # Use recommended type if it's likely better
            if (recommended_type == 'Flat' and num_vectors < 10000) or \
               (recommended_type == 'PQ' and num_vectors > 1000000):
                config['index_type'] = recommended_type
                logger.info(f"Using recommended index type: {config['index_type']}")
    
    # Create the index
    index = create_faiss_index(dimension, config['index_type'], config, hw_info)
    
    # Optimize index for hardware
    index = optimize_for_hardware(index, hw_info, config)
    
    # Train the index if needed
    if hasattr(index, 'train'):
        if train_sample_size and train_sample_size < num_vectors:
            # Sample vectors for training
            indices = np.random.choice(num_vectors, train_sample_size, replace=False)
            train_vectors = embeddings[indices].astype(np.float32)
            logger.info(f"Training index with {train_sample_size} sampled vectors")
        else:
            train_vectors = embeddings.astype(np.float32)
            logger.info(f"Training index with all {num_vectors} vectors")
        
        try:
            train_start = time.time()
            index.train(train_vectors)
            train_time = time.time() - train_start
            logger.info(f"Index training completed in {train_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error training index: {str(e)}")
    
    # Add vectors to the index
    try:
        add_start = time.time()
        # Ensure embeddings are float32 (required by FAISS)
        embeddings_f32 = embeddings.astype(np.float32)
        index.add(embeddings_f32)
        add_time = time.time() - add_start
        logger.info(f"Added {num_vectors} vectors to index in {add_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error adding vectors to index: {str(e)}")
        return None, {"error": f"Failed to add vectors: {str(e)}"}
    
    # Convert back to CPU index for saving if it's a GPU index
    if hasattr(index, 'is_gpu') and index.is_gpu:
        try:
            index = faiss.index_gpu_to_cpu(index)
            logger.info("Converted GPU index back to CPU for saving")
        except Exception as e:
            logger.warning(f"Failed to convert GPU index to CPU: {str(e)}")
    
    # Save the index
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        
        save_start = time.time()
        faiss.write_index(index, index_file)
        save_time = time.time() - save_start
        logger.info(f"Index saved to {index_file} in {save_time:.2f} seconds")
        
        # Save index metadata
        metadata = {
            "index_type": config['index_type'],
            "dimension": dimension,
            "num_vectors": num_vectors,
            "construction_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": hw_info,
            "config": config
        }
        
        metadata_file = f"{os.path.splitext(index_file)[0]}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Index metadata saved to {metadata_file}")
    except Exception as e:
        logger.error(f"Error saving index: {str(e)}")
    
    # Calculate statistics
    total_time = time.time() - start_time
    stats = {
        "index_type": config['index_type'],
        "num_vectors": num_vectors,
        "dimension": dimension,
        "total_build_time": total_time,
        "vectors_per_second": num_vectors / total_time if total_time > 0 else 0,
        "hardware": {
            "platform": hw_info['platform'],
            "cpu_count": hw_info['cpu_count'],
            "memory_gb": round(hw_info['memory_gb'], 1),
            "gpu_available": hw_info['gpu_available'],
            "apple_silicon": hw_info['apple_silicon']
        }
    }
    
    logger.info(f"Index building completed in {total_time:.2f} seconds")
    logger.info(f"Performance: {stats['vectors_per_second']:.1f} vectors/second")
    
    return index, stats

def test_index_performance(
    index: faiss.Index,
    embeddings: np.ndarray,
    num_queries: int = 100,
    k: int = 10
) -> Dict[str, Any]:
    """
    Test the performance of the FAISS index.
    
    Args:
        index: FAISS index to test
        embeddings: NumPy array of embeddings
        num_queries: Number of queries to run
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Dictionary with performance statistics
    """
    if index is None or embeddings is None:
        logger.error("Cannot test index or embeddings that are None")
        return {"error": "Invalid index or embeddings"}
    
    # Sample query vectors
    num_vectors = embeddings.shape[0]
    query_indices = np.random.choice(num_vectors, min(num_queries, num_vectors), replace=False)
    query_vectors = embeddings[query_indices].astype(np.float32)
    
    # Measure search time
    start_time = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - start_time
    
    # Calculate statistics
    stats = {
        "num_queries": len(query_vectors),
        "k": k,
        "total_search_time": search_time,
        "queries_per_second": len(query_vectors) / search_time if search_time > 0 else 0,
        "avg_search_time_ms": (search_time / len(query_vectors) * 1000) if len(query_vectors) > 0 else 0
    }
    
    logger.info(f"Performance test: {stats['queries_per_second']:.1f} queries/second")
    logger.info(f"Average search time: {stats['avg_search_time_ms']:.2f} ms per query")
    
    return stats

# Test the function
if __name__ == '__main__':
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Create default config if it doesn't exist
    config_path = 'config/faiss_config.json'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    
    # Load the embeddings
    try:
        embeddings_file = 'embeddings/product_embeddings.npy'
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        exit(1)
    
    # Build the index
    index, stats = build_faiss_index(
        embeddings,
        index_file='embeddings/faiss_index',
        config_path=config_path
    )
    
    # Test index performance
    if index is not None:
        performance_stats = test_index_performance(index, embeddings)
        
        print("\nIndex Building Summary:")
        print(f"- Index type: {stats['index_type']}")
        print(f"- Vectors: {stats['num_vectors']:,}")
        print(f"- Dimensions: {stats['dimension']}")
        print(f"- Build time: {stats['total_build_time']:.2f} seconds")
        print(f"- Indexing speed: {stats['vectors_per_second']:.1f} vectors/second")
        
        print("\nSearch Performance:")
        print(f"- Queries per second: {performance_stats['queries_per_second']:.1f}")
        print(f"- Average search time: {performance_stats['avg_search_time_ms']:.2f} ms per query")