"""
Fixed version of the generate_embeddings.py script
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amazon_chatbot.embedding_generator')

# Import data processing modules if available
try:
    from src.load_data import load_amazon_data
    from src.preprocess_data import clean_data, prepare_for_embeddings
except ImportError:
    try:
        from load_data import load_amazon_data
        from preprocess_data import clean_data, prepare_for_embeddings
    except ImportError:
        logger.warning("Could not import data processing modules. Running in standalone mode.")

# Create directories if they don't exist
os.makedirs('embeddings', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 32,
    'max_seq_length': 128,
    'use_gpu': True,
    'embedding_dim': 384,
    'save_dir': 'embeddings',
    'input_column': 'embedding_text',
}

# Create or load config
config_path = 'config/embedding_config.json'
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    logger.info(f"Created default configuration file at {config_path}")
    config = DEFAULT_CONFIG
else:
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = DEFAULT_CONFIG

# Main process
def main():
    # 1. Load data
    logger.info("Loading processed dataset...")
    try:
        data = load_amazon_data('data/amazon_products_processed.csv')
        if data is None:
            logger.error("Failed to load data")
            return
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Optional preprocessing
    try:
        # Only clean if needed (this dataset is already processed)
        data, _ = clean_data(data)
        data = prepare_for_embeddings(data)
    except Exception as e:
        logger.warning(f"Error in preprocessing: {str(e)}")
    
    # 2. Load model
    logger.info(f"Loading SentenceTransformer model: {config['model_name']}")
    try:
        model = SentenceTransformer(config['model_name'])
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {embedding_dim}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # 3. Generate embeddings
    logger.info(f"Generating embeddings for {len(data)} products...")
    start_time = time.time()
    
    try:
        # Get text column
        input_column = config['input_column']
        if input_column not in data.columns:
            input_column = 'embedding_text' if 'embedding_text' in data.columns else 'title'
            logger.warning(f"Input column not found, using {input_column} instead")
        
        # Convert texts to list
        texts = data[input_column].astype(str).tolist()
        
        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=config['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Embeddings generated in {elapsed_time:.2f} seconds")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
        # 4. Save embeddings
        embedding_file = os.path.join(config['save_dir'], 'product_embeddings.npy')
        np.save(embedding_file, embeddings)
        logger.info(f"Embeddings saved to {embedding_file}")
        
        # 5. Save metadata
        metadata = {
            "model_name": config['model_name'],
            "embedding_dim": embeddings.shape[1],
            "num_products": len(data),
            "elapsed_time": elapsed_time,
            "products_per_second": len(data) / elapsed_time,
            "batch_size": config['batch_size'],
            "input_column": input_column,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_file = os.path.join(config['save_dir'], 'product_embeddings_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
        
        print("\nEmbedding generation completed successfully!")
        print(f"Generated {len(data):,} embeddings in {elapsed_time:.2f} seconds")
        print(f"Processing speed: {len(data) / elapsed_time:.1f} products/second")
        
    except Exception as e:
        logger.error(f"Error in embedding process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()