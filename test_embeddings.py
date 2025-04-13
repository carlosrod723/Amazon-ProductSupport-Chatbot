"""
Test script for embedding generation with a small sample of data
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import json

# Create embeddings directory if it doesn't exist
os.makedirs('embeddings', exist_ok=True)

# Load a small sample of data (1000 products)
print("Loading data sample...")
try:
    # Load full dataset
    data = pd.read_csv('data/amazon_products_processed.csv')
    # Take a sample
    sample_data = data.sample(1000, random_state=42)
    print(f"Loaded {len(sample_data)} sample products")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

# Initialize the model
print("Loading model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded with embedding dimension: {model.get_sentence_embedding_dimension()}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Generate embeddings
print("Generating embeddings...")
try:
    start_time = time.time()
    
    # Get text to encode
    texts = sample_data['embedding_text'].astype(str).tolist()
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Embeddings generated in {time.time() - start_time:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Save embeddings
    np.save('embeddings/test_embeddings.npy', embeddings)
    print("Embeddings saved successfully")
    
    # Save metadata
    metadata = {
        "model_name": "all-MiniLM-L6-v2",
        "embedding_dim": embeddings.shape[1],
        "num_samples": len(sample_data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('embeddings/test_embeddings_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Metadata saved successfully")
    
except Exception as e:
    print(f"Error in embedding process: {str(e)}")
    import traceback
    traceback.print_exc()