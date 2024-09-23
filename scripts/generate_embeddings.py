# Import necessary libraries and packages
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(data, model_name= 'all-MiniLM-L6-v2', batch_size= 64):
    model= SentenceTransformer(model_name)

    # Encode the product titles
    titles= data['title'].tolist()
    embeddings= model.encode(titles, batch_size= batch_size, show_progress_bar= True)

    return embeddings

# Test the function
if __name__ == '__main__':

    # Load the cleaned dataset
    file_path = 'data/amazon_products_cleaned.csv'
    data= pd.read_csv(file_path)

    # Generate embeddings
    embeddings= generate_embeddings(data)

    # Save the embeddings
    embeddings_dir= 'embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(f'{embeddings_dir}/product_embeddings.npy', embeddings)

    print(f'Embeddings generated and saved. Shape: {embeddings.shape}')
