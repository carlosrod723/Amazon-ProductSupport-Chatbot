# Import necessary libraries and packages
import os
import numpy as np
import faiss

# Define a function to build the FAISS index
def build_faiss_index(embeddings, index_file= 'embeddings/faiss_index'):

    # Get the dimensionality of the embeddings
    dimension= embeddings.shape[1]

    # Initialize the FAISS index with L2 distance
    index= faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings)
    
    # Save the FAISS index
    faiss.write_index(index, index_file)
    print(f'FAISS index built and saved at {index_file}')

# Test the function
if __name__ == '__main__':

    # Load the embeddings
    embeddings_file= 'embeddings/product_embeddings.npy'
    embeddings= np.load(embeddings_file)

    # Build and save the FAISS index
    build_faiss_index(embeddings)