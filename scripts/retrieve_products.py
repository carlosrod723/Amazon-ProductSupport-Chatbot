# Import necessary lirbaries and packages
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Define a function to load the FAISS index
def load_faiss_index(index_file= 'embeddings/faiss_index'):

    # Load the FAISS index
    index= faiss.read_index(index_file)
    return index

# Define a function to retrieve similar products
def retrieve_similar_products(query, index, data, model_name= 'all-MiniLM-L6-v2', top_k= 5):

    # Load the Sentence Transformer model
    model= SentenceTransformer(model_name)
    
    # Encode the user query into an embedding
    query_embedding= model.encode([query])
    
    # Search the FAISS index for the top_k nearest products
    distances, indices= index.search(query_embedding, top_k)
    
    # Retrieve the product titles from the original dataset
    similar_products= data.iloc[indices[0]]['title'].tolist()
    
    return similar_products

# Test the function
if __name__ == '__main__':

    # Load the FAISS index
    index= load_faiss_index()

    # Load the cleaned product data
    data= pd.read_csv('data/amazon_products_cleaned.csv')

    # Example user query
    query= 'affordable smartphones'
    
    # Retrieve similar products
    products= retrieve_similar_products(query, index, data)
    
    print('Top similar products:')
    for i, product in enumerate(products, 1):
        print(f'{i}. {product}')