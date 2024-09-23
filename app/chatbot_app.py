# Import necessary libraries and packages
import faiss
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Load FAISS index
def load_faiss_index(index_file= 'embeddings/faiss_index'):
    index= faiss.read_index(index_file)
    return index

# Retrieve similar products with price filter and details
def retrieve_similar_products(query, index, data, model_name= 'all-MiniLM-L6-v2', top_k= 5, price_min= 0, price_max= 10000):
    model= SentenceTransformer(model_name)
    query_embedding= model.encode([query])
    distances, indices= index.search(query_embedding, top_k)

    # Retrieve the product details (title, price, ratings, and link if available)
    similar_products= data.iloc[indices[0]]
    
    # Apply price filter
    filtered_products= similar_products[similar_products['price'].between(price_min, price_max)]
    
    products= []
    for _, row in filtered_products.iterrows():
        product = {
            'title': row['title'],
            'price': row.get('price', 'N/A'),
            'rating': row.get('rating', 'N/A'),
            'url': row.get('url', 'N/A')
        }
        products.append(product)
    return products

# Main function for Streamlit app
def main():
    st.title('Amazon Product Support Chatbot')
    
    # Load the FAISS index and data
    index= load_faiss_index()
    data= pd.read_csv('data/amazon_products_cleaned.csv')
    
    # Add price range filter
    price_min, price_max= st.slider('Select price range', min_value= 0, max_value= 1000, value=(100,500))
    
    # User input query
    query = st.text_input('Ask me for product recommendations (e.g., affordable smartphones, gaming laptops, etc.)')

    if query:
        st.write(f"Search results for: '{query}'")
        
        # Retrieve products
        products= retrieve_similar_products(query, index, data, price_min=price_min, price_max=price_max)
        
        # Display products with details
        for i, product in enumerate(products, 1):
            st.write(f"**{i}. {product['title']}**")
            st.write(f"Price: {product['price']}")
            st.write(f"Rating: {product['rating']}")
            if product['url'] != 'N/A':
                st.write(f"[View Product]({product['url']})")

# Test the function
if __name__ == '__main__':
    main()
