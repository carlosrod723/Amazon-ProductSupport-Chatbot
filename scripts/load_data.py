# Import necessary libraries and packages
import pandas as pd

# Define a function to load in the data
def load_amazon_data(file_path= 'data/amazon_products.csv'):
    try:
        data= pd.read_csv(file_path)
        print(f'Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.')
        return data
    except FileNotFoundError:
        print('File not found. Please check the file path.')
        return None

# Test the function
if __name__ == '__main__':
    load_amazon_data()