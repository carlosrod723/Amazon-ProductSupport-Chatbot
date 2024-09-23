# Import necessary libraries and packages
import pandas as pd

# Define a function to clean the data
def clean_data(data):

    # Remove duplicates
    data.drop_duplicates(subset='title', inplace= True)

    # Handle missing values
    data.dropna(subset=['title'], inplace= True)

    # Lowercase the titles for consistency
    data['title']= data['title'].str.lower()

    # Remove special characters from titles (optional)
    data['title']= data['title'].str.replace(r'[^a-zA-Z0-9\s]', '', regex= True)

    return data

# Test the function
if __name__ == '__main__':
    # Load the data
    file_path= 'data/amazon_products.csv'
    data= pd.read_csv(file_path)

    # Clean the data
    clean_data= clean_data(data)

    # Save the cleaned data to a new CSV
    clean_data.to_csv('data/amazon_products_cleaned.csv', index= False)
    print(f'Cleaned dataset saved. It now has {clean_data.shape[0]} rows and {clean_data.shape[1]} columns.')
