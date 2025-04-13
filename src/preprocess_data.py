"""
Amazon Product Data Preprocessing Module

This module handles data cleaning, preprocessing, and transformation for 
the Amazon product dataset.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import re
from functools import wraps

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amazon_chatbot.data_preprocessor')

# Import the data loader
try:
    from load_data import load_amazon_data, validate_amazon_data
except ImportError:
    logger.warning("Could not import load_data module. Make sure it's in the Python path.")
    # Fallback implementation
    def load_amazon_data(file_path='data/amazon_products.csv'):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def validate_amazon_data(data):
        return data is not None

def timing_decorator(func):
    """Decorator to measure execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def clean_data(
    data: pd.DataFrame, 
    columns_to_clean: List[str] = None,
    remove_duplicates: bool = True,
    handle_missing: bool = True,
    normalize_text: bool = True,
    remove_special_chars: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the Amazon product dataset with multiple preprocessing options.
    
    Args:
        data: DataFrame containing Amazon product data
        columns_to_clean: List of columns to clean (default: ['title'])
        remove_duplicates: Whether to remove duplicate products
        handle_missing: Whether to handle missing values
        normalize_text: Whether to lowercase text columns
        remove_special_chars: Whether to remove special characters from text
        
    Returns:
        Tuple containing:
        - Cleaned DataFrame
        - Dictionary with cleaning statistics
    """
    if data is None or len(data) == 0:
        logger.error("Cannot clean empty or None data")
        return None, {"error": "Empty or None data provided"}
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Define columns to clean if not provided
    if columns_to_clean is None:
        columns_to_clean = ['title']
    
    # Set up statistics dictionary
    stats = {
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "duplicate_rows_removed": 0,
        "missing_rows_removed": 0,
        "columns_cleaned": columns_to_clean,
    }

    # Check if required columns exist
    missing_columns = [col for col in columns_to_clean if col not in df.columns]
    if missing_columns:
        logger.warning(f"Columns {missing_columns} not found in the data")
        # Filter out missing columns
        columns_to_clean = [col for col in columns_to_clean if col in df.columns]
    
    # 1. Remove duplicates if enabled
    if remove_duplicates:
        original_len = len(df)
        if 'asin' in df.columns:
            # Use ASIN as the primary deduplication key if available
            df.drop_duplicates(subset='asin', inplace=True)
            logger.info(f"Removed {original_len - len(df)} duplicate ASINs")
        
        # Also deduplicate by title if it exists
        if 'title' in df.columns:
            title_dupes_len = len(df)
            df.drop_duplicates(subset='title', inplace=True)
            title_dupes_removed = title_dupes_len - len(df)
            logger.info(f"Removed {title_dupes_removed} duplicate titles")
        
        stats["duplicate_rows_removed"] = original_len - len(df)
    
    # 2. Handle missing values if enabled
    if handle_missing:
        original_len = len(df)
        
        # For critical columns, drop rows with missing values
        critical_columns = [col for col in ['asin', 'title', 'price'] if col in df.columns]
        if critical_columns:
            df.dropna(subset=critical_columns, inplace=True)
            logger.info(f"Removed {original_len - len(df)} rows with missing values in {critical_columns}")
        
        # For non-critical columns, fill with appropriate values
        if 'reviews' in df.columns:
            df['reviews'].fillna(0, inplace=True)
        
        if 'stars' in df.columns:
            df['stars'].fillna(0, inplace=True)
        
        if 'price' in df.columns:
            # Ensure price is numeric
            df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
            # Handle any resulting NaNs
            median_price = df['price'].median()
            df['price'].fillna(median_price, inplace=True)
        
        stats["missing_rows_removed"] = original_len - len(df)
    
    # 3. Process text columns
    for col in columns_to_clean:
        if col in df.columns:
            # Lowercase text if enabled
            if normalize_text:
                df[col] = df[col].astype(str).str.lower()
                logger.info(f"Normalized text in column '{col}'")
            
            # Remove special characters if enabled
            if remove_special_chars:
                # More targeted approach - keep alphanumeric, spaces, and common punctuation
                df[col] = df[col].astype(str).apply(
                    lambda x: re.sub(r'[^\w\s.,;:!?()[\]{}"\'-]', '', x)
                )
                logger.info(f"Removed special characters from column '{col}'")
    
    # 4. Additional preprocessing
    # Convert price to numeric if it exists
    if 'price' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['price']):
            df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('[$,]', '', regex=True), errors='coerce')
            logger.info("Converted price column to numeric format")
    
    # Update statistics
    stats["final_rows"] = len(df)
    stats["final_columns"] = len(df.columns)
    stats["rows_removed"] = stats["original_rows"] - stats["final_rows"]
    stats["percent_data_retained"] = round((stats["final_rows"] / stats["original_rows"]) * 100, 2) if stats["original_rows"] > 0 else 0
    
    logger.info(f"Data cleaning completed. Retained {stats['percent_data_retained']}% of original data.")
    return df, stats

@timing_decorator
def prepare_for_embeddings(
    data: pd.DataFrame,
    text_column: str = 'title',
    max_token_length: int = 512,
    add_category_info: bool = True
) -> pd.DataFrame:
    """
    Prepare data for embedding generation by optimizing text content.
    
    Args:
        data: DataFrame containing cleaned Amazon product data
        text_column: Column to use for embedding generation
        max_token_length: Maximum token length for text content
        add_category_info: Whether to include category information in the text
        
    Returns:
        DataFrame with a new 'embedding_text' column optimized for embedding generation
    """
    if data is None or len(data) == 0:
        logger.error("Cannot prepare empty data for embeddings")
        return None
    
    df = data.copy()
    
    # Ensure text column exists
    if text_column not in df.columns:
        logger.error(f"Text column '{text_column}' not found in data")
        return df
    
    # Create embedding text column
    df['embedding_text'] = df[text_column].astype(str)
    
    # Add category information if available and requested
    if add_category_info and 'category_id' in df.columns:
        # Check if we have a category mapping
        try:
            # Try to use category names if available
            df['embedding_text'] = df.apply(
                lambda row: f"{row[text_column]} category:{row['category_id']}", 
                axis=1
            )
            logger.info("Added category information to embedding text")
        except:
            logger.warning("Could not add category information to embedding text")
    
    # Truncate text to maximum token length (approximate)
    # This is a simple approach; a more accurate approach would use a tokenizer
    df['embedding_text'] = df['embedding_text'].apply(
        lambda x: ' '.join(x.split()[:max_token_length])
    )
    
    logger.info(f"Prepared {len(df)} products for embedding generation")
    return df

@timing_decorator
def process_data_pipeline(
    input_file: str = 'data/amazon_products.csv',
    output_file: str = 'data/amazon_products_processed.csv',
    save_output: bool = True
) -> pd.DataFrame:
    """
    Run the complete data processing pipeline from loading to final preparation.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
        save_output: Whether to save the processed data to a CSV file
        
    Returns:
        Processed DataFrame ready for embedding generation
    """
    # 1. Load the data
    data = load_amazon_data(input_file)
    if not validate_amazon_data(data):
        logger.error("Data validation failed. Aborting pipeline.")
        return None
    
    # 2. Clean the data
    cleaned_data, stats = clean_data(data)
    logger.info(f"Data cleaning statistics: {stats}")
    
    # 3. Prepare for embeddings
    processed_data = prepare_for_embeddings(cleaned_data)
    
    # 4. Save processed data if requested
    if save_output and processed_data is not None:
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        processed_data.to_csv(output_file, index=False)
        logger.info(f"Processed dataset saved to {output_file}")
        logger.info(f"Final dataset has {processed_data.shape[0]} rows and {processed_data.shape[1]} columns")
    
    return processed_data

# Test the function
if __name__ == '__main__':
    # Run the complete pipeline
    processed_data = process_data_pipeline()
    
    if processed_data is not None:
        # Print some sample data
        print("\nSample processed data:")
        print(processed_data[['title', 'embedding_text']].head())