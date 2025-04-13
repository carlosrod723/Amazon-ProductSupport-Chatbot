"""
Amazon Product Support Chatbot

A refined Streamlit application for product recommendations powered by
FAISS and SentenceTransformers.
"""

import os
import time
import logging
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amazon_chatbot.app')

# Cache for model and index to improve performance
_cache = {
    'model': None,
    'index': None,
    'data': None,
    'last_load_time': 0,
    'index_metadata': None
}

# App configuration
APP_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'index_file': 'embeddings/faiss_index',
    'data_file': 'data/amazon_products_processed.csv',
    'backup_data_file': 'data/amazon_products_cleaned.csv',
    'cache_ttl': 3600,  # Time to live in seconds (1 hour)
    'default_top_k': 10,
    'max_top_k': 50,
    'default_price_range': (0, 1000),
    'max_price': 5000,
    'colors': {
        'primary': '#2A2539',     # Dark Puce
        'secondary': '#EDEFC8',   # Chartreuse
        'accent': '#EFC8C8',      # Coral
        'neutral': '#7C7676',     # Spanish Gray
        'background': '#EAE9E5',  # Oatmeal
        'highlight': '#BDD3CC'    # Sea
    },
    'min_valid_price': 1.0,       # Minimum price to consider valid
}

# Apply custom theme
def apply_custom_theme():
    """Apply custom theme to the Streamlit app."""
    colors = APP_CONFIG['colors']
    
    # Custom CSS
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        * {{
            font-family: 'Poppins', sans-serif;
        }}
        
        /* Main background */
        .stApp {{
            background-color: {colors['background']};
        }}
        
        /* Hide sidebar */
        [data-testid="stSidebar"] {{
            display: none !important;
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {colors['primary']} !important;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}
        
        /* Main header */
        .title-container {{
            background: linear-gradient(135deg, white 0%, {colors['highlight']} 100%);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .title-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, {colors['accent']} 0%, {colors['secondary']} 100%);
        }}
        
        .title-container h1 {{
            font-size: 3rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, {colors['primary']} 0%, #483D6B 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}
        
        .title-container p {{
            font-size: 1.2rem;
            color: {colors['neutral']};
            max-width: 800px;
            margin: 0 auto;
        }}
        
        /* Filters section */
        .filters-container {{
            background-color: white;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            position: relative;
            border: none;
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: flex-end;
        }}
        
        .filter-item {{
            flex: 1;
            min-width: 200px;
        }}
        
        .filter-label {{
            font-size: 0.9rem;
            font-weight: 500;
            color: {colors['primary']};
            margin-bottom: 0.5rem;
        }}
        
        /* Buttons */
        .stButton>button {{
            background: linear-gradient(90deg, {colors['primary']} 0%, #483D6B 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 4px 12px rgba(42, 37, 57, 0.15) !important;
            transition: all 0.3s ease !important;
            height: 2.5rem !important; /* Match height with search input */
            margin-top: 0 !important;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(42, 37, 57, 0.25) !important;
        }}
        
        /* Text inputs - fix container styling */
        .stTextInput > div {{
            border: none !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }}
        
        /* Remove default Streamlit field backgrounds */
        [data-baseweb="base-input"] {{
            background-color: transparent !important;
            border: none !important;
        }}
        
        /* The actual input styling */
        .stTextInput>div>div>input {{
            border-radius: 8px !important;
            border: 1px solid {colors['highlight']} !important;
            padding: 0.8rem 1rem !important;
            font-size: 1rem !important;
            box-shadow: none !important;
            transition: all 0.3s ease !important;
            background-color: white !important;
            width: 100% !important;
            height: auto !important;
            min-height: 2.5rem !important;
        }}
        
        .stTextInput>div>div>input:focus {{
            border-color: {colors['accent']} !important;
            box-shadow: 0 4px 12px rgba(239, 200, 200, 0.15) !important;
        }}
        
        .stTextInput[data-testid="stTextInput"] label {{
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            color: {colors['primary']} !important;
            margin-bottom: 0.5rem !important;
        }}
        
        /* Slider */
        .stSlider {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }}
        
        .stSlider [data-testid="stThumbValue"] {{
            background: {colors['secondary']} !important;
            color: {colors['primary']} !important;
            font-weight: 600 !important;
            padding: 2px 8px !important;
            border-radius: 20px !important;
        }}
        
        /* Fix for slider track */
        div[data-testid="stSlider"] > div > div > div {{
            background-color: rgba(42, 37, 57, 0.1) !important;
        }}
        
        div[data-testid="stSlider"] > div > div > div > div {{
            background-color: {colors['primary']} !important;
        }}
        
        /* Number inputs */
        .stNumberInput label {{
            color: {colors['primary']} !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
        }}
        
        /* Checkboxes */
        [data-testid="stCheckbox"] {{
            color: {colors['primary']} !important;
        }}
        
        [data-testid="stCheckbox"] > div[role="checkbox"] > div[data-testid="stMarkdown"] p {{
            font-size: 1rem !important;
        }}
        
        /* Product cards */
        .product-card {{
            background-color: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
            transition: all 0.3s ease;
            display: flex;
            position: relative;
            border: none;
        }}
        
        .product-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        }}
        
        .product-image {{
            width: 200px;
            min-width: 200px;
            height: 200px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: white;
        }}
        
        .product-image img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        
        .product-details {{
            padding: 1.5rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }}
        
        .product-header {{
            position: relative;
            margin-bottom: 1rem;
        }}
        
        .product-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {colors['primary']};
            line-height: 1.4;
            margin-bottom: 0.5rem;
            max-width: 90%;
        }}
        
        .product-content {{
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            flex-grow: 1;
        }}
        
        .product-price {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {colors['primary']};
            margin-bottom: 0.5rem;
        }}
        
        .product-rating {{
            color: #FFB900;
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .product-rating .reviews {{
            color: {colors['neutral']};
            margin-left: 0.7rem;
            font-size: 0.95rem;
            font-weight: 400;
        }}
        
        .product-button {{
            background: linear-gradient(90deg, {colors['primary']} 0%, #483D6B 100%);
            color: white;
            text-decoration: none;
            padding: 0.8rem 1.5rem;
            border-radius: 30px;
            display: inline-block;
            text-align: center;
            width: fit-content;
            font-weight: 500;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 12px rgba(42, 37, 57, 0.15);
            transition: all 0.3s ease;
        }}
        
        .product-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(42, 37, 57, 0.25);
            background: linear-gradient(90deg, #483D6B 0%, {colors['primary']} 100%);
            text-decoration: none;
            color: white;
        }}
        
        /* Bestseller tag */
        .bestseller-tag {{
            position: absolute;
            top: 0;
            right: 0;
            background: linear-gradient(90deg, #FFD700 0%, #FFC107 100%);
            color: #333;
            padding: 0.3rem 0.7rem;
            border-radius: 0 0 0 8px;
            font-size: 0.8rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
        }}
        
        /* No results message */
        .no-results-container {{
            background-color: white;
            padding: 3rem;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            text-align: center;
            margin-top: 2rem;
            border: none;
        }}
        
        .no-results-container h3 {{
            color: {colors['primary']};
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
        }}
        
        .no-results-container p {{
            color: {colors['neutral']};
            font-size: 1.2rem;
        }}
        
        /* Welcome examples */
        .welcome-examples {{
            margin-top: 1.2rem;
            font-style: italic;
            opacity: 0.8;
        }}
        
        /* Helper text */
        .helper-text {{
            color: {colors['neutral']};
            font-size: 0.85rem;
            font-style: italic;
            margin-top: 0.3rem;
            margin-bottom: 0;
            opacity: 0.8;
            line-height: 1.2;
        }}
        
        /* Unified text styles */
        p, .stMarkdown p {{
            font-size: 1rem;
            line-height: 1.6;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 3rem 0;
            margin-top: 4rem;
            color: {colors['neutral']};
            font-size: 0.95rem;
            border-top: 1px solid rgba(0,0,0,0.08);
            position: relative;
        }}
        
        .footer::before {{
            content: '';
            position: absolute;
            top: -1px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, {colors['accent']} 0%, {colors['secondary']} 100%);
            border-radius: 3px;
        }}
        
        /* Results section */
        .results-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.5rem;
            background: white;
            padding: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        
        .results-stats {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {colors['primary']};
        }}
        
        .results-filters {{
            font-size: 0.95rem;
            color: {colors['neutral']};
            background: {colors['background']};
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .animated {{
            animation: fadeIn 0.5s ease-out forwards;
        }}
        
        /* Loading animation */
        .stSpinner {{
            border-color: {colors['accent']} !important;
        }}
        
        /* Fix for extra container and cutoff issues */
        .block-container {{
            padding-top: 3rem !important;
            padding-bottom: 1rem !important;
            max-width: 100% !important;
        }}
        
        /* Fix cutoff in title container */
        .title-container {{
            margin-top: 1rem !important;
            padding-top: 3rem !important;
        }}
        
        /* Main column padding adjustment */
        .main .block-container {{
            padding-left: 4rem;
            padding-right: 4rem;
        }}
        
        /* Fix for streamlit elements */
        div[data-testid="stDecoration"] {{
            background-image: none !important;
        }}
        
        /* Fix for extra widget padding */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {{
            padding-left: 0 !important;
            padding-right: 0 !important;
            gap: 0 !important;
        }}
        
        /* Expander for filters */
        .stExpander {{
            border: none !important;
            box-shadow: none !important;
            margin-top: 10px !important;
        }}
        
        .stExpander [data-testid="stExpanderHeader"] {{
            background-color: transparent !important;
            color: {colors['primary']} !important;
            font-weight: 500 !important;
            border: 1px solid {colors['highlight']} !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
        }}
        
        /* Fix spacing between search button and expander */
        .row-widget.stButton {{
            margin-bottom: 10px !important;
        }}
        
        /* Search button */
        .search-button {{
            margin-top: 0.5rem;
        }}
        
        /* Two-column layout */
        .two-column {{
            display: flex;
            gap: 1rem;
        }}
        
        .column {{
            flex: 1;
        }}
    </style>
    """, unsafe_allow_html=True)

def load_config(config_path: str = 'config/app_config.json') -> Dict[str, Any]:
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
                return {**APP_CONFIG, **config}  # Merge with defaults
        else:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return APP_CONFIG
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return APP_CONFIG

@st.cache_resource(ttl=3600)
def setup_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Set up the SentenceTransformer model with caching.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        SentenceTransformer model
    """
    start_time = time.time()
    
    try:
        # Initialize the model
        model = SentenceTransformer(model_name)
        
        logger.info(f"Model '{model_name}' loaded in {time.time() - start_time:.2f} seconds")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading recommendation model: {str(e)}")
        return None

@st.cache_resource(ttl=3600)
def load_faiss_index(index_file: str = 'embeddings/faiss_index') -> Tuple[Optional[faiss.Index], Optional[Dict]]:
    """
    Load the FAISS index with caching.
    
    Args:
        index_file: Path to the FAISS index file
        
    Returns:
        Tuple containing the FAISS index and metadata dictionary
    """
    start_time = time.time()
    
    try:
        # Load the index
        index = faiss.read_index(index_file)
        
        # Load metadata if available
        metadata = None
        metadata_file = f"{os.path.splitext(index_file)[0]}_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Set nprobe parameter if available in IVF indices
            if hasattr(index, 'nprobe') and 'config' in metadata and 'nprobe' in metadata['config']:
                index.nprobe = metadata['config']['nprobe']
                logger.info(f"Set nprobe={index.nprobe} from metadata")
        
        logger.info(f"FAISS index loaded from {index_file} in {time.time() - start_time:.2f} seconds")
        
        return index, metadata
    
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        st.error(f"Error loading search index: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)
def load_product_data(data_file: str, backup_file: str = None) -> Optional[pd.DataFrame]:
    """
    Load the product data with caching.
    
    Args:
        data_file: Path to the product data file
        backup_file: Path to a backup data file if the primary fails
        
    Returns:
        DataFrame containing product data
    """
    start_time = time.time()
    
    try:
        # Load the data
        data = pd.read_csv(data_file)
        
        logger.info(f"Product data loaded from {data_file} in {time.time() - start_time:.2f} seconds")
        return data
    
    except Exception as e:
        logger.error(f"Error loading product data from {data_file}: {str(e)}")
        
        if backup_file:
            try:
                logger.info(f"Attempting to load backup data from {backup_file}")
                data = pd.read_csv(backup_file)
                logger.info(f"Backup data loaded successfully")
                return data
            except Exception as backup_e:
                logger.error(f"Error loading backup data: {str(backup_e)}")
        
        st.error(f"Error loading product data. Please check the data files.")
        return None

def parse_price_filters(query: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extract price filters from a query, handling various formats.
    
    Args:
        query: User query string
        
    Returns:
        Tuple containing min price, max price, and cleaned query
    """
    min_price = None
    max_price = None
    cleaned_query = query
    
    # Pattern for price range (e.g., $50-$200)
    range_pattern = r'\$(\d+(?:\.\d+)?)\s*-\s*\$?(\d+(?:\.\d+)?)'
    range_matches = re.findall(range_pattern, query)
    
    if range_matches:
        min_price = float(range_matches[0][0])
        max_price = float(range_matches[0][1])
        cleaned_query = re.sub(range_pattern, '', query).strip()
    else:
        # Pattern for "under $X" or "less than $X"
        under_pattern = r'(?:under|less than|below|cheaper than|at most|no more than)\s*\$?(\d+(?:\.\d+)?)'
        under_matches = re.findall(under_pattern, query, re.IGNORECASE)
        
        if under_matches:
            max_price = float(under_matches[0])
            cleaned_query = re.sub(under_pattern, '', query, flags=re.IGNORECASE).strip()
        
        # Pattern for "over $X" or "more than $X"
        over_pattern = r'(?:over|more than|above|at least|exceeding|min|minimum|starting at)\s*\$?(\d+(?:\.\d+)?)'
        over_matches = re.findall(over_pattern, query, re.IGNORECASE)
        
        if over_matches:
            min_price = float(over_matches[0])
            cleaned_query = re.sub(over_pattern, '', query, flags=re.IGNORECASE).strip()
    
    return min_price, max_price, cleaned_query

def filter_products_by_category(query: str, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter products based on a query's product category.
    
    Args:
        query: User query string
        products: List of product dictionaries
        
    Returns:
        Filtered list of products
    """
    # Comprehensive product categories with expanded keywords
    categories = {
        'smartphone': ['smartphone', 'phone', 'cell phone', 'mobile phone', 'cellphone', 'iphone', 'android', 'galaxy', 
                      'pixel', 'oneplus', 'xiaomi', 'redmi', 'huawei', 'oppo', 'vivo', 'realme', 'apple phone'],
        
        'laptop': ['laptop', 'notebook', 'macbook', 'chromebook', 'ultrabook', 'gaming laptop', 'portable computer', 
                  'lenovo', 'thinkpad', 'hp', 'dell', 'asus', 'acer', 'msi', 'razer', 'surface laptop', 'laptop computer'],
        
        'headphone': ['headphone', 'headset', 'earphone', 'earbud', 'airpod', 'earpiece', 'headphones', 'earbuds', 
                     'bluetooth headphone', 'wireless earbud', 'beats', 'bose', 'sennheiser', 'sony headphone', 
                     'noise cancelling', 'airpods', 'earphones', 'headsets'],
        
        'tv': ['tv', 'television', 'smart tv', 'lcd', 'led tv', 'oled', 'qled', 'hdtv', '4k tv', '8k tv', 'flat screen', 
              'curved tv', 'android tv', 'roku tv', 'samsung tv', 'lg tv', 'sony tv', 'vizio', 'fire tv'],
        
        'camera': ['camera', 'dslr', 'mirrorless', 'digital camera', 'video camera', 'webcam', 'security camera', 
                  'action camera', 'gopro', 'canon', 'nikon', 'sony camera', 'fujifilm', 'olympus', 'panasonic', 
                  'camcorder', 'point and shoot'],
        
        'tablet': ['tablet', 'ipad', 'android tablet', 'surface tablet', 'samsung tablet', 'ipad pro', 'kindle', 
                  'e-reader', 'galaxy tab', 'fire tablet', 'drawing tablet', 'wacom', 'tablet pc'],
        
        'monitor': ['monitor', 'display', 'screen', 'computer monitor', 'lcd monitor', 'led monitor', 'gaming monitor', 
                   'ultrawide', '4k monitor', 'curved monitor', 'dell monitor', 'lg monitor', 'samsung monitor', 
                   'acer monitor', 'asus monitor'],
        
        'speaker': ['speaker', 'sound bar', 'audio', 'bluetooth speaker', 'wireless speaker', 'smart speaker', 
                   'portable speaker', 'home theater', 'subwoofer', 'bookshelf speaker', 'surround sound', 
                   'echo', 'homepod', 'sonos', 'bose speaker', 'jbl', 'soundbar'],
        
        'watch': ['watch', 'smartwatch', 'apple watch', 'fitness tracker', 'garmin', 'fitbit', 'samsung watch', 
                 'sport watch', 'digital watch', 'analog watch', 'gps watch', 'time', 'wrist'],
        
        'gaming': ['gaming', 'ps5', 'playstation', 'xbox', 'nintendo', 'console', 'controller', 'ps4', 'game', 
                  'switch', 'video game', 'gaming pc', 'gaming headset', 'gaming keyboard', 'gaming mouse', 
                  'gaming chair', 'steam deck'],
                  
        'bicycle': ['bicycle', 'bike', 'mountain bike', 'road bike', 'cycle', 'cycling', 'biking', 'mountain biking', 
                   'bmx', 'trek', 'schwinn', 'giant bike', 'specialized bike', 'cannondale', 'electric bike', 'e-bike', 
                   'folding bike', 'cruiser bike', 'bikes'],
                   
        'keyboard': ['keyboard', 'mechanical keyboard', 'typing', 'keycaps', 'logitech keyboard', 'corsair', 
                    'razer keyboard', 'wireless keyboard', 'gaming keyboard', 'ergonomic keyboard', 'bluetooth keyboard'],
                    
        'mouse': ['mouse', 'computer mouse', 'wireless mouse', 'gaming mouse', 'logitech mouse', 'razer mouse', 
                 'ergonomic mouse', 'bluetooth mouse', 'optical mouse', 'track pad', 'trackball'],
                 
        'printer': ['printer', 'laser printer', 'inkjet printer', 'all-in-one printer', 'hp printer', 'canon printer', 
                   'epson printer', '3d printer', 'photo printer', 'wireless printer', 'printing'],
                   
        'router': ['router', 'wifi router', 'wireless router', 'mesh router', 'networking', 'netgear', 'linksys', 
                  'asus router', 'tp-link', 'wifi extender', 'network switch', 'ethernet', 'access point'],
                  
        'smart home': ['smart home', 'alexa', 'google home', 'home automation', 'smart light', 'smart plug', 
                      'smart thermostat', 'smart lock', 'ring doorbell', 'nest', 'philips hue', 'echo dot'],
                      
        'kitchen': ['kitchen', 'cookware', 'appliance', 'blender', 'mixer', 'coffee maker', 'microwave', 'toaster', 
                   'air fryer', 'instant pot', 'food processor', 'refrigerator', 'dishwasher']
    }
    
    # Extract product type from query with extra preprocessing
    query_lower = query.lower()
    query_words = query_lower.split()
    detected_categories = []
    
    # Check for category matches in the full query
    for category, keywords in categories.items():
        for keyword in keywords:
            # Check if the keyword is present as a whole word or phrase
            if keyword in query_lower:
                if keyword in query_words or f"{keyword}s" in query_words or any(kw.startswith(f"{keyword} ") or kw.endswith(f" {keyword}") for kw in [query_lower]):
                    detected_categories.append((category, len(keyword.split())))  # Store category and word count for priority
    
    # Sort detected categories by word count (longer matches are more specific)
    detected_categories.sort(key=lambda x: x[1], reverse=True)
    
    # If no category was detected, try to find partial matches
    if not detected_categories:
        return products  # Return all products if no category detected
    
    # Use the most specific (longest) category match
    primary_category = detected_categories[0][0]
    primary_keywords = categories[primary_category]
    
    # Filter products by looking for keywords in title
    filtered_products = []
    for product in products:
        title_lower = product['title'].lower()
        match_score = 0
        
        # Check each keyword from the detected category
        for keyword in primary_keywords:
            if keyword in title_lower:
                # Give higher score for exact word matches
                if keyword in title_lower.split() or f"{keyword}s" in title_lower.split():
                    match_score += 3
                else:
                    match_score += 1
        
        # Add product if it has any match
        if match_score > 0:
            # Add match score to product for sorting
            product['category_match_score'] = match_score
            filtered_products.append(product)
    
    # If no products match, check if there are other detected categories
    if not filtered_products and len(detected_categories) > 1:
        # Try the second most specific category
        secondary_category = detected_categories[1][0]
        secondary_keywords = categories[secondary_category]
        
        for product in products:
            title_lower = product['title'].lower()
            if any(keyword in title_lower for keyword in secondary_keywords):
                filtered_products.append(product)
    
    # If still no products match, return original list
    if not filtered_products:
        return products
    
    # Sort by category match score if available
    if all('category_match_score' in product for product in filtered_products):
        filtered_products.sort(key=lambda x: x.get('category_match_score', 0), reverse=True)
        
        # Remove the temporary score field
        for product in filtered_products:
            if 'category_match_score' in product:
                del product['category_match_score']
    
    return filtered_products

def retrieve_similar_products(
    query: str,
    index: faiss.Index,
    data: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 10,
    price_min: float = 0,
    price_max: float = 10000,
    min_rating: Optional[float] = None,
    bestsellers_only: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieve similar products based on query with filtering.
    
    Args:
        query: User query string
        index: FAISS index
        data: Product data DataFrame
        model: SentenceTransformer model
        top_k: Number of products to retrieve
        price_min: Minimum price filter
        price_max: Maximum price filter
        min_rating: Minimum rating filter
        bestsellers_only: Whether to return bestsellers only
        
    Returns:
        List of dictionaries containing product information
    """
    # Process query for price filters
    query_price_min, query_price_max, cleaned_query = parse_price_filters(query)
    
    # Override slider values if price is in query
    if query_price_min is not None:
        price_min = query_price_min
    if query_price_max is not None:
        price_max = query_price_max
    
    # Enhance query with category words if possible
    enhanced_query = cleaned_query
    
    # Generate query embedding for semantic search
    query_embedding = model.encode([enhanced_query], convert_to_numpy=True)
    
    # Retrieve many more candidates for better filtering
    # Using a much larger multiplier to ensure comprehensive filtering
    search_k = min(top_k * 200, index.ntotal)  # Get 200x results for better filtering
    distances, indices = index.search(query_embedding, search_k)
    
    # Retrieve product details for all candidates
    candidate_indices = indices[0]
    candidate_products = data.iloc[candidate_indices].copy()
    
    # Add distance scores
    candidate_products['distance'] = distances[0]
    
    # Filter out invalid prices (zero or unreasonably low)
    if 'price' in candidate_products.columns:
        candidate_products = candidate_products[
            candidate_products['price'] >= APP_CONFIG['min_valid_price']
        ]
    
    # Apply price filter
    if 'price' in candidate_products.columns:
        candidate_products = candidate_products[
            candidate_products['price'].between(price_min, price_max)
        ]
    
    # Apply rating filter if requested
    if min_rating is not None and min_rating > 0 and 'stars' in candidate_products.columns:
        candidate_products = candidate_products[candidate_products['stars'] >= min_rating]
    
    # Filter bestsellers if requested
    if bestsellers_only and 'isBestSeller' in candidate_products.columns:
        candidate_products = candidate_products[candidate_products['isBestSeller'] == True]
    
    # Sort by relevance (distance)
    candidate_products = candidate_products.sort_values('distance')
    
    # Limit to top_k * 5 to give room for category filtering
    result_products = candidate_products.head(top_k * 5)
    
    # Convert to list of dictionaries
    products = []
    for _, row in result_products.iterrows():
        product = {}
        
        # Basic info
        product['title'] = row.get('title', 'Unknown Product')
        
        # Price information
        if 'price' in row and row['price'] >= APP_CONFIG['min_valid_price']:
            product['price'] = row['price']
            product['price_formatted'] = f"${row['price']:.2f}"
        else:
            product['price'] = None
            product['price_formatted'] = 'N/A'
        
        # Rating information
        if 'stars' in row:
            product['rating'] = row['stars']
            # Format stars as icons
            full_stars = int(row['stars'])
            half_star = (row['stars'] % 1) >= 0.5
            product['rating_stars'] = '★' * full_stars + ('½' if half_star else '')
        else:
            product['rating'] = None
            product['rating_stars'] = 'N/A'
        
        # Review count
        if 'reviews' in row:
            product['reviews'] = row['reviews']
            product['reviews_formatted'] = f"{row['reviews']:,} reviews"
        else:
            product['reviews'] = None
            product['reviews_formatted'] = 'N/A'
        
        # URL information
        if 'productURL' in row:
            product['url'] = row['productURL']
        elif 'url' in row:
            product['url'] = row['url']
        else:
            product['url'] = None
        
        # Image URL
        if 'imgUrl' in row:
            product['image_url'] = row['imgUrl']
        else:
            product['image_url'] = None
        
        # Best seller tag
        if 'isBestSeller' in row:
            product['is_bestseller'] = bool(row['isBestSeller'])
        else:
            product['is_bestseller'] = False
        
        # Distance score (relevance)
        product['relevance_score'] = row['distance']
        
        # Category
        if 'category_id' in row:
            product['category'] = row['category_id']
        else:
            product['category'] = None
            
        products.append(product)
    
    # Apply category filtering based on query
    filtered_products = filter_products_by_category(query, products)
    
    # Limit to top_k
    return filtered_products[:top_k]

def render_product_card(product: Dict[str, Any], index: int):
    """
    Render a product card in the UI.
    
    Args:
        product: Dictionary containing product information
        index: Index of the product in the results
    """
    # Skip products with invalid prices
    if product['price'] is None or product['price'] < APP_CONFIG['min_valid_price']:
        return
    
    # Create product card with HTML directly (no markdown conversion)
    col1, col2 = st.columns([1, 3])
    
    # Product image (left column)
    with col1:
        if product['image_url']:
            st.image(product['image_url'], width=180)
        else:
            st.markdown("""
            <div style="width:180px;height:180px;display:flex;align-items:center;justify-content:center;background:#f9f9f9;color:#aaa;">
                No image available
            </div>
            """, unsafe_allow_html=True)
    
    # Product details (right column)
    with col2:
        # Title with bestseller tag
        if product['is_bestseller']:
            st.markdown(f"""<h3 style="margin-top:0;">{index}. {product['title']} <span style="background-color:#FFD700;color:#333;padding:3px 7px;border-radius:4px;font-size:0.7em;margin-left:8px;">BESTSELLER</span></h3>""", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='margin-top:0;'>{index}. {product['title']}</h3>", unsafe_allow_html=True)
        
        # Price
        st.markdown(f"<div style='font-size:1.4rem;font-weight:700;color:#2A2539;'>{product['price_formatted']}</div>", unsafe_allow_html=True)
        
        # Rating
        if product['rating'] is not None:
            rating_stars = "★" * int(product['rating']) 
            if (product['rating'] % 1) >= 0.5:
                rating_stars += "½"
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-top:5px;">
                <span style="color:#FFB900;">{rating_stars}</span>
                <span style="color:#7C7676;margin-left:8px;font-size:0.9rem;">({product['reviews_formatted']})</span>
            </div>
            """, unsafe_allow_html=True)
        
        # View on Amazon button
        if product['url']:
            st.markdown(f"""
            <div style="margin-top:15px;">
                <a href="{product['url']}" target="_blank" style="background:#2A2539;color:white;text-decoration:none;padding:8px 18px;border-radius:30px;font-weight:500;display:inline-block;">
                    View on Amazon
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    # Divider between products
    st.markdown("<hr style='margin:1.5rem 0;opacity:0.2;'>", unsafe_allow_html=True)

def render_search_results(products: List[Dict[str, Any]], query: str, price_min: float, price_max: float):
    """
    Render search results in the UI.
    
    Args:
        products: List of dictionaries containing product information
        query: User query string
        price_min: Minimum price filter
        price_max: Maximum price filter
    """
    # Filter out products with invalid prices
    valid_products = [p for p in products if p['price'] is not None and p['price'] >= APP_CONFIG['min_valid_price']]
    
    # No results message
    if not valid_products:
        st.markdown("""
        <div class="no-results-container animated">
            <h3>No matching products found</h3>
            <p>Try adjusting your search terms or price range.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Results header with price range info
    st.markdown(f"""
    <div class="results-header animated">
        <div class="results-stats">Found {len(valid_products)} products matching '{query}'</div>
        <div class="results-filters">Price range: ${price_min:.2f} - ${price_max:.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display each product in a card
    for i, product in enumerate(valid_products, 1):
        render_product_card(product, i)

def main():
    """Main Streamlit app function."""
    # Set page config
    st.set_page_config(
        page_title="Amazon Product Explorer",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom theme
    apply_custom_theme()
    
    # Load app configuration
    config = load_config()
    
    # Load resources in the background
    with st.spinner("Loading resources..."):
        # Load the model
        model = setup_model(config['model_name'])
        
        # Load the FAISS index
        index, index_metadata = load_faiss_index(config['index_file'])
        
        # Load the product data
        data = load_product_data(config['data_file'], config['backup_data_file'])
    
    # Check if resources loaded successfully
    if model is None or index is None or data is None:
        st.error("Failed to load required resources. Please check the logs for details.")
        return
    
    # Main content
    # Title and description
    st.markdown("""
    <div class="title-container animated">
        <h1>Amazon Product Explorer</h1>
        <p>Discover the perfect products with our AI-powered recommendation engine.</p>
        <p class="welcome-examples">Try queries like "wireless headphones under $100" or "gaming laptop with good battery life"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple, clean layout for search
    # Search row
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Search input
        query = st.text_input(
            "What are you looking for today?",
            placeholder="E.g., affordable smartphones under $500, gaming laptops with good battery life...",
            key="product_query",
            label_visibility="collapsed" # Hide label for cleaner look
        )
        st.markdown(
            '<p class="helper-text">You can include price ranges in your query like "headphones $50-$200" or "laptops over $1000"</p>',
            unsafe_allow_html=True
        )
    
    with col2:
        # Search button first to align with search bar
        search_pressed = st.button("🔍 Search", key="search_button", use_container_width=True)
        
        # Filter options in a simple expander
        with st.expander("Filters"):
            # Price range
            price_min, price_max = st.slider(
                "Price range ($)",
                min_value=0,
                max_value=config['max_price'],
                value=config['default_price_range']
            )
            
            # Rating filter
            min_rating = st.slider(
                "Minimum rating",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5
            )
            min_rating = min_rating if min_rating > 0 else None
            
            # Number of results
            top_k = st.number_input(
                "Number of results",
                min_value=1,
                max_value=config['max_top_k'],
                value=config['default_top_k']
            )
            
            # Include bestsellers only
            bestsellers_only = st.checkbox("Bestsellers only", value=False)
    
    # Process search when query is provided and button is pressed
    if query and search_pressed:
        with st.spinner(f"Finding the best products for: '{query}'"):
            # Extract query-specific price filters
            query_price_min, query_price_max, _ = parse_price_filters(query)
            
            # Use price from query if provided, otherwise use slider values
            price_min_final = query_price_min if query_price_min is not None else price_min
            price_max_final = query_price_max if query_price_max is not None else price_max
            
            # Retrieve products with filters
            products = retrieve_similar_products(
                query=query,
                index=index,
                data=data,
                model=model,
                top_k=top_k,
                price_min=price_min_final,
                price_max=price_max_final,
                min_rating=min_rating,
                bestsellers_only=bestsellers_only
            )
        
        # Log the search query and number of results (useful for debugging)
        logger.info(f"Search query: '{query}' returned {len(products)} products")
        
        # Render results
        render_search_results(products, query, price_min_final, price_max_final)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Amazon Product Explorer | Powered by FAISS and SentenceTransformers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()