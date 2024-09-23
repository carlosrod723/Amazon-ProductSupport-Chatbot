# Amazon Product Support Chatbot

The **Amazon Product Support Chatbot** is a chatbot built using **FAISS** and **SentenceTransformers** for product recommendations. It allows users to input product-related queries and returns relevant products with details such as price, ratings, and links.

The chatbot leverages **Streamlit** for the user interface and **FAISS** for fast similarity search, using embeddings generated from product titles. It offers an efficient way to retrieve and recommend products from a large dataset of Amazon products.

## Dataset:

The dataset contains detailed information about **Amazon Products**, including product titles, prices, and other relevant features. The dataset was cleaned and preprocessed to ensure consistency and reliability.

### Dataset Shape:
- **Rows**: 1,385,430
- **Columns**: 11

### Key Columns:

| Column Name          | Description                                               |
|----------------------|-----------------------------------------------------------|
| `asin`               | Unique identifier for the product on Amazon               |
| `title`              | The product's title or name                               |
| `imgUrl`             | URL to the product image                                  |
| `productURL`         | URL to the product page                                   |
| `stars`              | Average star rating                                       |
| `reviews`            | Number of reviews                                         |
| `price`              | Price of the product in USD                               |
| `listPrice`          | Original listing price before discounts                   |
| `category_id`        | Category identifier for the product                       |
| `isBestSeller`       | Indicates if the product is a best seller (boolean)       |
| `boughtInLastMonth`  | Shows how often the product was bought in the last month  |



---

## Key Features:
- **Product Recommendation**: Retrieves relevant products based on user queries.
- **Price Filtering**: Allows users to filter product recommendations by specifying a price range.
- **Streamlit Interface**: Provides a user-friendly interface for real-time interaction.

---

## Technologies Used:
- **Python**: Core language for development.
- **Streamlit**: Web framework for building the user interface.
- **FAISS (Facebook AI Similarity Search)**: Used to create a similarity search index for efficient product retrieval.
- **SentenceTransformers**: Pre-trained transformer model to generate embeddings from product titles.
- **pandas & numpy**: Libraries for data processing and manipulation.

---

## Workflow:

### 1. Data Loading and Preprocessing:
Amazon product data is loaded and cleaned, ensuring the dataset is free of duplicates and missing values. The cleaned data contains relevant details like product titles, prices, ratings, and links, which are used for product recommendation.

### 2. Embedding Generation:
Product titles are transformed into dense vector embeddings using **SentenceTransformers**. These embeddings represent the semantic meaning of each product title, enabling accurate product retrieval based on user queries.

### 3. Building the FAISS Index:
The embeddings are indexed using **FAISS** to create a similarity search index. This index allows the chatbot to perform fast and scalable product searches, enabling efficient retrieval even with large datasets.

### 4. Product Retrieval:
User queries are converted into embeddings using **SentenceTransformers**. The FAISS index is then searched to find the most similar products. Results are filtered by the price range specified by the user and displayed through the chatbot interface.

### 5. Streamlit Web Interface:
The chatbot runs on a **Streamlit** web interface that allows users to interact with the system in real-time. Users can input product queries, adjust price filters, and view the recommended products in an easy-to-use format.

---

## Key Concepts and Integrations:

### FAISS:
**FAISS (Facebook AI Similarity Search)** is a library designed for fast and efficient similarity search, even in large datasets. The chatbot uses FAISS to search a precomputed index of product embeddings, returning the most relevant results based on user queries. This allows for rapid product retrieval, scaling to millions of products while maintaining low search times.

### SentenceTransformers:
**SentenceTransformers** is a pre-trained model used to generate dense vector embeddings from product titles. These embeddings capture the meaning of the text and enable the chatbot to perform semantic searches. When a user enters a query, the chatbot transforms it into an embedding and searches for similar products in the FAISS index.

### Streamlit:
**Streamlit** provides an intuitive web interface for users to interact with the chatbot. The interface allows users to input queries, set price filters, and view the recommended products in a structured, user-friendly format. It facilitates real-time interaction and ensures a smooth user experience.

---

## Installation and Setup:

1. Clone the repository:
git clone https://github.com/carlosrod723/Amazon-Product-Support-Chatbot.git

2. Install the required dependencies:
pip install -r requirements.txt


3. Run the chatbot using Streamlit:
streamlit run app/chatbot_app.py

4. Open the app in your browser and interact with the chatbot.

## License:
This project is licensed under the MIT License. See the LICENSE file for more information.
