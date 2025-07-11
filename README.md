
# E-commerce Recommender System

A comprehensive hybrid recommender system that combines collaborative filtering, content-based filtering, and matrix factorization to provide personalized product recommendations.

## Features

- **Collaborative Filtering**: Recommends products based on similar users' preferences
- **Content-Based Filtering**: Suggests similar products based on product descriptions and features
- **Matrix Factorization**: Uses SVD to discover hidden patterns in user behavior
- **Hybrid Approach**: Combines all methods for improved accuracy
- **Cold Start Handling**: Provides popular items for new users

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecommerce-recommender-system.git
cd ecommerce-recommender-system

pip install -r requirements.txt
from recommender_system import EcommerceRecommender
import pandas as pd

# Load your data
interactions_df = pd.read_csv('user_interactions.csv')
products_df = pd.read_csv('products.csv')

# Initialize recommender
recommender = EcommerceRecommender(n_recommendations=5)

# Train the system
recommender.load_data(interactions_df, products_df)
recommender.train_model()

# Get recommendations
recommendations = recommender.get_user_recommendations('user_123', method='hybrid')
user_id,item_id,rating
user_1,item_1,5
user_1,item_2,3
user_2,item_1,4
item_id,name,description,category,price
item_1,Product Name,Product description,Electronics,99.99
item_2,Another Product,Another description,Books,19.99
