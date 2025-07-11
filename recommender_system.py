import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EcommerceRecommender:
    """
    A hybrid recommender system that combines:
    1. Collaborative Filtering (user-item interactions)
    2. Content-Based Filtering (product descriptions)
    3. Matrix Factorization (SVD)
    """
    
    def __init__(self, n_recommendations=5):
        self.n_recommendations = n_recommendations
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self.tfidf_vectorizer = None
        self.products_df = None
        self.interactions_df = None
        
    def load_data(self, interactions_df, products_df):
        """
        Load user-item interaction data and product information
        
        interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        products_df: DataFrame with columns ['item_id', 'name', 'description', 'category', 'price']
        """
        self.interactions_df = interactions_df.copy()
        self.products_df = products_df.copy()
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        print(f"Loaded {len(interactions_df)} interactions for {len(interactions_df['user_id'].unique())} users")
        print(f"and {len(products_df)} products")
    
    def calculate_collaborative_filtering(self):
        """
        Calculate user-based and item-based collaborative filtering similarities
        """
        print("Calculating collaborative filtering similarities...")
        
        # Item-based collaborative filtering
        # Calculate similarity between items based on user ratings
        item_matrix = self.user_item_matrix.T  # Transpose to get items as rows
        self.item_similarity_matrix = cosine_similarity(item_matrix)
        
        # User-based collaborative filtering
        # Calculate similarity between users based on their ratings
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        
        print("Collaborative filtering similarities calculated!")
    
    def calculate_content_based_filtering(self):
        """
        Calculate content-based similarities using product descriptions
        """
        print("Calculating content-based similarities...")
        
        # Combine product features into a single text
        self.products_df['combined_features'] = (
            self.products_df['name'].fillna('') + ' ' +
            self.products_df['description'].fillna('') + ' ' +
            self.products_df['category'].fillna('')
        )
        
        # Create TF-IDF vectors for product descriptions
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['combined_features'])
        
        # Calculate content-based similarity matrix
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print("Content-based similarities calculated!")
    
    def train_matrix_factorization(self, n_components=50):
        """
        Train SVD (Singular Value Decomposition) for matrix factorization
        """
        print("Training matrix factorization model...")
        
        # Use SVD to decompose the user-item matrix
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit the model on the user-item matrix
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        
        # Get item factors
        item_factors = self.svd_model.components_.T
        
        # Reconstruct the matrix for predictions
        self.predicted_ratings = np.dot(user_factors, self.svd_model.components_)
        
        print("Matrix factorization model trained!")
    
    def get_user_recommendations(self, user_id, method='hybrid'):
        """
        Get recommendations for a specific user
        
        Methods:
        - 'collaborative': User-based collaborative filtering
        - 'content': Content-based filtering
        - 'matrix_factorization': SVD-based recommendations
        - 'hybrid': Combination of all methods
        """
        
        if user_id not in self.user_item_matrix.index:
            return self.get_popular_items()
        
        recommendations = []
        
        if method in ['collaborative', 'hybrid']:
            collab_recs = self._get_collaborative_recommendations(user_id)
            recommendations.extend(collab_recs)
        
        if method in ['content', 'hybrid']:
            content_recs = self._get_content_recommendations(user_id)
            recommendations.extend(content_recs)
        
        if method in ['matrix_factorization', 'hybrid']:
            mf_recs = self._get_matrix_factorization_recommendations(user_id)
            recommendations.extend(mf_recs)
        
        # Remove duplicates and items already rated by user
        user_rated_items = set(self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ]['item_id'].values)
        
        # Score and rank recommendations
        item_scores = {}
        for item_id, score in recommendations:
            if item_id not in user_rated_items:
                if item_id in item_scores:
                    item_scores[item_id] += score
                else:
                    item_scores[item_id] = score
        
        # Sort by score and return top recommendations
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:self.n_recommendations]
        
        return self._format_recommendations(top_items)
    
    def _get_collaborative_recommendations(self, user_id):
        """Get recommendations using collaborative filtering"""
        recommendations = []
        
        # Get user index
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        # Find similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
        
        # Get items rated by similar users
        for similar_user_idx in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            similar_user_items = self.interactions_df[
                self.interactions_df['user_id'] == similar_user_id
            ]
            
            for _, row in similar_user_items.iterrows():
                item_id = row['item_id']
                rating = row['rating']
                similarity = user_similarities[similar_user_idx]
                
                # Weight the rating by user similarity
                score = rating * similarity
                recommendations.append((item_id, score))
        
        return recommendations
    
    def _get_content_recommendations(self, user_id):
        """Get recommendations using content-based filtering"""
        recommendations = []
        
        # Get items the user has rated highly (rating >= 4)
        user_items = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) & 
            (self.interactions_df['rating'] >= 4)
        ]['item_id'].values
        
        if len(user_items) == 0:
            return recommendations
        
        # Find similar items based on content
        for item_id in user_items:
            if item_id in self.products_df['item_id'].values:
                item_idx = list(self.products_df['item_id']).index(item_id)
                item_similarities = self.content_similarity_matrix[item_idx]
                
                # Get top similar items
                similar_items = np.argsort(item_similarities)[::-1][1:6]  # Top 5 similar items
                
                for similar_item_idx in similar_items:
                    similar_item_id = self.products_df.iloc[similar_item_idx]['item_id']
                    similarity_score = item_similarities[similar_item_idx]
                    recommendations.append((similar_item_id, similarity_score))
        
        return recommendations
    
    def _get_matrix_factorization_recommendations(self, user_id):
        """Get recommendations using matrix factorization"""
        recommendations = []
        
        # Get user index
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        # Get predicted ratings for all items
        user_predictions = self.predicted_ratings[user_idx]
        
        # Create recommendations from predictions
        for item_idx, predicted_rating in enumerate(user_predictions):
            item_id = self.user_item_matrix.columns[item_idx]
            recommendations.append((item_id, predicted_rating))
        
        return recommendations
    
    def get_popular_items(self):
        """Get popular items for new users (cold start problem)"""
        
        # Calculate item popularity based on ratings
        item_popularity = self.interactions_df.groupby('item_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        item_popularity.columns = ['item_id', 'avg_rating', 'rating_count']
        
        # Filter items with at least 5 ratings
        popular_items = item_popularity[item_popularity['rating_count'] >= 5]
        
        # Sort by average rating
        popular_items = popular_items.sort_values('avg_rating', ascending=False)
        
        top_items = popular_items.head(self.n_recommendations)
        top_items_list = [(row['item_id'], row['avg_rating']) for _, row in top_items.iterrows()]
        
        return self._format_recommendations(top_items_list)
    
    def _format_recommendations(self, item_scores):
        """Format recommendations with product details"""
        formatted_recs = []
        
        for item_id, score in item_scores:
            product_info = self.products_df[self.products_df['item_id'] == item_id]
            
            if not product_info.empty:
                product = product_info.iloc[0]
                formatted_recs.append({
                    'item_id': item_id,
                    'name': product['name'],
                    'category': product['category'],
                    'price': product['price'],
                    'description': product['description'][:100] + '...',
                    'recommendation_score': round(score, 3)
                })
        
        return formatted_recs
    
    def train_model(self):
        """Train all components of the recommender system"""
        print("Training recommender system...")
        
        # Calculate all similarity matrices
        self.calculate_collaborative_filtering()
        self.calculate_content_based_filtering()
        self.train_matrix_factorization()
        
        print("Recommender system training completed!")
    
    def get_item_recommendations(self, item_id, n_recommendations=5):
        """Get items similar to a given item"""
        
        if item_id not in self.products_df['item_id'].values:
            return []
        
        # Get item index
        item_idx = list(self.products_df['item_id']).index(item_id)
        
        # Get content-based similarities
        content_similarities = self.content_similarity_matrix[item_idx]
        
        # Get collaborative filtering similarities (if available)
        if item_id in self.user_item_matrix.columns:
            collab_item_idx = list(self.user_item_matrix.columns).index(item_id)
            collab_similarities = self.item_similarity_matrix[collab_item_idx]
        else:
            collab_similarities = np.zeros(len(self.products_df))
        
        # Combine similarities (weighted average)
        combined_similarities = 0.6 * content_similarities + 0.4 * collab_similarities
        
        # Get top similar items
        similar_items = np.argsort(combined_similarities)[::-1][1:n_recommendations+1]
        
        similar_items_list = [(self.products_df.iloc[idx]['item_id'], 
                              combined_similarities[idx]) 
                             for idx in similar_items]
        
        return self._format_recommendations(similar_items_list)


# Example usage and demonstration
def create_sample_data():
    """Create sample data for demonstration"""
    
    # Create sample interactions data
    np.random.seed(42)
    
    # Generate users and items
    users = [f"user_{i}" for i in range(1, 101)]  # 100 users
    items = [f"item_{i}" for i in range(1, 51)]   # 50 items
    
    # Generate interactions
    interactions = []
    for user in users:
        # Each user rates 5-15 items
        n_ratings = np.random.randint(5, 16)
        user_items = np.random.choice(items, n_ratings, replace=False)
        
        for item in user_items:
            # Generate rating (1-5 scale, biased towards higher ratings)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.35, 0.35])
            interactions.append({
                'user_id': user,
                'item_id': item,
                'rating': rating
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    # Create sample products data
    categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports', 'Beauty']
    products = []
    
    for item in items:
        item_num = int(item.split('_')[1])
        category = np.random.choice(categories)
        
        products.append({
            'item_id': item,
            'name': f"Product {item_num}",
            'description': f"This is a great {category.lower()} product with excellent features. "
                          f"Perfect for everyday use and highly rated by customers. "
                          f"Category: {category}. Item number: {item_num}.",
            'category': category,
            'price': np.random.uniform(10, 500)
        })
    
    products_df = pd.DataFrame(products)
    
    return interactions_df, products_df


def demonstrate_recommender():
    """Demonstrate the recommender system"""
    
    print("=== E-commerce Recommender System Demo ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    interactions_df, products_df = create_sample_data()
    
    # Initialize recommender
    recommender = EcommerceRecommender(n_recommendations=5)
    
    # Load data
    print("\n2. Loading data...")
    recommender.load_data(interactions_df, products_df)
    
    # Train the model
    print("\n3. Training the recommender system...")
    recommender.train_model()
    
    # Test different recommendation methods
    test_user = "user_1"
    
    print(f"\n4. Getting recommendations for {test_user}:\n")
    
    # Show user's rating history
    user_history = interactions_df[interactions_df['user_id'] == test_user]
    print(f"User {test_user}'s rating history:")
    for _, row in user_history.head().iterrows():
        product_name = products_df[products_df['item_id'] == row['item_id']]['name'].iloc[0]
        print(f"  - {product_name}: {row['rating']} stars")
    
    print("\n" + "="*50)
    
    # Collaborative filtering recommendations
    print("\nA. Collaborative Filtering Recommendations:")
    collab_recs = recommender.get_user_recommendations(test_user, method='collaborative')
    for i, rec in enumerate(collab_recs, 1):
        print(f"{i}. {rec['name']} (Score: {rec['recommendation_score']})")
        print(f"   Category: {rec['category']}, Price: ${rec['price']:.2f}")
    
    # Content-based recommendations
    print("\nB. Content-Based Recommendations:")
    content_recs = recommender.get_user_recommendations(test_user, method='content')
    for i, rec in enumerate(content_recs, 1):
        print(f"{i}. {rec['name']} (Score: {rec['recommendation_score']})")
        print(f"   Category: {rec['category']}, Price: ${rec['price']:.2f}")
    
    # Matrix factorization recommendations
    print("\nC. Matrix Factorization Recommendations:")
    mf_recs = recommender.get_user_recommendations(test_user, method='matrix_factorization')
    for i, rec in enumerate(mf_recs, 1):
        print(f"{i}. {rec['name']} (Score: {rec['recommendation_score']})")
        print(f"   Category: {rec['category']}, Price: ${rec['price']:.2f}")
    
    # Hybrid recommendations
    print("\nD. Hybrid Recommendations (Best Overall):")
    hybrid_recs = recommender.get_user_recommendations(test_user, method='hybrid')
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"{i}. {rec['name']} (Score: {rec['recommendation_score']})")
        print(f"   Category: {rec['category']}, Price: ${rec['price']:.2f}")
    
    # Item-to-item recommendations
    print("\n" + "="*50)
    print("\n5. Item-to-Item Recommendations:")
    test_item = "item_1"
    test_item_name = products_df[products_df['item_id'] == test_item]['name'].iloc[0]
    print(f"\nItems similar to '{test_item_name}':")
    
    similar_items = recommender.get_item_recommendations(test_item, n_recommendations=3)
    for i, rec in enumerate(similar_items, 1):
        print(f"{i}. {rec['name']} (Similarity: {rec['recommendation_score']:.3f})")
        print(f"   Category: {rec['category']}, Price: ${rec['price']:.2f}")
    
    # Popular items (for new users)
    print("\n" + "="*50)
    print("\n6. Popular Items (for new users):")
    popular_items = recommender.get_popular_items()
    for i, rec in enumerate(popular_items, 1):
        print(f"{i}. {rec['name']} (Avg Rating: {rec['recommendation_score']:.2f})")
        print(f"   Category: {rec['category']}, Price: ${rec['price']:.2f}")


# Run the demonstration
if __name__ == "__main__":
    demonstrate_recommender()
