# Recommendation Systems: Complete Guide

## Table of Contents
1. [Introduction to Recommendation Systems](#introduction-to-recommendation-systems)
2. [Collaborative Filtering](#collaborative-filtering)
3. [Content-Based Filtering](#content-based-filtering)
4. [Hybrid Methods](#hybrid-methods)
5. [Matrix Factorization](#matrix-factorization)
6. [Deep Learning for Recommendations](#deep-learning-for-recommendations)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Cold Start Problem](#cold-start-problem)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)

---

## Introduction to Recommendation Systems

Recommendation systems predict user preferences and suggest items they might like. They're used by Netflix, Amazon, Spotify, and many other platforms.

### Types of Recommendations

- **Collaborative Filtering**: Based on user behavior similarity
- **Content-Based**: Based on item features
- **Hybrid**: Combines multiple approaches
- **Knowledge-Based**: Based on explicit requirements
- **Demographic**: Based on user demographics

### Applications

- **E-commerce**: Product recommendations
- **Streaming**: Movie/music recommendations
- **Social Media**: Content recommendations
- **News**: Article recommendations
- **Job Platforms**: Job recommendations

---

## Collaborative Filtering

### User-Based Collaborative Filtering

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarities = None
    
    def fit(self, ratings_df):
        """Build user-item matrix"""
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Calculate user similarities
        self.user_similarities = cosine_similarity(self.user_item_matrix)
        return self
    
    def predict(self, user_id, item_id, k=10):
        """Predict rating for user-item pair"""
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Get similar users
        user_sim = self.user_similarities[user_idx]
        similar_users_idx = np.argsort(user_sim)[::-1][1:k+1]  # Exclude self
        
        # Weighted average
        numerator = 0
        denominator = 0
        
        for sim_user_idx in similar_users_idx:
            rating = self.user_item_matrix.iloc[sim_user_idx, item_idx]
            if rating > 0:
                similarity = user_sim[sim_user_idx]
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def recommend(self, user_id, n=10):
        """Recommend top N items"""
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get unrated items
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Predict ratings
        predictions = {}
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions[item_id] = pred_rating
        
        # Sort and return top N
        top_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return [item_id for item_id, rating in top_items]

# Usage
cf = UserBasedCF()
cf.fit(ratings_df)
recommendations = cf.recommend(user_id=1, n=10)
```

### Item-Based Collaborative Filtering

```python
class ItemBasedCF:
    def __init__(self):
        self.user_item_matrix = None
        self.item_similarities = None
    
    def fit(self, ratings_df):
        """Build item similarity matrix"""
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Calculate item similarities
        self.item_similarities = cosine_similarity(self.user_item_matrix.T)
        return self
    
    def predict(self, user_id, item_id, k=10):
        """Predict rating using item similarities"""
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Get similar items
        item_sim = self.item_similarities[item_idx]
        similar_items_idx = np.argsort(item_sim)[::-1][1:k+1]
        
        # Weighted average
        numerator = 0
        denominator = 0
        
        for sim_item_idx in similar_items_idx:
            rating = user_ratings.iloc[sim_item_idx]
            if rating > 0:
                similarity = item_sim[sim_item_idx]
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
```

---

## Content-Based Filtering

### TF-IDF Based Recommendations

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.item_features = None
        self.item_similarities = None
    
    def fit(self, items_df, text_column='description'):
        """Fit TF-IDF on item descriptions"""
        descriptions = items_df[text_column].fillna('')
        self.item_features = self.tfidf.fit_transform(descriptions)
        
        # Calculate item similarities
        self.item_similarities = cosine_similarity(self.item_features)
        return self
    
    def recommend(self, item_id, n=10):
        """Recommend similar items"""
        item_idx = self.items_df.index.get_loc(item_id)
        
        # Get similar items
        similarities = self.item_similarities[item_idx]
        similar_items_idx = np.argsort(similarities)[::-1][1:n+1]
        
        return [self.items_df.index[i] for i in similar_items_idx]
```

### Feature-Based Recommendations

```python
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class FeatureBasedRecommender:
    def __init__(self):
        self.scaler = StandardScaler()
        self.item_features = None
    
    def fit(self, items_df, feature_columns):
        """Fit on item features"""
        features = items_df[feature_columns]
        self.item_features = self.scaler.fit_transform(features)
        return self
    
    def recommend(self, user_preferences, n=10):
        """Recommend based on user preferences"""
        # Normalize preferences
        user_vector = self.scaler.transform([user_preferences])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([user_vector], self.item_features)[0]
        
        # Get top N
        top_items_idx = np.argsort(similarities)[::-1][:n]
        return [self.items_df.index[i] for i in top_items_idx]
```

---

## Hybrid Methods

### Weighted Hybrid

```python
class HybridRecommender:
    def __init__(self, cf_model, cb_model, cf_weight=0.6):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.cf_weight = cf_weight
        self.cb_weight = 1 - cf_weight
    
    def recommend(self, user_id, item_id=None, n=10):
        """Hybrid recommendation"""
        # Get CF recommendations
        cf_scores = self.cf_model.get_scores(user_id)
        
        # Get CB recommendations
        if item_id:
            cb_scores = self.cb_model.get_similar_scores(item_id)
        else:
            cb_scores = self.cb_model.get_user_scores(user_id)
        
        # Combine scores
        hybrid_scores = {}
        all_items = set(cf_scores.keys()) | set(cb_scores.keys())
        
        for item in all_items:
            cf_score = cf_scores.get(item, 0)
            cb_score = cb_scores.get(item, 0)
            hybrid_scores[item] = (
                self.cf_weight * cf_score + 
                self.cb_weight * cb_score
            )
        
        # Return top N
        top_items = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [item for item, score in top_items]
```

---

## Matrix Factorization

### SVD (Singular Value Decomposition)

```python
from scipy.sparse.linalg import svds

class SVDRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, ratings_matrix):
        """Factorize rating matrix"""
        # Mean centering
        user_means = ratings_matrix.mean(axis=1)
        ratings_centered = ratings_matrix - user_means.values.reshape(-1, 1)
        
        # SVD
        U, sigma, Vt = svds(ratings_centered, k=self.n_components)
        
        # Reconstruct
        self.user_factors = U
        self.item_factors = Vt.T
        
        return self
    
    def predict(self, user_idx, item_idx):
        """Predict rating"""
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
```

### Non-Negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import NMF

class NMFRecommender:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.model = NMF(n_components=n_components, random_state=42)
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, ratings_matrix):
        """Fit NMF"""
        self.item_factors = self.model.fit_transform(ratings_matrix)
        self.user_factors = self.model.components_.T
        return self
    
    def predict(self, user_idx, item_idx):
        """Predict rating"""
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
```

---

## Deep Learning for Recommendations

### Neural Collaborative Filtering

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_ncf_model(num_users, num_items, embedding_dim=50):
    """Neural Collaborative Filtering model"""
    
    # User embedding
    user_input = layers.Input(shape=(), name='user_id')
    user_embedding = layers.Embedding(num_users, embedding_dim)(user_input)
    user_vec = layers.Flatten()(user_embedding)
    
    # Item embedding
    item_input = layers.Input(shape=(), name='item_id')
    item_embedding = layers.Embedding(num_items, embedding_dim)(item_input)
    item_vec = layers.Flatten()(item_embedding)
    
    # Concatenate
    concat = layers.Concatenate()([user_vec, item_vec])
    
    # MLP layers
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model([user_input, item_input], output)
    return model

# Usage
model = create_ncf_model(num_users=1000, num_items=5000)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    [user_ids, item_ids],
    ratings,
    epochs=10,
    batch_size=256,
    validation_split=0.2
)
```

### Wide & Deep Model

```python
def create_wide_deep_model(num_users, num_items, num_features):
    """Wide & Deep model for recommendations"""
    
    # Wide part (linear)
    user_input = layers.Input(shape=(), name='user_id')
    item_input = layers.Input(shape=(), name='item_id')
    
    user_embedding = layers.Embedding(num_users, 1)(user_input)
    item_embedding = layers.Embedding(num_items, 1)(item_input)
    
    wide_output = layers.Add()([
        layers.Flatten()(user_embedding),
        layers.Flatten()(item_embedding)
    ])
    
    # Deep part (non-linear)
    user_embedding_deep = layers.Embedding(num_users, 50)(user_input)
    item_embedding_deep = layers.Embedding(num_items, 50)(item_input)
    
    user_vec = layers.Flatten()(user_embedding_deep)
    item_vec = layers.Flatten()(item_embedding_deep)
    
    concat = layers.Concatenate()([user_vec, item_vec])
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    deep_output = layers.Dense(1)(x)
    
    # Combine wide and deep
    combined = layers.Add()([wide_output, deep_output])
    output = layers.Dense(1, activation='sigmoid')(combined)
    
    model = keras.Model([user_input, item_input], output)
    return model
```

---

## Evaluation Metrics

### Precision@K and Recall@K

```python
def precision_at_k(y_true, y_pred, k=10):
    """Precision at K"""
    y_pred_k = y_pred[:k]
    relevant = set(y_true)
    recommended = set(y_pred_k)
    
    if len(recommended) == 0:
        return 0
    
    return len(relevant & recommended) / len(recommended)

def recall_at_k(y_true, y_pred, k=10):
    """Recall at K"""
    y_pred_k = y_pred[:k]
    relevant = set(y_true)
    recommended = set(y_pred_k)
    
    if len(relevant) == 0:
        return 0
    
    return len(relevant & recommended) / len(relevant)

# Usage
precision = precision_at_k(true_items, recommended_items, k=10)
recall = recall_at_k(true_items, recommended_items, k=10)
```

### Mean Average Precision (MAP)

```python
def average_precision(y_true, y_pred):
    """Average Precision"""
    relevant = set(y_true)
    if len(relevant) == 0:
        return 0
    
    precisions = []
    relevant_count = 0
    
    for i, item in enumerate(y_pred, 1):
        if item in relevant:
            relevant_count += 1
            precisions.append(relevant_count / i)
    
    if len(precisions) == 0:
        return 0
    
    return np.mean(precisions)

def mean_average_precision(y_true_list, y_pred_list):
    """Mean Average Precision"""
    aps = [average_precision(true, pred) for true, pred in zip(y_true_list, y_pred_list)]
    return np.mean(aps)
```

### NDCG (Normalized Discounted Cumulative Gain)

```python
def dcg(relevance_scores, k=None):
    """Discounted Cumulative Gain"""
    if k:
        relevance_scores = relevance_scores[:k]
    
    scores = np.array(relevance_scores)
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return np.sum(scores / discounts)

def ndcg(y_true, y_pred, k=10):
    """Normalized DCG"""
    y_pred_k = y_pred[:k]
    
    # Get relevance scores
    relevance = [1 if item in y_true else 0 for item in y_pred_k]
    
    # Calculate DCG
    dcg_score = dcg(relevance)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg_score = dcg(ideal_relevance)
    
    if idcg_score == 0:
        return 0
    
    return dcg_score / idcg_score
```

---

## Cold Start Problem

### New User Cold Start

```python
def recommend_for_new_user(user_features, content_model, n=10):
    """Recommend for new user using content-based"""
    # Use content-based filtering
    recommendations = content_model.recommend_by_features(user_features, n=n)
    return recommendations
```

### New Item Cold Start

```python
def recommend_new_item(item_features, content_model, n_users=10):
    """Find users who might like new item"""
    # Find similar items
    similar_items = content_model.find_similar(item_features)
    
    # Get users who liked similar items
    users = set()
    for similar_item in similar_items:
        item_users = ratings_df[ratings_df['item_id'] == similar_item]['user_id'].unique()
        users.update(item_users)
    
    return list(users)[:n_users]
```

---

## Practical Examples

### Example 1: Movie Recommendation System

```python
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Load data
ratings_df = pd.read_csv('ratings.csv')

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# Split data
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
algo = SVD()
algo.fit(trainset)

# Predict
predictions = algo.test(testset)
accuracy.rmse(predictions)

# Recommend for user
def recommend_movies(user_id, n=10):
    """Recommend movies for user"""
    user_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique()
    all_movies = ratings_df['movie_id'].unique()
    unrated_movies = set(all_movies) - set(user_movies)
    
    predictions = []
    for movie_id in unrated_movies:
        pred = algo.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [movie_id for movie_id, rating in predictions[:n]]
```

### Example 2: E-commerce Product Recommendations

```python
class ProductRecommender:
    def __init__(self):
        self.cf_model = None
        self.cb_model = None
        self.hybrid_model = None
    
    def fit(self, ratings_df, products_df):
        """Fit all models"""
        # Collaborative filtering
        self.cf_model = ItemBasedCF()
        self.cf_model.fit(ratings_df)
        
        # Content-based
        self.cb_model = ContentBasedRecommender()
        self.cb_model.fit(products_df, text_column='description')
        
        # Hybrid
        self.hybrid_model = HybridRecommender(
            self.cf_model,
            self.cb_model,
            cf_weight=0.7
        )
    
    def recommend(self, user_id, n=10):
        """Get recommendations"""
        return self.hybrid_model.recommend(user_id, n=n)
```

---

## Best Practices

1. **Handle Cold Start**: Use content-based for new users/items
2. **Diversity**: Ensure recommendations are diverse
3. **Explanations**: Explain why items are recommended
4. **Evaluation**: Use multiple metrics (Precision, Recall, NDCG)
5. **A/B Testing**: Test different approaches
6. **Scalability**: Consider computational efficiency
7. **Privacy**: Respect user privacy
8. **Bias**: Monitor for bias and fairness

---

## Resources

- **Surprise**: scikit-surprise.readthedocs.io
- **Implicit**: github.com/benfred/implicit
- **Papers**: 
  - Collaborative Filtering (1992)
  - Matrix Factorization (2009)
  - Neural CF (2017)

---

## Conclusion

Recommendation systems are crucial for many applications. Key takeaways:

1. **Start with Collaborative Filtering**: Often works well
2. **Use Hybrid Approaches**: Combine multiple methods
3. **Handle Cold Start**: Use content-based for new users/items
4. **Evaluate Properly**: Use appropriate metrics
5. **Consider Scalability**: Design for production

Remember: Good recommendations improve user experience and business metrics!

