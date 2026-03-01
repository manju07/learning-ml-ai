# Recommendation Systems: Complete Guide

## Table of Contents
1. [Introduction to Recommendation Systems](#introduction-to-recommendation-systems)
2. [Collaborative Filtering](#collaborative-filtering)
3. [Content-Based Filtering](#content-based-filtering)
4. [Hybrid Methods](#hybrid-methods)
5. [Matrix Factorization](#matrix-factorization)
6. [Deep Learning for Recommendations](#deep-learning-for-recommendations)
7. [Two-Tower and Dual-Encoder Models](#two-tower-and-dual-encoder-models)
8. [Multi-Task Learning for Recommendations](#multi-task-learning-for-recommendations)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Cold Start Problem](#cold-start-problem)
11. [Pitfalls and Failure Modes](#pitfalls-and-failure-modes)
12. [Benchmarks and Datasets](#benchmarks-and-datasets)
13. [Practical Examples](#practical-examples)
14. [Best Practices](#best-practices)

---

## Introduction to Recommendation Systems

Recommendation systems predict user preferences and suggest items they might like. They're used by Netflix, Amazon, Spotify, and many other platforms. At their core, recsys address an **information overload** problem: given millions of items and sparse user feedback, how do we surface the right items to the right users at the right time?

**Key paradigms**: (1) **Retrieval** (candidate generation)—reduce millions of items to hundreds; (2) **Ranking**—order candidates by relevance; (3) **Re-ranking**—apply business constraints, diversity, fairness. Industrial systems often use a funnel: retrieval → ranking → re-ranking.

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

Collaborative filtering (CF) assumes **users who agreed in the past will agree in the future**. It relies purely on interaction data (ratings, clicks, purchases) without item attributes. CF excels when behavior is rich and item features are missing or noisy. **Sparsity** is the main challenge: the user-item matrix is typically 99%+ empty.

### User-Based Collaborative Filtering

Finds users similar to the target user and recommends items those similar users liked. Works well when user bases are stable and preferences are coherent. **Scaling issue**: similarity computation is O(n²) in number of users.

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

Uses item-item similarity: "users who liked X also liked Y." Often preferred over user-based because items are more stable than users, and item catalogs change less frequently. Better for **scalability** when items < users. Used by Amazon's "Customers who bought this also bought."

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

Matrix factorization assumes the rating matrix **R ≈ U × Vᵀ**: users and items lie in a latent space of dimension *k*. Each user is a *k*-dim vector; each item is a *k*-dim vector; predicted rating = dot product. This **latent factor** interpretation: dimensions can capture "action vs comedy," "depth vs entertainment," etc., even without explicit features.

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

## Pitfalls and Failure Modes

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Filter bubbles** | Users only see similar content; diversity drops | Add diversity/exploration in ranking, MMR, slate diversity |
| **Popularity bias** | Popular items dominate; long-tail items never surface | Inverse propensity scoring, calibration, exposure-aware loss |
| **Cold start** | New users/items have no or few interactions | Content-based, transfer learning, exploration (bandits) |
| **Data leakage** | Using future info (e.g., labels from post-click) in training | Strict temporal splits, avoid future features |
| **Evaluation mismatch** | Offline metrics (NDCG) ≠ online metrics (CTR, revenue) | A/B test; use unbiased offline estimators (IPS) when possible |
| **Sparse feedback** | Implicit feedback is noisy (clicks ≠ likes) | Handle position bias, use dwell time, multiple signals |
| **Drift** | User preferences and item catalog change over time | Continuous retraining, online learning, periodic full retrains |
| **Scalability** | CF similarity O(n²); ranking every pair O(n·m) | Two-tower retrieval, approximate nearest neighbors, caching |

**Position bias**: Items higher in the list get more clicks regardless of relevance. Correct with **inverse propensity weighting** or **position-aware** models.

---

## Benchmarks and Datasets

| Dataset | Scale | Task | Notes |
|---------|-------|------|-------|
| **MovieLens** (1M, 25M) | 6K–162K users, 4K–62K movies | Explicit ratings | Classical benchmark; cold-start variants |
| **Netflix Prize** | 480K users, 18K movies | RMSE on ratings | Legacy; 10% improvement prize |
| **Amazon Reviews** | Millions of users/items | Rating, purchase prediction | Multiple categories (Books, Electronics) |
| **YouTube-8M** | 8M videos, 386K labels | Video recommendation | Large-scale, multi-label |
| **MIND** (Microsoft) | News | CTR, diversity | News recommendation |
| **RecSys Challenge** | Varies | Yearly competition | Real-world industrial data |

**Typical metrics by task**:
- **Rating prediction**: RMSE, MAE
- **Ranking**: NDCG@K, MAP@K, MRR, Precision@K, Recall@K
- **Implicit/CTR**: AUC, Log Loss, Precision/Recall
- **Diversity**: Coverage, Gini, ILD (Intra-List Diversity)

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

**Libraries**:
- **Surprise**: scikit-surprise.readthedocs.io — SVD, KNN, etc.
- **Implicit**: github.com/benfred/implicit — ALS for implicit feedback
- **LightFM**: github.com/lyst/lightfm — Hybrid matrix factorization
- **RecBole**: recbole.io — Comprehensive recsys toolkit

**Papers & References**:
- Resnick et al. (1994) — GroupLens: collaborative filtering
- Koren et al. (2009) — Matrix factorization techniques (Netflix)
- He et al. (2017) — Neural Collaborative Filtering
- Cheng et al. (2016) — Wide & Deep
- Yi et al. (2019) — Sampling-bias-corrected NCE for two-tower
- Ma et al. (2018) — MMoE for multi-task learning
- **RecSys Handbook** (Ricci et al.) — Comprehensive textbook

---

## Conclusion

Recommendation systems are crucial for many applications. Key takeaways:

1. **Start with Collaborative Filtering**: Often works well
2. **Use Hybrid Approaches**: Combine multiple methods
3. **Handle Cold Start**: Use content-based for new users/items
4. **Evaluate Properly**: Use appropriate metrics
5. **Consider Scalability**: Design for production

Remember: Good recommendations improve user experience and business metrics!

### Summary of Enhancements

- **Deeper concepts**: Retrieval vs ranking vs re-ranking, sparsity, position bias
- **Two-tower models**: Dual-encoder architecture for scalable retrieval with ANN
- **Multi-task learning**: CTR/CVR and other MTL patterns (shared-bottom, MMoE)
- **Pitfalls**: Filter bubbles, popularity bias, cold start, evaluation mismatch, drift
- **Benchmarks**: MovieLens, Netflix, Amazon, MIND, RecSys Challenge
- **References**: Key papers and libraries (LightFM, RecBole)

