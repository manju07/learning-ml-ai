# Data Science: Comprehensive Guide from Basics to Advanced

## Table of Contents
1. [Data Science Lifecycle](#1-data-science-lifecycle)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Pandas: Advanced Operations](#3-pandas-advanced-operations)
4. [Data Quality and Cleaning](#4-data-quality-and-cleaning)
5. [Feature Engineering](#5-feature-engineering)
6. [Feature Selection](#6-feature-selection)
7. [Imbalanced Data Handling](#7-imbalanced-data-handling)
8. [Statistical Analysis and Hypothesis Testing](#8-statistical-analysis-and-hypothesis-testing)
9. [Data Visualization](#9-data-visualization)
10. [SQL for Data Science](#10-sql-for-data-science)
11. [Data Pipelines: Pandas, Dask, Polars](#11-data-pipelines-pandas-dask-polars)
12. [Experiment Tracking Basics](#12-experiment-tracking-basics)
13. [Data Versioning with DVC](#13-data-versioning-with-dvc)
14. [Data Lineage and Provenance](#14-data-lineage-and-provenance)
15. [Feature Stores](#15-feature-stores)
16. [Full EDA Example: Titanic Dataset](#16-full-eda-example-titanic-dataset)
17. [Full EDA Example: Housing Dataset](#17-full-eda-example-housing-dataset)

---

## 1. Data Science Lifecycle

### 1.1 CRISP-DM (Cross-Industry Standard Process for Data Mining)

CRISP-DM is the most widely used data science methodology, providing a structured approach to planning and executing data science projects.

```
┌─────────────────────────────────────────────────────────────┐
│                    CRISP-DM Process                          │
│                                                             │
│    ┌──────────┐                                             │
│    │ Business │◄──────────────────────────┐                │
│    │Understanding│                        │                │
│    └────┬─────┘                           │                │
│         │                                 │                │
│         ▼                                 │                │
│    ┌──────────┐      ┌──────────┐         │                │
│    │  Data    │◄────►│  Data    │         │                │
│    │Understanding│   │Preparation│        │                │
│    └────┬─────┘      └────┬─────┘         │                │
│         │                │                │                │
│         ▼                ▼                │                │
│    ┌──────────┐      ┌──────────┐         │                │
│    │ Modeling │◄────►│Evaluation│─────────┘                │
│    └────┬─────┘      └──────────┘                          │
│         │                                                   │
│         ▼                                                   │
│    ┌──────────┐                                             │
│    │Deployment│                                             │
│    └──────────┘                                             │
└─────────────────────────────────────────────────────────────┘
```

**Phase 1: Business Understanding**
- Define the business objective and success criteria
- Identify constraints (budget, timeline, data availability)
- Convert business problem to data mining goal
- Create project plan with milestones

**Phase 2: Data Understanding**
- Collect initial data from all sources
- Perform initial data exploration
- Verify data quality (completeness, consistency, accuracy)
- Discover initial insights

**Phase 3: Data Preparation**
- Select relevant data
- Clean data (handle missing values, outliers)
- Construct new features
- Integrate and format data for modeling

**Phase 4: Modeling**
- Select modeling technique based on problem type
- Create test design (train/val/test split strategy)
- Build multiple models
- Assess models internally

**Phase 5: Evaluation**
- Evaluate results against business objectives
- Review the entire process for potential issues
- Determine next steps: deploy or iterate

**Phase 6: Deployment**
- Plan and execute deployment
- Create monitoring and maintenance plan
- Produce final report
- Review lessons learned

**CRISP-DM in practice** — Phase-specific tips:
- **Business Understanding**: Define success metrics upfront (e.g., AUC > 0.85, latency < 50ms). Avoid vague goals like "improve conversions."
- **Data Understanding**: Profile data early; detect schema drift, missingness patterns, and distribution shifts. Document data dictionary.
- **Data Preparation**: Version preprocessing code; use train/test splits before any transformation to prevent leakage.
- **Modeling**: Start simple (baseline). Use cross-validation and hold out a final test set until the end.
- **Evaluation**: Check against business metrics, not just ML metrics. Run bias/fairness analysis.
- **Deployment**: Plan for model monitoring, retraining triggers, and rollback.

**Common CRISP-DM mistakes**: Skipping business alignment, treating phases as strictly sequential (they iterate), not documenting decisions, overfitting to a single holdout, deploying without monitoring.

### 1.2 TDSP (Team Data Science Process) - Microsoft's Framework

TDSP is Microsoft's agile, iterative framework optimized for team collaboration.

```python
# TDSP Project Structure
"""
project_name/
├── Code/                      # All code artifacts
│   ├── DataAcquisition/       # Data ingestion scripts
│   ├── Preprocessing/         # Data preprocessing
│   ├── Modeling/              # Model training and evaluation
│   └── Operationalization/    # Deployment code
├── Data/
│   ├── Raw/                   # Original immutable data
│   ├── Processed/             # Cleaned, transformed data
│   └── Modeling/              # Features for modeling
├── Docs/
│   ├── Project/               # Project charter, requirements
│   ├── Data/                  # Data dictionary, reports
│   └── Delivery/              # Final reports, presentations
├── Sample_Data/               # Sample datasets for testing
├── Utilities/                 # Utility scripts
└── README.md
"""
```

**TDSP Lifecycle Stages:**
1. **Business Understanding**: Charter, data sources, stakeholder identification
2. **Data Acquisition & Understanding**: Ingestion, profiling, quality analysis
3. **Modeling**: Feature engineering, model training, model evaluation
4. **Deployment**: Scoring pipeline, acceptance testing, handoff
5. **Customer Acceptance**: System validation, project close-out

### 1.3 Cookiecutter Data Science Project Template

```bash
# Install cookiecutter
pip install cookiecutter

# Create project from template
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

Standard project structure:
```
project/
├── data/
│   ├── external/        # Data from third party sources
│   ├── interim/         # Intermediate transformed data
│   ├── processed/       # Final datasets for modeling
│   └── raw/             # Original, immutable data dump
├── models/              # Trained and serialized models
├── notebooks/           # Jupyter notebooks (numbered for ordering)
│   ├── 1.0-eda.ipynb
│   ├── 2.0-preprocessing.ipynb
│   └── 3.0-modeling.ipynb
├── reports/
│   └── figures/         # Generated graphics for reports
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/
│       └── visualize.py
├── Makefile             # Makefile with commands for data/modeling
├── requirements.txt
└── setup.py
```

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Univariate Analysis

Univariate analysis examines each variable in isolation to understand its distribution and characteristics.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def univariate_analysis(df, column):
    """
    Comprehensive univariate analysis for a single column.
    Handles both numeric and categorical data.
    """
    print(f"\n{'='*60}")
    print(f"Univariate Analysis: {column}")
    print(f"{'='*60}")

    if df[column].dtype in ['int64', 'float64']:
        # Numeric column
        desc = df[column].describe()
        print(f"\nDescriptive Statistics:")
        print(desc)
        print(f"\nSkewness: {df[column].skew():.4f}")
        print(f"Kurtosis: {df[column].kurtosis():.4f}")
        print(f"Missing: {df[column].isnull().sum()} ({df[column].isnull().mean()*100:.2f}%)")

        # IQR and outlier detection
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower) | (df[column] > upper)][column]
        print(f"\nIQR: {IQR:.4f}")
        print(f"Outlier bounds: [{lower:.4f}, {upper:.4f}]")
        print(f"Outlier count: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

        # Normality test (Shapiro-Wilk for small samples, D'Agostino for large)
        if len(df[column].dropna()) <= 5000:
            stat, p = stats.shapiro(df[column].dropna())
            print(f"\nShapiro-Wilk test: stat={stat:.4f}, p={p:.6f}")
        else:
            stat, p = stats.normaltest(df[column].dropna())
            print(f"\nD'Agostino-Pearson test: stat={stat:.4f}, p={p:.6f}")
        print(f"Normal distribution: {'Yes' if p > 0.05 else 'No'}")

        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Univariate Analysis: {column}', fontsize=16, fontweight='bold')

        # Histogram with KDE
        df[column].hist(ax=axes[0, 0], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        df[column].plot.kde(ax=axes[0, 0].twinx(), color='red')
        axes[0, 0].set_title('Histogram + KDE')

        # Box plot
        axes[0, 1].boxplot(df[column].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[0, 1].set_title('Box Plot')

        # QQ plot (normality check)
        stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')

        # CDF
        sorted_data = np.sort(df[column].dropna())
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cdf, color='steelblue')
        axes[1, 1].set_title('Cumulative Distribution Function')

        plt.tight_layout()
        plt.show()

    else:
        # Categorical column
        value_counts = df[column].value_counts()
        print(f"\nValue Counts:")
        print(value_counts)
        print(f"\nUnique values: {df[column].nunique()}")
        print(f"Missing: {df[column].isnull().sum()} ({df[column].isnull().mean()*100:.2f}%)")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        value_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
        axes[0].set_title(f'Bar Chart: {column}')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

        axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        axes[1].set_title(f'Pie Chart: {column}')
        plt.tight_layout()
        plt.show()
```

### 2.2 Bivariate Analysis

Bivariate analysis examines the relationship between two variables.

```python
def bivariate_analysis(df, col1, col2):
    """
    Comprehensive bivariate analysis between two columns.
    """
    print(f"\n{'='*60}")
    print(f"Bivariate Analysis: {col1} vs {col2}")
    print(f"{'='*60}")

    t1 = df[col1].dtype in ['int64', 'float64']
    t2 = df[col2].dtype in ['int64', 'float64']

    if t1 and t2:
        # Both numeric: scatter plot + correlation
        pearson_r, pearson_p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
        spearman_r, spearman_p = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
        print(f"\nPearson correlation: r={pearson_r:.4f}, p={pearson_p:.6f}")
        print(f"Spearman correlation: r={spearman_r:.4f}, p={spearman_p:.6f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot with regression line
        axes[0].scatter(df[col1], df[col2], alpha=0.5, color='steelblue')
        m, b = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
        x_line = np.linspace(df[col1].min(), df[col1].max(), 100)
        axes[0].plot(x_line, m * x_line + b, 'r-', linewidth=2)
        axes[0].set_xlabel(col1)
        axes[0].set_ylabel(col2)
        axes[0].set_title(f'Scatter Plot\nr={pearson_r:.3f}')

        # Hexbin for dense data
        axes[1].hexbin(df[col1].dropna(), df[col2].dropna(), gridsize=30, cmap='Blues')
        plt.colorbar(axes[1].collections[0], ax=axes[1])
        axes[1].set_title('Hexbin Density Plot')
        plt.tight_layout()
        plt.show()

    elif t1 and not t2:
        # Numeric vs categorical: grouped box plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df.boxplot(column=col1, by=col2, ax=axes[0])
        axes[0].set_title(f'{col1} by {col2}')

        # Violin plot
        df.groupby(col2)[col1].apply(list)
        groups = [group.values for name, group in df.groupby(col2)[col1]]
        labels = df[col2].unique()
        axes[1].violinplot(groups, showmeans=True, showmedians=True)
        axes[1].set_xticks(range(1, len(labels) + 1))
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].set_title(f'Violin Plot: {col1} by {col2}')
        plt.tight_layout()
        plt.show()

        # ANOVA test
        groups_list = [df[df[col2] == cat][col1].dropna() for cat in df[col2].unique()]
        f_stat, p_value = stats.f_oneway(*groups_list)
        print(f"\nANOVA: F={f_stat:.4f}, p={p_value:.6f}")
        print(f"Groups significantly different: {'Yes' if p_value < 0.05 else 'No'}")

    else:
        # Both categorical: contingency table + chi-squared
        contingency = pd.crosstab(df[col1], df[col2])
        print(f"\nContingency Table:")
        print(contingency)
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-squared: {chi2:.4f}, p={p:.6f}, dof={dof}")
        print(f"Association significant: {'Yes' if p < 0.05 else 'No'}")

        # Stacked bar chart
        contingency.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Stacked Bar: {col1} vs {col2}')
        plt.tight_layout()
        plt.show()


def correlation_analysis(df, method='pearson', threshold=0.5):
    """
    Comprehensive correlation analysis with heatmap.

    Parameters:
    -----------
    method : 'pearson', 'spearman', or 'kendall'
    threshold : threshold for highlighting strong correlations
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method=method)

    # Find highly correlated pairs
    print(f"\nHighly correlated pairs (|r| > {threshold}):")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], r))
                print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {r:.4f}")

    # Heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()

    return corr_matrix, corr_pairs
```

### 2.3 Multivariate Analysis

```python
def multivariate_analysis(df, target_col=None):
    """
    Multivariate EDA: pair plots, PCA, dimensionality visualization.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Pair plot
    if target_col and target_col in df.columns:
        sns.pairplot(df[numeric_cols], hue=target_col, diag_kind='kde',
                     plot_kws={'alpha': 0.5})
    else:
        sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Pair Plot', y=1.02, fontsize=16)
    plt.show()

    # PCA for dimensionality reduction and visualization
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols].dropna())

    pca = PCA(n_components=min(10, len(numeric_cols)))
    pca.fit(X_scaled)

    # Explained variance plot
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, color='steelblue')
    axes[0].plot(range(1, len(explained_var) + 1), cumulative_var, 'r-o', linewidth=2)
    axes[0].axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('PCA Explained Variance')
    axes[0].legend()

    # 2D PCA scatter
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, color='steelblue')
    axes[1].set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title('PCA: First Two Components')
    plt.tight_layout()
    plt.show()

    # Feature loadings for PCA interpretation
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        columns=['PC1', 'PC2'],
        index=numeric_cols
    )
    print("\nPCA Feature Loadings:")
    print(loadings.sort_values('PC1', ascending=False))

    return pca, X_pca, loadings
```

---

## 3. Pandas: Advanced Operations

### 3.1 GroupBy Operations

```python
import pandas as pd
import numpy as np

# Sample e-commerce dataset
df = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 1, 2],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing', 'Electronics', 'Food', 'Clothing'],
    'amount': [100, 50, 200, 30, 80, 150, 25, 90],
    'date': pd.date_range('2024-01-01', periods=8),
    'quantity': [1, 2, 1, 3, 2, 1, 5, 1]
})

# Basic groupby
print("Total amount by category:")
print(df.groupby('category')['amount'].sum())

# Multiple aggregations on multiple columns
print("\nMultiple aggregations:")
agg_result = df.groupby('category').agg({
    'amount': ['sum', 'mean', 'std', 'count', 'min', 'max'],
    'quantity': ['sum', 'mean']
})
# Flatten multi-level columns
agg_result.columns = ['_'.join(col).strip() for col in agg_result.columns]
print(agg_result)

# Named aggregations (pandas >= 0.25)
print("\nNamed aggregations:")
named_agg = df.groupby('category').agg(
    total_revenue=('amount', 'sum'),
    avg_order=('amount', 'mean'),
    order_count=('amount', 'count'),
    total_units=('quantity', 'sum')
).reset_index()
print(named_agg)

# Custom aggregation function
def revenue_per_unit(x):
    return x['amount'].sum() / x['quantity'].sum()

print("\nRevenue per unit:")
print(df.groupby('category').apply(revenue_per_unit))

# Transform: group-level statistics back to original index
df['user_total'] = df.groupby('user_id')['amount'].transform('sum')
df['user_avg'] = df.groupby('user_id')['amount'].transform('mean')
df['amount_vs_user_avg'] = df['amount'] - df['user_avg']  # Deviation from user mean
print("\nTransform results:")
print(df[['user_id', 'amount', 'user_total', 'user_avg', 'amount_vs_user_avg']])

# Filter: keep only groups meeting a condition
high_value_categories = df.groupby('category').filter(lambda x: x['amount'].mean() > 80)
print("\nCategories with avg amount > 80:")
print(high_value_categories['category'].unique())

# Cumulative operations within groups
df_sorted = df.sort_values(['user_id', 'date'])
df_sorted['cumulative_spend'] = df_sorted.groupby('user_id')['amount'].cumsum()
df_sorted['rolling_avg'] = df_sorted.groupby('user_id')['amount'].transform(
    lambda x: x.rolling(window=2, min_periods=1).mean()
)
print("\nCumulative operations:")
print(df_sorted[['user_id', 'date', 'amount', 'cumulative_spend', 'rolling_avg']])

# Pivot aggregation using groupby
cross_tab = df.groupby(['user_id', 'category'])['amount'].sum().unstack(fill_value=0)
print("\nCross-tabulation (user x category spend):")
print(cross_tab)
```

### 3.2 Pivot Tables

```python
# Pivot table - the most flexible aggregation tool
pivot = pd.pivot_table(
    df,
    values=['amount', 'quantity'],
    index='user_id',
    columns='category',
    aggfunc={'amount': 'sum', 'quantity': 'mean'},
    fill_value=0,
    margins=True,   # Add row/column totals
    margins_name='Total'
)
print("Pivot table:")
print(pivot)

# Pivot (reshape without aggregation - must be unique index/column pairs)
df_unique = df.drop_duplicates(subset=['user_id', 'category'])
pivoted = df_unique.pivot(index='user_id', columns='category', values='amount')
print("\nSimple pivot:")
print(pivoted)

# Melt: wide to long format (inverse of pivot)
wide_df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'jan_sales': [100, 200, 150],
    'feb_sales': [120, 180, 160],
    'mar_sales': [130, 220, 140]
})
long_df = wide_df.melt(
    id_vars=['user_id'],
    value_vars=['jan_sales', 'feb_sales', 'mar_sales'],
    var_name='month',
    value_name='sales'
)
long_df['month'] = long_df['month'].str.replace('_sales', '')
print("\nMelted (wide to long):")
print(long_df)

# Stack and unstack
stacked = pivot['amount'].stack()
print("\nStacked pivot:")
print(stacked.head(10))
```

### 3.3 Merge and Join Operations

```python
# Sample dataframes
users = pd.DataFrame({
    'user_id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Dave'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'user_id': [1, 2, 1, 3, 5],  # user 5 doesn't exist in users
    'amount': [50, 100, 75, 200, 30]
})

products = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 106],  # order 106 doesn't exist
    'product': ['Laptop', 'Phone', 'Tablet', 'Camera', 'Watch']
})

# Inner join: only matching rows
inner = pd.merge(orders, users, on='user_id', how='inner')
print("Inner join (matched users only):")
print(inner)

# Left join: all orders, matched users where available
left = pd.merge(orders, users, on='user_id', how='left')
print("\nLeft join (all orders):")
print(left)

# Right join: all users, matched orders where available
right = pd.merge(orders, users, on='user_id', how='right')
print("\nRight join (all users):")
print(right)

# Outer join: all rows from both
outer = pd.merge(orders, users, on='user_id', how='outer', indicator=True)
print("\nOuter join with indicator:")
print(outer)

# Merge on different column names
orders_with_city = pd.merge(
    orders, users,
    left_on='user_id', right_on='user_id',
    suffixes=('_order', '_user')
)

# Multi-key merge
df_a = pd.DataFrame({'key1': ['A', 'B', 'C'], 'key2': [1, 2, 1], 'val_a': [10, 20, 30]})
df_b = pd.DataFrame({'key1': ['A', 'B', 'C'], 'key2': [1, 2, 2], 'val_b': [100, 200, 300]})
multi_key_merge = pd.merge(df_a, df_b, on=['key1', 'key2'], how='outer')
print("\nMulti-key merge:")
print(multi_key_merge)

# Concatenation
dfs_to_concat = [
    pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
    pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
]
concatenated = pd.concat(dfs_to_concat, ignore_index=True)

# Merge asof (for time series / sorted merges)
import pandas as pd
df_prices = pd.DataFrame({
    'time': pd.to_datetime(['2024-01-01 09:00', '2024-01-01 09:30', '2024-01-01 10:00']),
    'price': [100, 102, 98]
})
df_trades = pd.DataFrame({
    'time': pd.to_datetime(['2024-01-01 09:15', '2024-01-01 09:45']),
    'trade_amount': [500, 300]
})
# Merge each trade with the most recent price at or before trade time
df_merged = pd.merge_asof(
    df_trades.sort_values('time'),
    df_prices.sort_values('time'),
    on='time',
    direction='backward'
)
print("\nMerge asof (time-based):")
print(df_merged)
```

### 3.4 Reshaping and Time Series Operations

```python
# Reshaping
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=365),
    'store': np.random.choice(['StoreA', 'StoreB', 'StoreC'], 365),
    'sales': np.random.normal(1000, 200, 365)
})

# Time series operations
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Resample to monthly
monthly = df.groupby('store')['sales'].resample('ME').sum().reset_index()

# Rolling statistics
df['rolling_7d'] = df.groupby('store')['sales'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df['rolling_30d_std'] = df.groupby('store')['sales'].transform(
    lambda x: x.rolling(30, min_periods=1).std()
)

# Lag features
df['sales_lag1'] = df.groupby('store')['sales'].shift(1)
df['sales_lag7'] = df.groupby('store')['sales'].shift(7)
df['sales_lag30'] = df.groupby('store')['sales'].shift(30)

# Expanding window (cumulative)
df['cumulative_sales'] = df.groupby('store')['sales'].cumsum()

# Period over period changes
df['pct_change_1d'] = df.groupby('store')['sales'].pct_change(1)
df['pct_change_7d'] = df.groupby('store')['sales'].pct_change(7)

# Exponentially weighted moving average
df['ewm_sales'] = df.groupby('store')['sales'].transform(
    lambda x: x.ewm(span=7).mean()
)

# Date feature extraction
df_reset = df.reset_index()
df_reset['year'] = df_reset['date'].dt.year
df_reset['month'] = df_reset['date'].dt.month
df_reset['day'] = df_reset['date'].dt.day
df_reset['dayofweek'] = df_reset['date'].dt.dayofweek
df_reset['quarter'] = df_reset['date'].dt.quarter
df_reset['week'] = df_reset['date'].dt.isocalendar().week
df_reset['is_weekend'] = df_reset['dayofweek'].isin([5, 6]).astype(int)
df_reset['is_month_start'] = df_reset['date'].dt.is_month_start.astype(int)
df_reset['is_month_end'] = df_reset['date'].dt.is_month_end.astype(int)
```

---

## 4. Data Quality and Cleaning

### 4.1 Missing Value Analysis: MCAR, MAR, MNAR

Missing data mechanisms are critical to understand before imputation:

**MCAR (Missing Completely At Random):** Missingness is unrelated to any data.
- *Example*: Survey respondents randomly skip questions
- *Test*: Little's MCAR test
- *Treatment*: Any imputation method works; simple mean/median is acceptable

**MAR (Missing At Random):** Missingness depends on observed data, but not on the missing values themselves.
- *Example*: Younger respondents less likely to report income (age is observed)
- *Treatment*: Multiple imputation, model-based imputation

**MNAR (Missing Not At Random):** Missingness depends on the unobserved missing values themselves.
- *Example*: High earners less likely to report income
- *Treatment*: Requires domain knowledge, sensitivity analysis, or specialized models

```python
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import missingno as msno

def analyze_missing_values(df):
    """
    Comprehensive missing value analysis.
    Returns a detailed report and visualizations.
    """
    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    # Summary statistics
    missing = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct,
        'Dtype': df.dtypes
    }).sort_values('Missing %', ascending=False)
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    print(f"\nTotal rows: {len(df)}")
    print(f"Columns with missing values: {len(missing_df)}")
    print(f"\n{missing_df.to_string()}")

    # Pattern analysis
    if len(missing_df) > 1:
        print("\nMissingness Correlation (do columns go missing together?):")
        # Identify rows with missing values for each column
        miss_indicators = df[missing_df.index].isnull().astype(int)
        if len(miss_indicators.columns) > 1:
            miss_corr = miss_indicators.corr()
            print(miss_corr)

    # Visual analysis
    if len(missing_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart of missing percentages
        missing_pct[missing_pct > 0].sort_values().plot(
            kind='barh', ax=axes[0], color='steelblue'
        )
        axes[0].set_title('Missing Value Percentage by Column')
        axes[0].set_xlabel('Missing %')

        # Missing value heatmap (shows patterns)
        # msno.heatmap(df, ax=axes[1])  # requires missingno
        axes[1].imshow(df.isnull(), aspect='auto', cmap='RdYlGn_r')
        axes[1].set_title('Missing Value Patterns\n(Red = Missing, Green = Present)')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Samples')
        plt.tight_layout()
        plt.show()

    return missing_df


def impute_missing_values(df, strategy='auto'):
    """
    Smart missing value imputation based on data type and distribution.

    Strategies:
    - 'auto': Automatic selection based on data properties
    - 'simple': Mean/median/mode
    - 'knn': K-Nearest Neighbors imputation
    - 'iterative': MICE (Multiple Imputation by Chained Equations)
    """
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if strategy == 'simple' or strategy == 'auto':
        # For numeric columns: use median (robust to outliers)
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                skewness = abs(df[col].skew())
                if skewness > 1:  # Skewed distribution: use median
                    df_imputed[col] = df[col].fillna(df[col].median())
                else:  # Symmetric: use mean
                    df_imputed[col] = df[col].fillna(df[col].mean())

        # For categorical columns: use mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df_imputed[col] = df[col].fillna(df[col].mode()[0])

    elif strategy == 'knn':
        # KNN imputation (numeric only)
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
        df_imputed[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

        # Categorical: mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df_imputed[col] = df[col].fillna(df[col].mode()[0])

    elif strategy == 'iterative':
        # MICE imputation (Multiple Imputation by Chained Equations)
        from sklearn.ensemble import RandomForestRegressor
        iter_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            random_state=42
        )
        df_imputed[numeric_cols] = iter_imputer.fit_transform(df[numeric_cols])

        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df_imputed[col] = df[col].fillna(df[col].mode()[0])

    # Add missing indicator columns (captures MNAR information)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df_imputed[f'{col}_was_missing'] = df[col].isnull().astype(int)

    return df_imputed
```

### 4.2 Outlier Detection

```python
def detect_outliers(df, columns=None, methods=['iqr', 'zscore', 'isolation_forest']):
    """
    Multi-method outlier detection with ensemble voting.

    Methods:
    - IQR: Interquartile Range (univariate, robust)
    - Z-score: Standard score (assumes normality)
    - Isolation Forest: Tree-based (multivariate, handles high-dim data)
    - DBSCAN: Density-based (multivariate, detects clusters of outliers)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_flags = pd.DataFrame(index=df.index)

    # Method 1: IQR (univariate)
    if 'iqr' in methods:
        iqr_outliers = pd.Series(False, index=df.index)
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            iqr_outliers |= (df[col] < lower) | (df[col] > upper)
            print(f"IQR [{col}]: bounds=({lower:.2f}, {upper:.2f}), "
                  f"outliers={iqr_outliers.sum()}")
        outlier_flags['iqr'] = iqr_outliers

    # Method 2: Z-score (univariate, assumes normality)
    if 'zscore' in methods:
        z_outliers = pd.Series(False, index=df.index)
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            col_outliers = pd.Series(False, index=df[col].dropna().index)
            col_outliers[z_scores > 3] = True
            z_outliers = z_outliers | col_outliers.reindex(df.index, fill_value=False)
        outlier_flags['zscore'] = z_outliers
        print(f"\nZ-score outliers (|z| > 3): {z_outliers.sum()}")

    # Method 3: Modified Z-score (more robust, uses median)
    if 'modified_zscore' in methods:
        mz_outliers = pd.Series(False, index=df.index)
        for col in columns:
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))  # Median Absolute Deviation
            modified_z = 0.6745 * (df[col] - median) / mad
            mz_outliers |= np.abs(modified_z) > 3.5
        outlier_flags['modified_zscore'] = mz_outliers

    # Method 4: Isolation Forest (multivariate)
    if 'isolation_forest' in methods:
        X = df[columns].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Assumes 10% outliers
            random_state=42
        )
        iso_predictions = iso_forest.fit_predict(X_scaled)
        iso_outliers = pd.Series(iso_predictions == -1, index=X.index)
        outlier_flags['isolation_forest'] = iso_outliers.reindex(df.index, fill_value=False)
        print(f"\nIsolation Forest outliers: {iso_outliers.sum()}")

    # Ensemble: flag if majority of methods agree
    n_methods = len(outlier_flags.columns)
    outlier_flags['votes'] = outlier_flags.sum(axis=1)
    outlier_flags['is_outlier'] = outlier_flags['votes'] >= max(1, n_methods // 2 + 1)

    print(f"\nEnsemble outliers (majority vote): {outlier_flags['is_outlier'].sum()}")

    return outlier_flags


def handle_outliers(df, outlier_flags, strategy='cap'):
    """
    Handle outliers using various strategies.

    Strategies:
    - 'remove': Delete outlier rows
    - 'cap': Winsorize to IQR bounds (most common)
    - 'transform': Log or Box-Cox transform
    - 'keep': Keep as-is (for tree-based models)
    """
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if strategy == 'remove':
        df_clean = df[~outlier_flags['is_outlier']].reset_index(drop=True)
        print(f"Removed {outlier_flags['is_outlier'].sum()} outlier rows")

    elif strategy == 'cap':
        # Winsorize: cap values at percentile bounds
        for col in numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df_clean[col] = df[col].clip(lower=lower, upper=upper)
        print(f"Capped values at 1st/99th percentile")

    elif strategy == 'transform':
        for col in numeric_cols:
            if (df[col] > 0).all():
                df_clean[col] = np.log1p(df[col])
                print(f"Log-transformed {col}")

    return df_clean
```

---

## 5. Feature Engineering

### 5.1 Polynomial Features and Interaction Terms

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

# Polynomial features (captures non-linear relationships)
# For features [a, b], generates: [1, a, b, a², ab, b²]
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X[['feature1', 'feature2']])
poly_feature_names = poly.get_feature_names_out(['feature1', 'feature2'])
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

# Manual interaction terms (more control)
df['price_x_quantity'] = df['price'] * df['quantity']
df['price_per_unit'] = df['price'] / (df['quantity'] + 1e-6)
df['revenue_margin'] = (df['revenue'] - df['cost']) / (df['revenue'] + 1e-6)

# Ratio features
df['debt_to_income'] = df['debt'] / (df['income'] + 1e-6)
df['price_to_earnings'] = df['price'] / (df['earnings'] + 1e-6)

# Statistical aggregations as features
df['amount_mean_by_user'] = df.groupby('user_id')['amount'].transform('mean')
df['amount_std_by_user'] = df.groupby('user_id')['amount'].transform('std')
df['amount_rank_by_user'] = df.groupby('user_id')['amount'].rank(pct=True)
```

### 5.2 Target Encoding

Target encoding replaces categorical values with statistics of the target variable, capturing the relationship while reducing dimensionality.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class TargetEncoder:
    """
    Target encoding with cross-validation to prevent target leakage.

    Uses k-fold CV: for each fold, the encoding is computed from
    the other folds (training data), preventing information leakage.
    """

    def __init__(self, cols, smoothing=10, n_splits=5):
        """
        Parameters:
        -----------
        cols : list of column names to encode
        smoothing : smoothing factor (higher = more towards global mean)
        n_splits : number of CV folds
        """
        self.cols = cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.encodings = {}

    def _encode_column(self, series, target, encoding_map, global_mean):
        """Apply encoding with smoothing toward global mean."""
        # Smoothing formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)
        return series.map(encoding_map).fillna(global_mean)

    def fit_transform(self, X, y):
        """Fit encoder with cross-validation and transform training data."""
        X_encoded = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in self.cols:
            global_mean = y.mean()
            X_encoded[col + '_te'] = global_mean

            # Cross-validation encoding
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]

                # Compute statistics for training fold
                stats = y_train_fold.groupby(X_train_fold[col]).agg(['mean', 'count'])
                # Smoothed estimate
                smoothed = (stats['count'] * stats['mean'] + self.smoothing * global_mean) / \
                           (stats['count'] + self.smoothing)

                X_encoded.loc[X_encoded.index[val_idx], col + '_te'] = \
                    X.iloc[val_idx][col].map(smoothed).fillna(global_mean).values

            # Fit final encoding on full dataset for test transformation
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            self.encodings[col] = (stats['count'] * stats['mean'] + self.smoothing * global_mean) / \
                                  (stats['count'] + self.smoothing)
            self.encodings[col + '_global_mean'] = global_mean

        return X_encoded

    def transform(self, X):
        """Transform test data using learned encodings."""
        X_encoded = X.copy()
        for col in self.cols:
            global_mean = self.encodings[col + '_global_mean']
            X_encoded[col + '_te'] = X[col].map(self.encodings[col]).fillna(global_mean)
        return X_encoded


# Catboost-style ordered target encoding (leak-free on training data)
def ordered_target_encoding(df, col, target, alpha=10):
    """
    Ordered target encoding (CatBoost approach).
    Encodes row i using only rows 0..i-1 to prevent leakage.
    """
    df = df.copy().reset_index(drop=True)
    cumsum = df.groupby(col)[target].cumsum() - df[target]
    cumcount = df.groupby(col)[target].cumcount()
    global_mean = df[target].mean()
    encoding = (cumsum + alpha * global_mean) / (cumcount + alpha)
    return encoding
```

### 5.3 Bin Encoding and Quantile Binning

```python
# Equal-width binning
df['age_bin_equal'] = pd.cut(
    df['age'],
    bins=5,
    labels=['Very Young', 'Young', 'Middle', 'Senior', 'Old']
)

# Quantile binning (equal-frequency bins)
df['age_bin_quantile'] = pd.qcut(
    df['age'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

# Custom bins based on domain knowledge
df['credit_score_band'] = pd.cut(
    df['credit_score'],
    bins=[0, 580, 670, 740, 800, 850],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
)

# Binary binning (thresholding)
df['is_high_value'] = (df['lifetime_value'] > df['lifetime_value'].quantile(0.75)).astype(int)

# Optimal binning using WoE (Weight of Evidence) - common in credit risk
def woe_binning(df, feature, target, n_bins=10):
    """
    Weight of Evidence binning - optimal for logistic regression in credit.
    WoE = ln(P(Events) / P(Non-Events))
    IV (Information Value) measures predictive power.
    """
    df = df[[feature, target]].copy()
    df['bin'] = pd.qcut(df[feature], q=n_bins, duplicates='drop')

    grouped = df.groupby('bin')[target].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()

    grouped['dist_events'] = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events

    # Replace zeros with small value to avoid log(0)
    grouped['dist_events'] = grouped['dist_events'].clip(lower=1e-10)
    grouped['dist_non_events'] = grouped['dist_non_events'].clip(lower=1e-10)

    grouped['woe'] = np.log(grouped['dist_events'] / grouped['dist_non_events'])
    grouped['iv'] = (grouped['dist_events'] - grouped['dist_non_events']) * grouped['woe']

    iv = grouped['iv'].sum()
    print(f"\nInformation Value (IV) for {feature}: {iv:.4f}")
    print("IV Interpretation: <0.02=Useless, 0.02-0.1=Weak, 0.1-0.3=Medium, >0.3=Strong")
    print(grouped)
    return grouped, iv
```

---

## 6. Feature Selection

### 6.1 Correlation-Based Selection

```python
def correlation_feature_selection(df, target, threshold=0.05, corr_threshold=0.95):
    """
    Two-step correlation-based feature selection:
    1. Remove features with low target correlation
    2. Remove redundant highly correlated features
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Step 1: Correlation with target
    target_corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
    low_corr_features = target_corr[target_corr < threshold].index.tolist()
    print(f"Features with |correlation| < {threshold} with target:")
    print(low_corr_features)

    # Step 2: Remove highly correlated feature pairs (multicollinearity)
    feature_corr = numeric_df.drop(columns=[target]).corr().abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))

    redundant = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    print(f"\nHighly correlated features (|r| > {corr_threshold}) to remove:")
    print(redundant)

    selected = [col for col in numeric_df.columns
                if col not in low_corr_features and col not in redundant and col != target]
    return selected
```

### 6.2 Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def mutual_information_selection(X, y, task='classification', n_features=20):
    """
    Select features based on mutual information with target.
    MI captures non-linear relationships, unlike correlation.

    Mathematical formulation:
    I(X;Y) = Σ Σ P(x,y) * log[P(x,y) / (P(x)*P(y))]
    """
    if task == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mutual_information': mi_scores
    }).sort_values('mutual_information', ascending=False)

    print("Mutual Information Scores:")
    print(mi_df.head(20).to_string())

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(mi_df['feature'][:n_features], mi_df['mutual_information'][:n_features])
    plt.xlabel('Mutual Information Score')
    plt.title('Feature Selection: Mutual Information')
    plt.tight_layout()
    plt.show()

    selected = mi_df[mi_df['mutual_information'] > 0]['feature'].tolist()[:n_features]
    return selected, mi_df
```

### 6.3 RFECV (Recursive Feature Elimination with Cross-Validation)

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

def rfecv_selection(X, y, estimator=None, cv=5, scoring='roc_auc'):
    """
    Recursive Feature Elimination with Cross-Validation.
    Automatically selects the optimal number of features.

    RFECV algorithm:
    1. Train model on all features
    2. Rank features by importance
    3. Remove least important feature(s)
    4. Repeat until minimum features reached
    5. Select feature set with best CV score
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=estimator,
        step=1,                    # Remove 1 feature at a time
        cv=cv_strategy,
        scoring=scoring,
        min_features_to_select=1,
        n_jobs=-1
    )
    rfecv.fit(X, y)

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Best CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}")

    selected_features = X.columns[rfecv.support_].tolist()
    print(f"Selected features: {selected_features}")

    # Plot CV scores vs number of features
    plt.figure(figsize=(10, 5))
    mean_scores = rfecv.cv_results_['mean_test_score']
    std_scores = rfecv.cv_results_['std_test_score']
    plt.plot(range(1, len(mean_scores) + 1), mean_scores)
    plt.fill_between(
        range(1, len(mean_scores) + 1),
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.3
    )
    plt.axvline(x=rfecv.n_features_, color='r', linestyle='--',
                label=f'Optimal: {rfecv.n_features_} features')
    plt.xlabel('Number of Features')
    plt.ylabel(f'CV Score ({scoring})')
    plt.title('RFECV: Feature Count vs Performance')
    plt.legend()
    plt.show()

    return selected_features, rfecv
```

### 6.4 Boruta Algorithm

```python
# pip install boruta
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def boruta_feature_selection(X, y, max_iter=100, alpha=0.05):
    """
    Boruta algorithm: all-relevant feature selection.

    Unlike RFECV (which selects minimal set), Boruta finds ALL relevant features.

    Algorithm:
    1. Create 'shadow' features by shuffling each real feature
    2. Train random forest on real + shadow features
    3. A feature is relevant if it consistently outperforms the best shadow feature
    4. Uses statistical testing with Bonferroni correction
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        max_iter=max_iter,
        alpha=alpha,
        random_state=42
    )

    boruta.fit(X.values, y.values)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'support': boruta.support_,           # Confirmed relevant
        'support_weak': boruta.support_weak_, # Tentative
        'ranking': boruta.ranking_
    })

    confirmed = feature_importance[feature_importance['support']]['feature'].tolist()
    tentative = feature_importance[feature_importance['support_weak']]['feature'].tolist()

    print(f"Confirmed features ({len(confirmed)}): {confirmed}")
    print(f"Tentative features ({len(tentative)}): {tentative}")
    print(f"Rejected features: {X.shape[1] - len(confirmed) - len(tentative)}")

    return confirmed, tentative, feature_importance
```

---

## 7. Imbalanced Data Handling

### 7.1 Understanding Imbalanced Data

Class imbalance occurs when one class significantly outnumbers another (e.g., 99:1 ratio in fraud detection). Standard accuracy becomes misleading; a model predicting all negative achieves 99% accuracy.

```python
from collections import Counter
import matplotlib.pyplot as plt

def analyze_class_imbalance(y):
    """Analyze the degree of class imbalance."""
    counter = Counter(y)
    total = len(y)
    print("Class Distribution:")
    for cls, count in sorted(counter.items()):
        print(f"  Class {cls}: {count} ({count/total*100:.2f}%)")

    imbalance_ratio = max(counter.values()) / min(counter.values())
    print(f"\nImbalance ratio: {imbalance_ratio:.1f}:1")

    if imbalance_ratio < 3:
        print("Mild imbalance - standard techniques may work")
    elif imbalance_ratio < 10:
        print("Moderate imbalance - consider oversampling or class weights")
    else:
        print("Severe imbalance - requires careful treatment (SMOTE, cost-sensitive learning)")

    plt.bar([str(k) for k in sorted(counter.keys())], [counter[k] for k in sorted(counter.keys())])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()
```

### 7.2 SMOTE and Variants

```python
# pip install imbalanced-learn
from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE,
                                     SVMSMOTE, KMeansSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, TomekLinks,
                                      EditedNearestNeighbours, ClusterCentroids)
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE: Synthetic Minority Over-sampling TEchnique
# Creates synthetic samples along the line between a minority sample
# and one of its k nearest minority neighbors
smote = SMOTE(
    sampling_strategy='auto',  # Auto-balance to 1:1
    k_neighbors=5,
    random_state=42,
    n_jobs=-1
)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE: {Counter(y_train)} -> {Counter(y_smote)}")

# ADASYN: Adaptive Synthetic Sampling
# Generates more synthetic data for harder-to-classify examples
adasyn = ADASYN(
    sampling_strategy=0.5,  # Minority to be 50% of majority
    n_neighbors=5,
    random_state=42
)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)

# Borderline-SMOTE: focuses on borderline minority examples
borderline_smote = BorderlineSMOTE(
    kind='borderline-1',  # or 'borderline-2'
    k_neighbors=5,
    random_state=42
)

# SMOTE + Tomek Links (clean noisy majority samples after SMOTE)
smote_tomek = SMOTETomek(random_state=42)
X_combined, y_combined = smote_tomek.fit_resample(X_train, y_train)

# SMOTE + ENN (more aggressive cleaning)
smote_enn = SMOTEENN(random_state=42)
X_cleaned, y_cleaned = smote_enn.fit_resample(X_train, y_train)

# Undersampling methods
rus = RandomUnderSampler(random_state=42, replacement=False)
X_under, y_under = rus.fit_resample(X_train, y_train)

# Tomek Links: remove majority samples that are nearest neighbors to minority
tomek = TomekLinks(n_jobs=-1)
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)
```

### 7.3 Class Weights and Threshold Tuning

```python
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (precision_recall_curve, f1_score,
                              roc_auc_score, average_precision_score)

# Class weights: penalize misclassification of minority class
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"Class weights: {class_weight_dict}")

# Use with scikit-learn models
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Most sklearn classifiers support class_weight
lr = LogisticRegression(class_weight='balanced')
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)

# XGBoost: scale_pos_weight = count(negative) / count(positive)
import xgboost as xgb
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42)


def tune_classification_threshold(model, X_val, y_val, metric='f1'):
    """
    Find optimal decision threshold for imbalanced classification.
    Default threshold of 0.5 is rarely optimal for imbalanced data.
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.01, 1.0, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(y_val, y_pred, zero_division=0)
        scores.append(score)

    best_threshold = thresholds[np.argmax(scores)]
    best_score = max(scores)
    print(f"Best threshold: {best_threshold:.2f}, Best {metric}: {best_score:.4f}")

    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_val, y_proba)
    auprc = average_precision_score(y_val, y_proba)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, scores)
    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Best threshold: {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric} Score')
    plt.title(f'Threshold Tuning: {metric}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUPRC={auprc:.3f})')
    plt.tight_layout()
    plt.show()

    return best_threshold
```

### 7.4 Stratified Sampling

```python
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Stratified train/test split (preserves class proportions)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,           # Key: maintain class balance in each split
    random_state=42
)

# Verify proportions
print("Overall:", Counter(y))
print("Train:", Counter(y_train))
print("Test:", Counter(y_test))

# Stratified K-Fold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    # Train model, evaluate...
    print(f"Fold {fold+1}: Train={Counter(y_fold_train)}, Val={Counter(y_fold_val)}")
```

---

## 8. Statistical Analysis and Hypothesis Testing

### 8.1 Hypothesis Testing Framework

The hypothesis testing framework follows these steps:
1. State null (H₀) and alternative (H₁) hypotheses
2. Choose significance level α (typically 0.05)
3. Select appropriate test based on data characteristics
4. Compute test statistic and p-value
5. Make decision: reject H₀ if p < α

```python
from scipy import stats
import numpy as np

def hypothesis_test(sample1, sample2=None, test_type='auto', alpha=0.05):
    """
    Comprehensive hypothesis testing with automatic test selection.

    Test selection logic:
    - One sample: one-sample t-test or Wilcoxon signed-rank test
    - Two samples (normal): independent t-test
    - Two samples (non-normal): Mann-Whitney U test
    - Paired samples: paired t-test or Wilcoxon signed-rank
    - Multiple groups: ANOVA or Kruskal-Wallis
    """
    print(f"\nHypothesis Testing (α = {alpha})")
    print("=" * 50)

    if test_type == 'auto':
        # Check normality
        stat1, p_norm1 = stats.normaltest(sample1)
        normal1 = p_norm1 > alpha
        print(f"Sample 1 normality: {'Normal' if normal1 else 'Non-normal'} (p={p_norm1:.4f})")

        if sample2 is not None:
            stat2, p_norm2 = stats.normaltest(sample2)
            normal2 = p_norm2 > alpha
            print(f"Sample 2 normality: {'Normal' if normal2 else 'Non-normal'} (p={p_norm2:.4f})")

            # Two-sample tests
            if normal1 and normal2:
                # Levene's test for equal variances
                lev_stat, lev_p = stats.levene(sample1, sample2)
                equal_var = lev_p > alpha
                print(f"Equal variances: {'Yes' if equal_var else 'No'} (Levene p={lev_p:.4f})")

                t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
                print(f"\nTwo-sample t-test: t={t_stat:.4f}, p={p_value:.6f}")
                print("(Welch's t-test)" if not equal_var else "(Student's t-test)")
            else:
                u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
                print(f"\nMann-Whitney U test (non-parametric): U={u_stat:.4f}, p={p_value:.6f}")

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(sample1)**2 + np.std(sample2)**2) / 2)
            cohens_d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
            print(f"\nCohen's d (effect size): {cohens_d:.4f}")
            print("Effect: " + ('Negligible' if abs(cohens_d) < 0.2 else
                               'Small' if abs(cohens_d) < 0.5 else
                               'Medium' if abs(cohens_d) < 0.8 else 'Large'))

    # Decision
    print(f"\nDecision: {'Reject H₀' if p_value < alpha else 'Fail to reject H₀'}")
    print(f"Conclusion: {'Statistically significant difference' if p_value < alpha else 'No significant difference'}")

    return p_value
```

### 8.2 A/B Testing

```python
from scipy import stats
import numpy as np

class ABTest:
    """
    Complete A/B testing framework with statistical rigor.

    Tests for proportions (click rates, conversion rates) and means.
    Includes power analysis, sequential testing, and multiple testing correction.
    """

    def __init__(self, alpha=0.05, power=0.8, alternative='two-sided'):
        self.alpha = alpha
        self.power = power
        self.alternative = alternative

    def required_sample_size(self, baseline_rate, minimum_detectable_effect,
                             test_type='proportion'):
        """
        Calculate required sample size for adequate statistical power.

        For proportions: uses normal approximation
        For means: uses t-distribution approximation
        """
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower, TTestIndPower

        if test_type == 'proportion':
            p1 = baseline_rate
            p2 = baseline_rate + minimum_detectable_effect
            effect_size = proportion_effectsize(p1, p2)

            power_analysis = NormalIndPower()
            n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=self.power,
                alternative=self.alternative
            )
        else:  # means
            effect_size = minimum_detectable_effect  # In standard deviations
            power_analysis = TTestIndPower()
            n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=self.power,
                alternative=self.alternative
            )

        print(f"Required sample size per group: {int(np.ceil(n))}")
        print(f"Total required: {int(np.ceil(n)) * 2}")
        return int(np.ceil(n))

    def test_proportions(self, n_a, n_b, conversions_a, conversions_b):
        """
        Two-proportion z-test for A/B testing conversion rates.

        H₀: p_a = p_b (no difference)
        H₁: p_a ≠ p_b (two-sided) or p_b > p_a (one-sided)
        """
        p_a = conversions_a / n_a
        p_b = conversions_b / n_b
        p_pool = (conversions_a + conversions_b) / (n_a + n_b)

        # Z-statistic
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z_stat = (p_b - p_a) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-sided

        # Confidence interval for difference
        se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        z_ci = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = (p_b - p_a) - z_ci * se_diff
        ci_upper = (p_b - p_a) + z_ci * se_diff

        # Relative lift
        lift = (p_b - p_a) / p_a * 100

        print(f"\nA/B Test Results")
        print("=" * 50)
        print(f"Control (A): {conversions_a}/{n_a} = {p_a:.4f} ({p_a*100:.2f}%)")
        print(f"Treatment (B): {conversions_b}/{n_b} = {p_b:.4f} ({p_b*100:.2f}%)")
        print(f"Absolute difference: {(p_b - p_a)*100:.4f}%")
        print(f"Relative lift: {lift:.2f}%")
        print(f"95% CI for difference: [{ci_lower*100:.4f}%, {ci_upper*100:.4f}%]")
        print(f"Z-statistic: {z_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significant: {'Yes' if p_value < self.alpha else 'No'}")

        return {'p_value': p_value, 'z_stat': z_stat, 'lift': lift,
                'ci': (ci_lower, ci_upper), 'significant': p_value < self.alpha}

    def power_analysis_plot(self, baseline_rate, mde_range):
        """Plot sample size vs MDE to visualize the power-sample size tradeoff."""
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower

        power_analysis = NormalIndPower()
        sample_sizes = []
        for mde in mde_range:
            p1 = baseline_rate
            p2 = baseline_rate + mde
            effect_size = proportion_effectsize(p1, p2)
            n = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=self.power
            )
            sample_sizes.append(n)

        plt.figure(figsize=(10, 5))
        plt.plot(mde_range * 100, sample_sizes)
        plt.xlabel('Minimum Detectable Effect (%)')
        plt.ylabel('Required Sample Size per Group')
        plt.title(f'Power Analysis (α={self.alpha}, power={self.power})')
        plt.grid(True)
        plt.show()


# Multiple Testing Correction (Bonferroni and Benjamini-Hochberg)
def correct_multiple_tests(p_values, method='fdr_bh', alpha=0.05):
    """
    Correct for multiple comparisons to control false discovery rate.

    Methods:
    - 'bonferroni': Very conservative, controls FWER
    - 'fdr_bh': Benjamini-Hochberg, controls FDR (recommended)
    - 'fdr_by': Benjamini-Yekutieli, more conservative FDR
    """
    from statsmodels.stats.multitest import multipletests

    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method=method, is_sorted=False
    )

    print(f"\nMultiple Testing Correction ({method})")
    for i, (orig, corr, rej) in enumerate(zip(p_values, p_corrected, reject)):
        print(f"Test {i+1}: original p={orig:.4f}, corrected p={corr:.4f}, Reject H₀: {rej}")

    return p_corrected, reject
```

---

## 9. Data Visualization

### 9.1 Seaborn Advanced Visualizations

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication-quality style
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 100

def create_eda_dashboard(df, target=None):
    """
    Create a comprehensive EDA dashboard with multiple visualization types.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit for display
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:3]

    # Distribution dashboard
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        if target and target in df.columns and df[target].nunique() <= 5:
            # Distribution by target
            for cls in df[target].unique():
                subset = df[df[target] == cls][col].dropna()
                axes[i].hist(subset, alpha=0.5, label=str(cls), bins=30)
            axes[i].legend()
        else:
            axes[i].hist(df[col].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='white')
            axes[i].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[i].axvline(df[col].median(), color='orange', linestyle='--', label='Median')
            axes[i].legend()
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    # Hide empty subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Box plots by category
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        fig, axes = plt.subplots(1, min(3, len(categorical_cols)), figsize=(15, 5))
        if len(categorical_cols) == 1:
            axes = [axes]

        for i, cat_col in enumerate(categorical_cols[:3]):
            if len(numeric_cols) > 0:
                num_col = numeric_cols[0] if target is None else (
                    target if target in df.select_dtypes(include=[np.number]).columns
                    else numeric_cols[0]
                )
                df.boxplot(column=num_col, by=cat_col, ax=axes[i])
                axes[i].set_title(f'{num_col} by {cat_col}')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()


# Plotly interactive visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def interactive_eda(df, target_col=None):
    """Create interactive Plotly visualizations for EDA."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Interactive scatter matrix
    if target_col:
        fig = px.scatter_matrix(
            df,
            dimensions=numeric_cols[:5],
            color=target_col,
            title="Interactive Scatter Matrix",
            opacity=0.5
        )
    else:
        fig = px.scatter_matrix(df, dimensions=numeric_cols[:5])
    fig.show()

    # Interactive correlation heatmap
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=corr.round(2).values,
        texttemplate='%{text}',
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(title='Interactive Correlation Heatmap', height=600)
    fig.show()

    # Distribution comparison
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=numeric_cols[:6]
    )
    for i, col in enumerate(numeric_cols[:6]):
        row = i // 3 + 1
        col_idx = i % 3 + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=row, col=col_idx
        )
    fig.update_layout(title='Feature Distributions', height=600)
    fig.show()


# Plotly Dash Dashboard
def create_plotly_dash_app(df):
    """
    Create a Dash web application for interactive EDA.
    Run with: python app.py and open http://127.0.0.1:8050
    """
    import dash
    from dash import dcc, html, Input, Output
    import plotly.express as px

    app = dash.Dash(__name__)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    app.layout = html.Div([
        html.H1('EDA Dashboard', style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label('X-Axis'),
                dcc.Dropdown(id='x-col', options=[{'label': c, 'value': c} for c in all_cols],
                             value=numeric_cols[0] if numeric_cols else all_cols[0])
            ], style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.Label('Y-Axis'),
                dcc.Dropdown(id='y-col', options=[{'label': c, 'value': c} for c in numeric_cols],
                             value=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
            ], style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.Label('Color'),
                dcc.Dropdown(id='color-col', options=[{'label': 'None', 'value': 'None'}] +
                             [{'label': c, 'value': c} for c in all_cols], value='None')
            ], style={'width': '30%', 'display': 'inline-block'}),
        ]),

        dcc.Graph(id='scatter-plot'),
        dcc.Graph(id='distribution-plot')
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('x-col', 'value'), Input('y-col', 'value'), Input('color-col', 'value')]
    )
    def update_scatter(x_col, y_col, color_col):
        color = None if color_col == 'None' else color_col
        return px.scatter(df, x=x_col, y=y_col, color=color,
                          title=f'{x_col} vs {y_col}', opacity=0.6)

    @app.callback(
        Output('distribution-plot', 'figure'),
        [Input('x-col', 'value'), Input('color-col', 'value')]
    )
    def update_dist(x_col, color_col):
        color = None if color_col == 'None' else color_col
        if df[x_col].dtype in ['int64', 'float64']:
            return px.histogram(df, x=x_col, color=color, marginal='box')
        else:
            return px.bar(df[x_col].value_counts().reset_index(),
                          x='index', y=x_col, title=f'Distribution of {x_col}')

    return app
```

---

## 10. SQL for Data Science

### 10.1 Window Functions

```sql
-- Window functions are among the most powerful SQL features for data analysis

-- ROW_NUMBER, RANK, DENSE_RANK
SELECT
    user_id,
    product_id,
    amount,
    order_date,
    -- Row number within user's orders (by date)
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS order_sequence,
    -- Rank (with gaps for ties)
    RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS amount_rank,
    -- Dense rank (no gaps)
    DENSE_RANK() OVER (ORDER BY amount DESC) AS global_dense_rank,
    -- Percentile rank
    PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY amount) AS percentile_rank,
    -- NTILE: divide into N equal groups (quartiles)
    NTILE(4) OVER (ORDER BY amount) AS quartile
FROM orders;

-- LAG and LEAD: access previous/next rows
SELECT
    date,
    daily_revenue,
    LAG(daily_revenue, 1) OVER (ORDER BY date) AS prev_day_revenue,
    LEAD(daily_revenue, 1) OVER (ORDER BY date) AS next_day_revenue,
    daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY date) AS day_over_day_change,
    (daily_revenue - LAG(daily_revenue, 7) OVER (ORDER BY date)) /
        LAG(daily_revenue, 7) OVER (ORDER BY date) * 100 AS wow_change_pct
FROM daily_metrics;

-- Running totals and moving averages
SELECT
    date,
    revenue,
    -- Running total
    SUM(revenue) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) AS cumulative_revenue,
    -- 7-day moving average
    AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d,
    -- 30-day moving average
    AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS ma_30d,
    -- 7-day moving sum
    SUM(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_sum
FROM daily_sales;

-- FIRST_VALUE, LAST_VALUE, NTH_VALUE
SELECT
    user_id,
    order_date,
    amount,
    FIRST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY order_date) AS first_order_amount,
    LAST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_order_amount,
    NTH_VALUE(amount, 2) OVER (PARTITION BY user_id ORDER BY order_date) AS second_order_amount
FROM orders;
```

### 10.2 CTEs (Common Table Expressions)

```sql
-- Basic CTE
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue,
        COUNT(DISTINCT user_id) AS unique_customers,
        COUNT(*) AS total_orders
    FROM orders
    WHERE order_date >= '2024-01-01'
    GROUP BY 1
),
monthly_growth AS (
    SELECT
        month,
        revenue,
        unique_customers,
        LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
        (revenue - LAG(revenue) OVER (ORDER BY month)) /
            LAG(revenue) OVER (ORDER BY month) * 100 AS growth_pct
    FROM monthly_revenue
)
SELECT * FROM monthly_growth ORDER BY month;

-- Recursive CTE: for hierarchical data / organizational charts
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level employees
    SELECT
        employee_id,
        name,
        manager_id,
        0 AS level,
        CAST(name AS VARCHAR) AS hierarchy_path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case
    SELECT
        e.employee_id,
        e.name,
        e.manager_id,
        eh.level + 1,
        CAST(eh.hierarchy_path || ' > ' || e.name AS VARCHAR)
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy ORDER BY level, name;

-- Complex analytical query with multiple CTEs
WITH user_segments AS (
    -- RFM Segmentation
    SELECT
        user_id,
        MAX(order_date) AS last_order_date,
        COUNT(*) AS frequency,
        SUM(amount) AS monetary,
        DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days
    FROM orders
    GROUP BY user_id
),
rfm_scores AS (
    SELECT
        user_id,
        NTILE(5) OVER (ORDER BY recency_days ASC) AS r_score,    -- Lower recency = higher score
        NTILE(5) OVER (ORDER BY frequency DESC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary DESC) AS m_score
    FROM user_segments
),
customer_segments AS (
    SELECT
        user_id,
        r_score, f_score, m_score,
        r_score + f_score + m_score AS rfm_total,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 THEN 'Recent Customers'
            WHEN f_score >= 4 AND m_score >= 4 THEN 'Big Spenders'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
            ELSE 'Others'
        END AS segment
    FROM rfm_scores
)
SELECT
    segment,
    COUNT(*) AS customer_count,
    AVG(rfm_total) AS avg_rfm_score,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS segment_pct
FROM customer_segments
GROUP BY segment
ORDER BY avg_rfm_score DESC;
```

### 10.3 Advanced Aggregations

```sql
-- GROUPING SETS, ROLLUP, CUBE
SELECT
    region,
    product_category,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY GROUPING SETS (
    (region, product_category),   -- Subtotals by region + category
    (region),                      -- Subtotals by region only
    (product_category),            -- Subtotals by category only
    ()                             -- Grand total
);

-- ROLLUP (hierarchical aggregation)
SELECT
    year,
    quarter,
    month,
    SUM(revenue) AS total_revenue
FROM sales
GROUP BY ROLLUP(year, quarter, month);  -- year > quarter > month > grand total

-- Conditional aggregation (CASE inside aggregate)
SELECT
    product_category,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN order_status = 'completed' THEN 1 ELSE 0 END) AS completed_orders,
    SUM(CASE WHEN order_status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    AVG(CASE WHEN user_type = 'premium' THEN amount END) AS avg_premium_order,
    ROUND(
        100.0 * SUM(CASE WHEN order_status = 'cancelled' THEN 1 ELSE 0 END) /
        COUNT(*), 2
    ) AS cancellation_rate
FROM orders
GROUP BY product_category;
```

---

## 11. Data Pipelines: Pandas, Dask, Polars

### 11.1 Efficient Pandas Pipelines

```python
import pandas as pd
import numpy as np
from functools import reduce

# Method chaining pattern for clean, readable pipelines
def build_features(df):
    return (df
        .assign(
            log_price=lambda x: np.log1p(x['price']),
            price_sq=lambda x: x['price'] ** 2,
            is_high_value=lambda x: (x['price'] > x['price'].quantile(0.75)).astype(int),
            age_group=lambda x: pd.cut(x['age'], bins=[0, 25, 40, 60, 100],
                                        labels=['Young', 'Adult', 'Middle', 'Senior'])
        )
        .dropna(subset=['price', 'age'])
        .query('price > 0 and age > 0')
        .reset_index(drop=True)
    )


# Sklearn-compatible pipeline with custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns."""

    def __init__(self, date_col='date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            dt = pd.to_datetime(X[self.date_col])
            X['year'] = dt.dt.year
            X['month'] = dt.dt.month
            X['dayofweek'] = dt.dt.dayofweek
            X['quarter'] = dt.dt.quarter
            X['is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            X.drop(columns=[self.date_col], inplace=True)
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers at specified percentiles."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.bounds = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=[np.number]).columns:
            self.bounds[col] = (X[col].quantile(self.lower), X[col].quantile(self.upper))
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X
```

### 11.2 Dask for Big Data

```python
# pip install dask[dataframe]
import dask.dataframe as dd
import dask
from dask.distributed import Client

# Start a local Dask cluster
client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
print(client)

# Read large datasets in parallel
df = dd.read_csv('large_dataset_*.csv')  # Reads multiple files in parallel
df = dd.read_parquet('data/') # Preferred for analytics workloads

# Dask operations are lazy (built into a task graph)
# They look identical to pandas:
result = (df
    .groupby('category')['sales']
    .agg(['sum', 'mean', 'count'])
    .reset_index()
)

# Trigger computation
result_computed = result.compute()  # Returns pandas DataFrame

# Dask for large-scale groupby
monthly = (df
    .assign(month=df['date'].dt.to_period('M'))
    .groupby('month')['revenue']
    .sum()
    .compute()
)

# Map partitions for custom functions
def process_partition(partition):
    """Apply custom logic to each partition."""
    partition['feature'] = partition['value'] ** 2
    return partition

df_processed = df.map_partitions(process_partition)

# Dask pipeline for ML preprocessing
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.model_selection import train_test_split as dask_train_test_split

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = dask_train_test_split(X, y, test_size=0.2, shuffle=True)

scaler = DaskStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

client.close()
```

### 11.3 Polars (Ultra-Fast DataFrame Library)

```python
# pip install polars
import polars as pl
import numpy as np

# Read data (polars is much faster than pandas for large files)
df = pl.read_csv('large_dataset.csv')
df = pl.read_parquet('data.parquet')

# Polars uses lazy evaluation for optimization
q = (
    pl.scan_csv('large_dataset.csv')  # Lazy scanning
    .filter(pl.col('amount') > 0)
    .with_columns([
        pl.col('amount').log1p().alias('log_amount'),
        (pl.col('price') * pl.col('quantity')).alias('revenue'),
        pl.col('date').str.strptime(pl.Date, '%Y-%m-%d').alias('parsed_date')
    ])
    .groupby('category')
    .agg([
        pl.col('revenue').sum().alias('total_revenue'),
        pl.col('amount').mean().alias('avg_amount'),
        pl.col('user_id').n_unique().alias('unique_users'),
        pl.col('amount').std().alias('std_amount')
    ])
    .sort('total_revenue', descending=True)
)

# Execute the lazy query
result = q.collect()
print(result)

# Performance comparison: Polars vs Pandas
import time

# Polars: ~10x faster than pandas for large datasets
start = time.time()
polars_result = (
    pl.scan_csv('big_data.csv')
    .groupby('category')
    .agg(pl.col('value').sum())
    .collect()
)
print(f"Polars: {time.time() - start:.2f}s")

# Window functions in Polars
df_with_windows = df.with_columns([
    pl.col('sales').rolling_mean(window_size=7).over('store_id').alias('rolling_7d_avg'),
    pl.col('sales').cumsum().over('store_id').alias('cumulative_sales'),
    pl.col('sales').rank().over('category').alias('rank_in_category')
])
```

---

## 12. Experiment Tracking Basics

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import json

# Start MLflow server: mlflow server --host 0.0.0.0 --port 5000

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("eda_experiments")

def run_experiment(model_class, params, X_train, y_train, X_test, y_test, tags=None):
    """
    Run a model training experiment with full MLflow tracking.
    """
    with mlflow.start_run():
        # Log tags for organization
        if tags:
            mlflow.set_tags(tags)

        # Log parameters
        mlflow.log_params(params)

        # Log dataset info
        mlflow.log_param('n_train_samples', len(X_train))
        mlflow.log_param('n_features', X_train.shape[1])
        mlflow.log_param('feature_names', list(X_train.columns))

        # Train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                      precision_score, recall_score)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
        }
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model",
                                  registered_model_name="production_model")

        # Log feature importance as artifact
        if hasattr(model, 'feature_importances_'):
            fi_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            fi_df.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

        # Log confusion matrix as figure
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, 'confusion_matrix.png')
        plt.close()

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Metrics: {metrics}")

        return model, metrics
```

---

## 13. Data Versioning with DVC

```bash
# Install DVC
pip install dvc dvc-s3  # or dvc-gcs, dvc-azure

# Initialize in git repository
git init
dvc init
git add .dvc/
git commit -m "Initialize DVC"

# Configure remote storage (S3 example)
dvc remote add -d myremote s3://mybucket/dvc-store
git add .dvc/config
git commit -m "Configure DVC remote"

# Track data files
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc data/raw/.gitignore
git commit -m "Track training data with DVC"

# Push data to remote
dvc push

# Pull data on another machine
git clone <repo_url>
dvc pull

# Create a DVC pipeline
# dvc.yaml
```

```yaml
# dvc.yaml - Define the ML pipeline
stages:
  prepare:
    cmd: python src/prepare_data.py --input data/raw/data.csv --output data/processed/
    deps:
      - src/prepare_data.py
      - data/raw/data.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  featurize:
    cmd: python src/featurize.py --input data/processed/ --output data/features/
    deps:
      - src/featurize.py
      - data/processed/train.csv
      - data/processed/test.csv
    outs:
      - data/features/train_features.csv
      - data/features/test_features.csv
    params:
      - params.yaml:
        - featurize.n_features
        - featurize.seed

  train:
    cmd: python src/train.py --features data/features/ --model models/
    deps:
      - src/train.py
      - data/features/train_features.csv
    outs:
      - models/model.pkl
    params:
      - params.yaml:
        - train.learning_rate
        - train.n_estimators
        - train.max_depth
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py --model models/ --features data/features/ --metrics metrics/
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/features/test_features.csv
    metrics:
      - metrics/test_metrics.json:
          cache: false
    plots:
      - metrics/confusion_matrix.csv
```

```bash
# Run the pipeline
dvc repro

# Check pipeline status
dvc status

# Compare experiments
dvc params diff
dvc metrics diff

# DVC experiments (like MLflow for DVC)
dvc exp run --set-param train.learning_rate=0.1
dvc exp run --set-param train.learning_rate=0.05
dvc exp show  # Compare experiments
dvc exp apply <exp-id>  # Apply best experiment
```

---

## 14. Data Lineage and Provenance

**Data lineage** tracks where data comes from, how it is transformed, and where it flows. Critical for reproducibility, debugging, and compliance.

### 14.1 Why Data Lineage Matters

- **Reproducibility**: Trace a model prediction back to the exact raw data and transformations
- **Debugging**: When outputs are wrong, trace upstream to find which transformation introduced the error
- **Compliance**: GDPR, HIPAA — know which data sources feed personal/health data
- **Impact analysis**: Before changing a pipeline, see which models and reports depend on it

### 14.2 Lineage Concepts

| Concept | Description |
|--------|-------------|
| **Column lineage** | Which source columns feed a derived column |
| **Table lineage** | Which tables are joined/aggregated to produce another |
| **Job lineage** | Which jobs (scripts, DAGs) produce which assets |
| **End-to-end lineage** | Raw data → features → model → prediction |

### 14.3 Tools and Practices

```python
# Manual lineage via metadata (example pattern)
import json
from datetime import datetime

def record_lineage(output_path, inputs, transforms, version="1.0"):
    """Record lineage metadata alongside output artifact."""
    lineage = {
        "version": version,
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": [{"path": p, "checksum": compute_checksum(p)} for p in inputs],
        "transforms": transforms,  # List of (function_name, params)
        "output": output_path,
    }
    with open(output_path + ".lineage.json", "w") as f:
        json.dump(lineage, f, indent=2)
```

**Lineage tools**: **OpenLineage** (open standard, integrates with Airflow, Spark, dbt), **Apache Atlas**, **DataHub**, **Great Expectations** (data quality + lineage). In ML: **MLflow** (experiments + artifacts), **DVC** (data versioning), **Feast** (feature lineage).

### 14.4 Feature Stores and Lineage

Feast and similar feature stores record which sources and transforms produce each feature. Use `feature_view` definitions and materialization logs for lineage.

---

## 15. Feature Stores

### 15.1 Feast (Open-Source Feature Store)

```python
# pip install feast
# feast init my_feature_store
# cd my_feature_store

# feature_store.yaml
"""
project: my_feature_store
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
offline_store:
    type: file
"""

# features.py - Feature definitions
from datetime import timedelta
from feast import (
    Entity, Feature, FeatureView, FileSource, ValueType,
    FeatureService
)

# Define entity (the "key" for features)
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

# Define data source
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created"
)

# Define feature view (a group of related features)
user_stats_fv = FeatureView(
    name="user_statistics",
    entities=["user_id"],
    ttl=timedelta(days=30),  # How long features remain valid
    features=[
        Feature(name="total_purchases", dtype=ValueType.INT64),
        Feature(name="avg_order_value", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_order", dtype=ValueType.INT32),
        Feature(name="total_lifetime_value", dtype=ValueType.FLOAT),
        Feature(name="preferred_category", dtype=ValueType.STRING),
    ],
    online=True,  # Serve in real-time
    source=user_stats_source,
)

# Feature service (group features for a specific ML use case)
recommendation_fs = FeatureService(
    name="recommendation_service",
    features=[user_stats_fv],
    description="Features for product recommendation model"
)
```

```python
# Using Feast in training and serving

from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# Offline retrieval (for training)
training_df = store.get_historical_features(
    entity_df=pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "event_timestamp": pd.to_datetime([
            "2024-01-15", "2024-01-15", "2024-02-01",
            "2024-02-01", "2024-03-01"
        ])
    }),
    features=[
        "user_statistics:total_purchases",
        "user_statistics:avg_order_value",
        "user_statistics:days_since_last_order"
    ]
).to_df()

# Materialize features to online store
store.materialize_incremental(end_date=pd.Timestamp.now())

# Online retrieval (for real-time inference)
feature_vector = store.get_online_features(
    features=[
        "user_statistics:total_purchases",
        "user_statistics:avg_order_value",
    ],
    entity_rows=[{"user_id": 1001}, {"user_id": 1002}]
).to_df()

print(feature_vector)
```

---

## 16. Full EDA Example: Titanic Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

print("=" * 60)
print("TITANIC DATASET - COMPLETE EDA")
print("=" * 60)

# 1. Basic Overview
print(f"\nShape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive stats:\n{df.describe()}")

# 2. Missing Value Analysis
missing = df.isnull().sum()
missing_pct = df.isnull().mean() * 100
print(f"\nMissing Values:")
print(pd.DataFrame({'Count': missing, 'Percent': missing_pct})[missing > 0])

# Age: 19.9% missing - use median by Pclass + Sex
df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)
# Embarked: 0.2% missing - use mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Cabin: 77% missing - create binary indicator
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
df.drop(columns=['Cabin'], inplace=True)

# 3. Target Variable Analysis
print(f"\nSurvival Rate: {df['Survived'].mean():.3f}")
print(f"Survivors: {df['Survived'].sum()} / {len(df)}")

# 4. Feature Engineering
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                   'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                       labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# 5. Survival Analysis by Feature
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Titanic: Survival Analysis by Feature', fontsize=18, fontweight='bold')

# By Sex
survival_by_sex = df.groupby('Sex')['Survived'].mean()
axes[0, 0].bar(survival_by_sex.index, survival_by_sex.values, color=['salmon', 'steelblue'])
axes[0, 0].set_title('Survival Rate by Sex')
axes[0, 0].set_ylabel('Survival Rate')
for i, v in enumerate(survival_by_sex.values):
    axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center')

# By Pclass
survival_by_class = df.groupby('Pclass')['Survived'].mean()
axes[0, 1].bar(survival_by_class.index, survival_by_class.values,
               color=['gold', 'silver', '#CD7F32'])
axes[0, 1].set_title('Survival Rate by Passenger Class')
axes[0, 1].set_ylabel('Survival Rate')

# Age distribution by survival
df[df['Survived'] == 1]['Age'].hist(ax=axes[0, 2], bins=30, alpha=0.5,
                                     label='Survived', color='green')
df[df['Survived'] == 0]['Age'].hist(ax=axes[0, 2], bins=30, alpha=0.5,
                                     label='Not Survived', color='red')
axes[0, 2].set_title('Age Distribution by Survival')
axes[0, 2].legend()

# By Embarked
survival_by_embarked = df.groupby('Embarked')['Survived'].mean()
axes[1, 0].bar(survival_by_embarked.index, survival_by_embarked.values)
axes[1, 0].set_title('Survival Rate by Embarked Port')

# By Family Size
survival_by_family = df.groupby('FamilySize')['Survived'].mean()
axes[1, 1].plot(survival_by_family.index, survival_by_family.values, 'o-', color='steelblue')
axes[1, 1].set_title('Survival Rate by Family Size')
axes[1, 1].set_xlabel('Family Size')

# Fare distribution
axes[1, 2].hist(df['Fare'].clip(upper=200), bins=50, color='steelblue', alpha=0.7)
axes[1, 2].set_title('Fare Distribution (capped at 200)')

# Title analysis
title_survival = df.groupby('Title')['Survived'].mean().sort_values(ascending=False)
axes[2, 0].barh(title_survival.index, title_survival.values)
axes[2, 0].set_title('Survival Rate by Title')

# Heatmap: Pclass x Sex
pivot = df.pivot_table(values='Survived', index='Pclass', columns='Sex', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[2, 1])
axes[2, 1].set_title('Survival Rate: Pclass vs Sex')

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', ax=axes[2, 2], center=0)
axes[2, 2].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()

# 6. Statistical Tests
print("\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("=" * 60)

# Chi-squared test: Sex vs Survived
contingency = pd.crosstab(df['Sex'], df['Survived'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nSex vs Survived: chi2={chi2:.2f}, p={p:.6f}, significant={'Yes' if p < 0.05 else 'No'}")

# T-test: Age difference between survivors and non-survivors
survived_age = df[df['Survived'] == 1]['Age'].dropna()
not_survived_age = df[df['Survived'] == 0]['Age'].dropna()
t_stat, p_value = stats.ttest_ind(survived_age, not_survived_age)
print(f"Age (Survived vs Not): t={t_stat:.3f}, p={p_value:.6f}")
print(f"Mean age (survived): {survived_age.mean():.1f}, Mean age (not survived): {not_survived_age.mean():.1f}")

print(f"\nSurvival rates:")
print(f"By Sex: {df.groupby('Sex')['Survived'].mean().to_dict()}")
print(f"By Class: {df.groupby('Pclass')['Survived'].mean().to_dict()}")
print(f"By Title: {df.groupby('Title')['Survived'].mean().to_dict()}")
print(f"By FareBin: {df.groupby('FareBin')['Survived'].mean().to_dict()}")
```

---

## 17. Full EDA Example: Housing Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Dataset shape: {df.shape}")
print(f"Features: {housing.feature_names}")
print(f"Target: MedHouseVal (Median house value in $100,000)")
print(f"\nDescriptive Stats:\n{df.describe()}")

# Rename for clarity
df.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
              'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']

target = 'MedHouseVal'

# 1. Target Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df[target].hist(ax=axes[0], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('House Value Distribution')

df[target].plot.box(ax=axes[1])
axes[1].set_title('House Value Box Plot')

stats.probplot(df[target], dist='norm', plot=axes[2])
axes[2].set_title('QQ Plot: House Value')
plt.tight_layout()
plt.show()

print(f"\nSkewness: {df[target].skew():.3f}")
print(f"Kurtosis: {df[target].kurtosis():.3f}")
# Target is right-skewed -> consider log transformation
df['log_target'] = np.log1p(df[target])

# 2. Feature Correlations with Target
corr_with_target = df.corr()[target].sort_values(ascending=False)
print(f"\nCorrelations with {target}:")
print(corr_with_target)

plt.figure(figsize=(10, 6))
corr_with_target.drop(target).plot(kind='barh',
                                    color=['green' if v > 0 else 'red'
                                           for v in corr_with_target.drop(target)])
plt.axvline(x=0, color='black', linestyle='-')
plt.title(f'Feature Correlations with {target}')
plt.tight_layout()
plt.show()

# 3. Geographic Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter = axes[0].scatter(df['Longitude'], df['Latitude'],
                           c=df[target], cmap='viridis',
                           alpha=0.4, s=5)
plt.colorbar(scatter, ax=axes[0])
axes[0].set_title('House Values by Location')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

scatter2 = axes[1].scatter(df['Longitude'], df['Latitude'],
                            c=df['MedInc'], cmap='plasma',
                            alpha=0.4, s=5)
plt.colorbar(scatter2, ax=axes[1])
axes[1].set_title('Median Income by Location')
plt.tight_layout()
plt.show()

# 4. Feature Engineering
df['rooms_per_person'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
df['income_per_room'] = df['MedInc'] / df['AveRooms']

# Log-transform skewed features
skewed_features = ['MedInc', 'Population', 'AveOccup']
for feature in skewed_features:
    df[f'log_{feature}'] = np.log1p(df[feature])
    print(f"{feature} skewness: {df[feature].skew():.3f} -> log: {df[f'log_{feature}'].skew():.3f}")

# 5. Bivariate Analysis: Features vs Target
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']

for i, col in enumerate(feature_cols):
    row, col_idx = i // 4, i % 4
    axes[row, col_idx].scatter(df[col], df[target], alpha=0.1, s=2, color='steelblue')

    # Add regression line
    m, b = np.polyfit(df[col], df[target], 1)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    axes[row, col_idx].plot(x_line, m * x_line + b, 'r-', linewidth=2)

    r, p = stats.pearsonr(df[col], df[target])
    axes[row, col_idx].set_title(f'{col}\nr={r:.3f}')
    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel(target)

plt.suptitle('Feature vs Target Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 6. Outlier Analysis
print("\nOutlier Analysis:")
for col in ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# 7. Full Correlation Matrix
full_corr = df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(full_corr, dtype=bool))
sns.heatmap(full_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
plt.title('Full Correlation Matrix (Housing Dataset)', fontsize=16)
plt.tight_layout()
plt.show()

print("\nKey Insights:")
print("1. MedInc has strongest correlation with house value (r=0.69)")
print("2. Latitude/Longitude capture geographic price patterns")
print("3. AveOccup shows weak negative correlation (crowded areas = lower prices)")
print("4. House age has minimal impact on price")
print("5. Geographic visualization reveals coastal premium (Bay Area, LA, San Diego)")
```

---

## Common Pitfalls in Data Science

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| **Train/test leakage** | Inflated metrics, poor production performance | Split before any transformation; use `sklearn.pipeline.Pipeline` |
| **Target leakage in features** | Unrealistic accuracy; features unavailable at inference | Exclude future/data-collection-time info (e.g., "days_since_click" at predict time) |
| **Ignoring missingness mechanism** | Biased imputation | Test MCAR (Little's test); use MICE for MAR |
| **Over-reliance on correlation** | Miss non-linear relationships | Use mutual information, scatter plots, partial dependence |
| **Data snooping / multiple testing** | False discoveries | Use Bonferroni or FDR correction; pre-register hypotheses |
| **Imbalanced data, accuracy only** | 99% accuracy on 99:1 class = useless | Use precision, recall, F1, PR-AUC; stratify splits |
| **No baseline model** | Can't tell if ML adds value | Start with mean, median, or simple rule |
| **Single random split** | High variance in reported performance | Use k-fold CV; report mean ± std |
| **Feature engineering after split** | Leakage from test into feature design | Do EDA on train only; lock features before test evaluation |
| **Unversioned data/code** | Irreproducible results | Use DVC, Git LFS, or similar for data and code |

---

## Quick Reference: Data Science Toolkit

| Task | Tool/Method | When to Use |
|------|-------------|-------------|
| Missing values (<5%) | Mean/Median imputation | MCAR, numeric features |
| Missing values (5-20%) | KNN imputation | MAR, correlated features |
| Missing values (>20%) | MICE/Iterative imputation | MAR, complex patterns |
| Outliers (mild) | IQR capping (Winsorize) | Mild outliers, keep data |
| Outliers (multivariate) | Isolation Forest | Complex multivariate data |
| Imbalanced (<10:1) | Class weights | Fast, no data modification |
| Imbalanced (>10:1) | SMOTE + class weights | Severe imbalance |
| Feature selection | Boruta + RFECV | All-relevant + optimal set |
| Large data (>10GB) | Dask or Polars | Memory constraints |
| Categorical encoding | Target encoding + CV | High-cardinality categories |
| A/B testing | Z-test proportions | Conversion rate comparison |
| Distribution comparison | Mann-Whitney U | Non-normal distributions |
| Multiple testing | Benjamini-Hochberg FDR | Testing many hypotheses |

---

*This guide covers the complete data science workflow from EDA to production-ready feature engineering. For modeling, see the companion ML guides.*

---

## References

| Topic | Resource |
|-------|----------|
| CRISP-DM | Chapman et al., CRISP-DM 1.0 Step-by-step guide (2000) |
| Data Quality | Little & Rubin, Statistical Analysis with Missing Data (2019) |
| Feature Engineering | Kuhn & Johnson, Feature Engineering and Selection (2019) |
| Imbalanced Learning | He & Ma, Imbalanced Learning (2013) |
| A/B Testing | Kohavi et al., Trustworthy Online Controlled Experiments (2020) |
| Data Lineage | OpenLineage: openlineage.io; DataHub: datahubproject.io |
| Pandas/Dask | McKinney, Python for Data Analysis (2022); Dask docs: dask.org |
