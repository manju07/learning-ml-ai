# Time Series Forecasting: Complete Guide

## Table of Contents
1. [Introduction to Time Series](#introduction)
2. [Time Series Components](#components)
3. [Preprocessing](#preprocessing)
4. [Traditional Methods](#traditional)
5. [Machine Learning Methods](#ml-methods)
6. [Deep Learning Methods](#deep-learning)
7. [LSTM for Time Series](#lstm)
8. [Transformer for Time Series](#transformer)
9. [Evaluation Metrics](#metrics)
10. [Practical Examples](#examples)

---

## Introduction to Time Series {#introduction}

Time series forecasting involves predicting future values based on historical data points collected over time.

### Key Characteristics
- **Temporal Dependencies**: Values depend on previous values
- **Trend**: Long-term increase or decrease
- **Seasonality**: Regular patterns repeating over time
- **Cyclical**: Irregular patterns
- **Noise**: Random variations

### Applications
- Stock price prediction
- Weather forecasting
- Sales forecasting
- Energy demand prediction
- Traffic prediction
- Anomaly detection

---

## Time Series Components {#components}

### Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pyplot as plt

# Load time series data
df = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')

# Decompose
decomposition = seasonal_decompose(df['value'], model='additive', period=12)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

---

## Preprocessing {#preprocessing}

### Stationarity

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    """Check if time series is stationary"""
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Stationary")
    else:
        print("Non-stationary")

# Make stationary: Differencing
df['diff'] = df['value'].diff()
df['diff'].dropna().plot()
check_stationarity(df['diff'].dropna())
```

### Normalization

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['value']])

# Standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['value']])
```

### Creating Sequences

```python
def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Example
seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

---

## Traditional Methods {#traditional}

### ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(df['value'], order=(1, 1, 1))
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=10)
print(f"Forecast: {forecast}")

# Auto ARIMA
from pmdarima import auto_arima
model = auto_arima(df['value'], seasonal=True, m=12)
forecast = model.predict(n_periods=10)
```

### Exponential Smoothing

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple Exponential Smoothing
model = ExponentialSmoothing(df['value'], trend='add', seasonal='add', seasonal_periods=12)
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=10)
```

### Prophet

```python
from prophet import Prophet

# Prepare data
prophet_df = df.reset_index()
prophet_df.columns = ['ds', 'y']

# Fit model
model = Prophet()
model.fit(prophet_df)

# Forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
model.plot(forecast)
```

---

## Machine Learning Methods {#ml-methods}

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

# Create features
def create_features(df):
    df['lag1'] = df['value'].shift(1)
    df['lag2'] = df['value'].shift(2)
    df['rolling_mean'] = df['value'].rolling(window=7).mean()
    return df.dropna()

df_features = create_features(df)
X = df_features[['lag1', 'lag2', 'rolling_mean']]
y = df_features['value']

# Train
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X, y)
predictions = model.predict(X)
```

---

## Deep Learning Methods {#deep-learning}

### Simple RNN

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.SimpleRNN(50, activation='relu', input_shape=(seq_length, 1)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, validation_split=0.2)
```

---

## LSTM for Time Series {#lstm}

### LSTM Model

```python
def build_lstm_model(seq_length, n_features):
    model = keras.Sequential([
        layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, n_features)),
        layers.LSTM(50, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_model(seq_length=60, n_features=1)
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ModelCheckpoint('best_lstm.h5', save_best_only=True)
    ]
)
```

### Bidirectional LSTM

```python
model = keras.Sequential([
    layers.Bidirectional(layers.LSTM(50, return_sequences=True), input_shape=(seq_length, n_features)),
    layers.Bidirectional(layers.LSTM(50)),
    layers.Dense(1)
])
```

### Multi-step Forecasting

```python
def create_multi_step_sequences(data, seq_length, forecast_horizon):
    """Create sequences for multi-step forecasting"""
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_multi_step_sequences(scaled_data, seq_length=60, forecast_horizon=10)

# Model for multi-step
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
    layers.LSTM(50, return_sequences=True),
    layers.LSTM(50),
    layers.Dense(10)  # Predict 10 steps ahead
])
```

---

## Transformer for Time Series {#transformer}

### Time Series Transformer

```python
from tensorflow.keras import layers, Model

class TimeSeriesTransformer(Model):
    def __init__(self, d_model=64, num_heads=4, dff=128, num_layers=2, seq_length=60):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        
        self.embedding = layers.Dense(d_model)
        self.pos_encoding = self.positional_encoding(seq_length, d_model)
        
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff)
            for _ in range(num_layers)
        ]
        
        self.final_layer = layers.Dense(1)
    
    def positional_encoding(self, position, d_model):
        angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training)
        
        x = self.final_layer(x[:, -1, :])
        return x

model = TimeSeriesTransformer()
model.compile(optimizer='adam', loss='mse')
```

---

## Evaluation Metrics {#metrics}

### Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_forecast(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

metrics = evaluate_forecast(y_test, predictions)
print(metrics)
```

---

## Practical Examples {#examples}

### Example: Stock Price Prediction

```python
import yfinance as yf
import pandas as pd

# Download stock data
ticker = yf.Ticker("AAPL")
data = ticker.history(period="5y")

# Prepare data
df = data[['Close']].reset_index()
df.columns = ['ds', 'y']

# Split train/test
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Scale
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['y']])
test_scaled = scaler.transform(test_data[['y']])

# Create sequences
X_train, y_train = create_sequences(train_scaled, seq_length=60)
X_test, y_test = create_sequences(test_scaled, seq_length=60)

# Build and train LSTM
model = build_lstm_model(seq_length=60, n_features=1)
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate
metrics = evaluate_forecast(y_test_actual, predictions)
print(metrics)
```

---

## Best Practices

1. **Check Stationarity**: Use differencing if needed
2. **Handle Seasonality**: Account for seasonal patterns
3. **Feature Engineering**: Create lag features, rolling statistics
4. **Cross-Validation**: Use time series cross-validation
5. **Multiple Models**: Ensemble different approaches
6. **Monitor Performance**: Track metrics over time
7. **Update Models**: Retrain periodically

---

## Resources

- **Statsmodels**: Statistical models
- **Prophet**: Facebook's forecasting tool
- **pmdarima**: Auto ARIMA
- **Datasets**: Kaggle, UCI Time Series

---

## Conclusion

Time series forecasting requires understanding temporal patterns. Key takeaways:

1. **Understand Components**: Trend, seasonality, noise
2. **Preprocess**: Make stationary, normalize
3. **Choose Method**: Traditional vs ML vs Deep Learning
4. **Evaluate Properly**: Use time-aware metrics
5. **Update Regularly**: Retrain with new data

Remember: Time series data has temporal dependencies that must be preserved!

