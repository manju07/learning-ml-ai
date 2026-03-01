# Time Series Analysis and Forecasting: Comprehensive Guide

## Table of Contents
1. [Time Series Fundamentals](#1-time-series-fundamentals)
2. [Stationarity and Transformations](#2-stationarity-and-transformations)
3. [Decomposition Methods](#3-decomposition-methods)
4. [Classical Statistical Models](#4-classical-statistical-models)
5. [Exponential Smoothing Family](#5-exponential-smoothing-family)
6. [Vector Autoregression (VAR)](#6-vector-autoregression-var)
7. [Prophet: Facebook's Forecasting Tool](#7-prophet-facebooks-forecasting-tool)
8. [State Space Models and Kalman Filter](#8-state-space-models-and-kalman-filter)
9. [Machine Learning for Time Series](#9-machine-learning-for-time-series)
10. [Deep Learning for Time Series](#10-deep-learning-for-time-series)
11. [Evaluation Metrics and Cross-Validation](#11-evaluation-metrics-and-cross-validation)
12. [Anomaly Detection in Time Series](#12-anomaly-detection-in-time-series)
13. [Changepoint Detection](#13-changepoint-detection)
14. [Full End-to-End Examples](#14-full-end-to-end-examples)

---

## 1. Time Series Fundamentals

### 1.1 What is a Time Series?

A time series is a sequence of observations indexed in time order. Formally:

\[Y_t = f(t, \epsilon_t), \quad t = 1, 2, \ldots, T\]

where \(\epsilon_t\) is white noise and \(f\) captures the systematic patterns.

### 1.2 Components of Time Series

Every time series can be decomposed into:

**1. Trend (T):** Long-term increase or decrease in the mean level.
- Linear trend: \(T_t = \beta_0 + \beta_1 t\)
- Exponential trend: \(T_t = \beta_0 e^{\beta_1 t}\)

**2. Seasonality (S):** Regular, periodic patterns repeating at fixed intervals.
- Daily, weekly, monthly, annual cycles
- Deterministic and predictable

**3. Cyclicality (C):** Long-term fluctuations around the trend, non-periodic.
- Business cycles (2-10 years)
- Economic expansions and contractions

**4. Noise/Remainder (R):** Random, irregular variations after removing T, S, C.
- White noise: \(\epsilon_t \sim N(0, \sigma^2)\) i.i.d.

**Additive model:** \(Y_t = T_t + S_t + C_t + R_t\)
- Use when seasonal variations are roughly constant over time

**Multiplicative model:** \(Y_t = T_t \times S_t \times C_t \times R_t\)
- Use when seasonal variations grow proportionally with the level

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate time series with all components
np.random.seed(42)
n = 365 * 3  # 3 years of daily data
t = np.arange(n)

# Components
trend = 50 + 0.05 * t
seasonal_weekly = 10 * np.sin(2 * np.pi * t / 7)
seasonal_annual = 20 * np.sin(2 * np.pi * t / 365)
cycle = 15 * np.sin(2 * np.pi * t / (365 * 1.5))
noise = np.random.normal(0, 5, n)

# Additive model
y_additive = trend + seasonal_weekly + seasonal_annual + cycle + noise

# Multiplicative model
y_multiplicative = (1 + 0.001 * t) * (1 + 0.2 * np.sin(2*np.pi*t/365)) * (
    1 + 0.05 * np.random.normal(0, 1, n)
)

dates = pd.date_range('2021-01-01', periods=n, freq='D')
ts = pd.Series(y_additive, index=dates, name='value')

fig, axes = plt.subplots(5, 1, figsize=(14, 12))
axes[0].plot(dates, trend, color='blue'); axes[0].set_title('Trend')
axes[1].plot(dates[:60], seasonal_weekly[:60]+seasonal_annual[:60], color='orange')
axes[1].set_title('Seasonality (first 60 days)')
axes[2].plot(dates, cycle, color='green'); axes[2].set_title('Cycle')
axes[3].plot(dates[:60], noise[:60], color='gray', alpha=0.5); axes[3].set_title('Noise')
axes[4].plot(dates, ts, color='black'); axes[4].set_title('Combined (Additive)')
plt.tight_layout()
plt.show()
```

### 1.3 Autocorrelation and Partial Autocorrelation

**ACF (Autocorrelation Function):** Correlation between \(Y_t\) and \(Y_{t-k}\) at lag \(k\):

\[\rho_k = \frac{\text{Cov}(Y_t, Y_{t-k})}{\text{Var}(Y_t)} = \frac{\sum_{t=k+1}^{T}(Y_t - \bar{Y})(Y_{t-k} - \bar{Y})}{\sum_{t=1}^{T}(Y_t - \bar{Y})^2}\]

**PACF (Partial Autocorrelation Function):** Correlation between \(Y_t\) and \(Y_{t-k}\) after removing the effect of intermediate lags.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Plot ACF and PACF
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(ts, lags=50, ax=axes[0], alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF)')

plot_pacf(ts, lags=50, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Reading ACF/PACF patterns:
# AR(p): PACF cuts off after lag p, ACF tails off
# MA(q): ACF cuts off after lag q, PACF tails off
# ARMA(p,q): both ACF and PACF tail off
# Seasonal pattern: spikes at seasonal lags (e.g., lag 12 for monthly)
```

---

## 2. Stationarity and Transformations

### 2.1 What is Stationarity?

A time series \(\{Y_t\}_{t \geq 1}\) is **weakly stationary** if:
1. **Constant mean:** \(E[Y_t] = \mu\) for all \(t\)
2. **Constant variance:** \(\text{Var}(Y_t) = \sigma^2\) for all \(t\)
3. **Autocovariance depends only on lag:** \(\text{Cov}(Y_t, Y_{t-k}) = \gamma_k\) for all \(t\)

Many time series are non-stationary due to:
- **Trend stationarity:** Has trend but stationary around it → detrend
- **Difference stationarity (unit root):** Differencing makes it stationary → difference

### 2.2 Augmented Dickey-Fuller (ADF) Test

Tests H₀: Series has a unit root (non-stationary) against H₁: Stationary.

\[ADF\text{ model: } \Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta Y_{t-i} + \epsilon_t\]

H₀: \(\gamma = 0\) (unit root, non-stationary)

```python
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(series, name='Series', verbose=True):
    """
    Comprehensive stationarity testing using ADF and KPSS.
    ADF: H0 = non-stationary (unit root)
    KPSS: H0 = stationary (level/trend stationary)

    Using both tests together gives more robust conclusions:
    - ADF reject + KPSS fail to reject: Stationary
    - ADF fail to reject + KPSS reject: Non-stationary (unit root)
    - Both reject: Trend stationary (deterministic trend, not unit root)
    - Both fail to reject: Inconclusive
    """
    results = {}

    # ADF Test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['adf'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'n_lags': adf_result[2],
        'critical_values': adf_result[4]
    }

    # KPSS Test (level stationarity)
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    results['kpss_level'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3]
    }

    # KPSS Test (trend stationarity)
    kpss_trend_result = kpss(series.dropna(), regression='ct', nlags='auto')
    results['kpss_trend'] = {
        'statistic': kpss_trend_result[0],
        'p_value': kpss_trend_result[1]
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Stationarity Tests: {name}")
        print(f"{'='*50}")
        print(f"ADF Statistic: {results['adf']['statistic']:.4f}")
        print(f"ADF P-value: {results['adf']['p_value']:.6f}")
        print(f"ADF Critical Values: {results['adf']['critical_values']}")
        adf_stat = 'Stationary' if results['adf']['p_value'] < 0.05 else 'Non-stationary'
        print(f"ADF Conclusion: {adf_stat}")

        print(f"\nKPSS Statistic (level): {results['kpss_level']['statistic']:.4f}")
        print(f"KPSS P-value: {results['kpss_level']['p_value']:.6f}")
        kpss_stat = 'Stationary' if results['kpss_level']['p_value'] > 0.05 else 'Non-stationary'
        print(f"KPSS Conclusion: {kpss_stat}")

        # Combined interpretation
        adf_stationary = results['adf']['p_value'] < 0.05
        kpss_stationary = results['kpss_level']['p_value'] > 0.05
        if adf_stationary and kpss_stationary:
            print(f"\n→ Both tests agree: STATIONARY")
        elif not adf_stationary and not kpss_stationary:
            print(f"\n→ Both tests agree: NON-STATIONARY (unit root)")
        elif adf_stationary and not kpss_stationary:
            print(f"\n→ Possibly TREND STATIONARY (has deterministic trend)")
        else:
            print(f"\n→ Inconclusive - further investigation needed")

    return results
```

### 2.3 Making Series Stationary

```python
def make_stationary(series, max_diff=3, seasonal_period=None):
    """
    Apply transformations to achieve stationarity.
    Returns transformed series and the transformations applied.
    """
    transformations = []
    transformed = series.copy()

    # Step 1: Handle variance instability with log/Box-Cox
    if (transformed > 0).all():
        skewness_orig = abs(transformed.skew())
        log_transformed = np.log1p(transformed)
        skewness_log = abs(log_transformed.skew())

        if skewness_log < skewness_orig:
            transformed = log_transformed
            transformations.append('log')
            print(f"Applied log transformation (skewness: {skewness_orig:.3f} → {skewness_log:.3f})")

    # Step 2: Seasonal differencing (remove seasonality)
    if seasonal_period:
        result = test_stationarity(transformed, verbose=False)
        if result['kpss_level']['p_value'] < 0.05:  # Non-stationary
            transformed = transformed.diff(seasonal_period).dropna()
            transformations.append(f'seasonal_diff_{seasonal_period}')
            print(f"Applied seasonal differencing (period={seasonal_period})")

    # Step 3: First differencing (remove trend)
    for d in range(1, max_diff + 1):
        result = test_stationarity(transformed, verbose=False)
        if result['adf']['p_value'] < 0.05:  # Already stationary
            print(f"Series is stationary after {d-1} difference(s)")
            break
        transformed = transformed.diff(1).dropna()
        transformations.append(f'diff_{d}')
        print(f"Applied differencing (d={d})")

    return transformed, transformations


# Box-Cox transformation (generalizes log transformation)
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def apply_boxcox(series):
    """Apply Box-Cox transformation to stabilize variance."""
    assert (series > 0).all(), "Box-Cox requires positive values"
    transformed, lambda_val = boxcox(series)
    print(f"Box-Cox lambda: {lambda_val:.4f}")
    print(f"(lambda=0 → log, lambda=1 → no transform, lambda=0.5 → sqrt)")
    return pd.Series(transformed, index=series.index), lambda_val

def inverse_boxcox(series, lambda_val):
    """Reverse Box-Cox transformation for forecast interpretation."""
    return pd.Series(inv_boxcox(series.values, lambda_val), index=series.index)
```

---

## 3. Decomposition Methods

### 3.1 Classical Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition: Y = T + S + R
decomp_additive = seasonal_decompose(ts, model='additive', period=365)

# Multiplicative decomposition: Y = T * S * R
decomp_mult = seasonal_decompose(ts, model='multiplicative', period=365)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
for i, (decomp, title) in enumerate([(decomp_additive, 'Additive'),
                                       (decomp_mult, 'Multiplicative')]):
    decomp.observed.plot(ax=axes[0, i], title=f'{title}: Observed')
    decomp.trend.plot(ax=axes[1, i], title=f'{title}: Trend')
    decomp.seasonal.plot(ax=axes[2, i], title=f'{title}: Seasonal')
    decomp.resid.plot(ax=axes[3, i], title=f'{title}: Residual')
plt.tight_layout()
plt.show()

# Check residuals (should be white noise)
residuals = decomp_additive.resid.dropna()
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(residuals, lags=20, return_df=True)
print("Ljung-Box test (residuals should be white noise):")
print(lb_result)
```

### 3.2 STL Decomposition (Seasonal and Trend using Loess)

STL is more robust and flexible than classical decomposition:
- Handles multiple seasonal periods
- Robust to outliers
- Allows varying seasonal component

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition
stl = STL(ts,
           period=365,           # Annual seasonality
           seasonal=13,          # Seasonal window (odd number)
           trend=None,           # Auto-calculated
           low_pass=None,
           seasonal_deg=1,       # 0=constant, 1=linear local fit
           trend_deg=1,
           low_pass_deg=1,
           robust=True)           # Robust to outliers

result = stl.fit()

fig = result.plot()
plt.suptitle('STL Decomposition', fontsize=16)
plt.tight_layout()
plt.show()

# STL with multiple seasonal periods (e.g., daily + weekly)
from statsmodels.tsa.seasonal import MSTL

# MSTL for data with multiple seasonalities
hourly_data = pd.Series(np.random.randn(24 * 365), name='load')
mstl = MSTL(hourly_data, periods=[24, 24*7])  # Daily and weekly seasonality
mstl_result = mstl.fit()
print(mstl_result.seasonal.head())
```

---

## 4. Classical Statistical Models

### 4.1 AR Model (Autoregressive)

An AR(p) model expresses the current value as a linear combination of p past values:

\[Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t\]

**Parameter estimation:** OLS regression with lagged values as predictors.
**Stationarity condition:** All roots of the characteristic polynomial \(1 - \phi_1 z - \phi_2 z^2 - \cdots - \phi_p z^p = 0\) must lie outside the unit circle.

```python
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf

# Select AR order using PACF (spikes at lag 1..p, then cuts off)
pacf_values, confint = pacf(ts.dropna(), nlags=20, alpha=0.05)
ar_order = sum(1 for i, (lo, hi) in enumerate(confint[1:], 1)
               if not (lo < 0 < hi))  # Count significant lags
print(f"Suggested AR order: {ar_order}")

# Fit AR model
ar_model = AutoReg(ts, lags=ar_order, old_names=False)
ar_result = ar_model.fit()
print(ar_result.summary())

# Forecast
ar_forecast = ar_result.predict(start=len(ts), end=len(ts)+30)
```

### 4.2 MA Model (Moving Average)

An MA(q) model expresses the current value as a linear combination of q past error terms:

\[Y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}\]

**Invertibility condition:** All roots of the MA polynomial must lie outside the unit circle.
**ACF pattern:** ACF cuts off after lag q; PACF tails off.

### 4.3 ARMA Model

Combines AR and MA components for more parsimonious models:

\[Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}\]

**Information criteria for order selection:**
- AIC: \(\text{AIC} = 2k - 2\ln(\hat{L})\), where \(k\) = number of parameters
- BIC: \(\text{BIC} = k\ln(n) - 2\ln(\hat{L})\) (penalizes complexity more)
- Select the model with the smallest AIC/BIC

```python
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA

def select_arma_order(series, max_p=5, max_q=5, criterion='aic'):
    """
    Select ARMA order using information criteria (AIC/BIC).
    Exhaustive search over all (p, q) combinations.
    """
    best_criterion = np.inf
    best_order = (0, 0)
    results = []

    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model = ARIMA(series, order=(p, 0, q))
                result = model.fit()
                val = result.aic if criterion == 'aic' else result.bic
                results.append({'p': p, 'q': q, criterion: val})
                if val < best_criterion:
                    best_criterion = val
                    best_order = (p, q)
        except Exception:
            pass

    results_df = pd.DataFrame(results).sort_values(criterion)
    print(f"Best ARMA order: ({best_order[0]}, {best_order[1]}), {criterion.upper()}={best_criterion:.2f}")
    print(results_df.head(10).to_string())

    return best_order, results_df
```

### 4.4 ARIMA Model: Full Implementation

ARIMA(p, d, q) = AR(p) + Integrated(d) + MA(q)

The **d** parameter: number of times series is differenced to achieve stationarity.

\[\text{ARIMA}(p,d,q): \phi(B)(1-B)^d Y_t = c + \theta(B)\epsilon_t\]

where \(B\) is the backshift operator: \(BY_t = Y_{t-1}\)

```python
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings

# Manual ARIMA fitting
def fit_arima_manual(train, test, p, d, q):
    """Fit ARIMA and evaluate on test set."""
    model = ARIMA(train, order=(p, d, q))
    fitted = model.fit()

    print(fitted.summary())
    print(f"\nAIC: {fitted.aic:.2f}")
    print(f"BIC: {fitted.bic:.2f}")

    # Diagnostic plots
    fig = fitted.plot_diagnostics(figsize=(14, 10))
    plt.suptitle(f'ARIMA({p},{d},{q}) Diagnostics', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Forecast
    forecast_result = fitted.get_forecast(steps=len(test))
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Test', color='black')
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
    plt.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     alpha=0.3, color='red', label='95% CI')
    plt.legend()
    plt.title(f'ARIMA({p},{d},{q}) Forecast')
    plt.show()

    return fitted, forecast_mean


# Auto ARIMA: automatically selects best order
def auto_arima_fit(train, test, seasonal=True, m=12):
    """
    Auto ARIMA using stepwise search.
    Parameters:
    - seasonal: whether to include seasonal components
    - m: seasonal period (12=monthly, 7=daily, 4=quarterly)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = auto_arima(
            train,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,                    # Auto-determine d
            seasonal=seasonal,
            m=m,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            D=None,                    # Auto-determine D
            information_criterion='aic',
            stepwise=True,             # Faster than exhaustive
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            n_jobs=-1
        )

    print(model.summary())
    print(f"\nSelected order: {model.order}")
    print(f"Selected seasonal order: {model.seasonal_order}")

    forecasts = model.predict(n_periods=len(test))
    return model, forecasts
```

### 4.5 SARIMA: Seasonal ARIMA

SARIMA(p, d, q)(P, D, Q)[m] adds seasonal AR and MA components:

\[\Phi(B^m)\phi(B)(1-B)^d(1-B^m)^D Y_t = \Theta(B^m)\theta(B)\epsilon_t\]

where:
- P: seasonal AR order
- D: seasonal differencing order
- Q: seasonal MA order
- m: seasonal period

**ACF/PACF for seasonal order selection:**
- Spikes at multiples of m in ACF → seasonal MA (Q)
- Spikes at multiples of m in PACF → seasonal AR (P)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(train, test, order, seasonal_order):
    """
    Fit SARIMA model with full diagnostics and forecast.

    Parameters:
    -----------
    order: (p, d, q) - non-seasonal component
    seasonal_order: (P, D, Q, m) - seasonal component
    """
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)

    print(result.summary())

    # Rolling forecasts (walk-forward validation)
    predictions = []
    for i in range(len(test)):
        # Fit on all available data up to this point
        model_step = SARIMAX(
            pd.concat([train, test.iloc[:i]]),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result_step = model_step.fit(disp=False, start_params=result.params)
        pred = result_step.forecast(steps=1)
        predictions.append(pred.iloc[0])

    predictions = pd.Series(predictions, index=test.index)
    mae = np.mean(np.abs(test - predictions))
    rmse = np.sqrt(np.mean((test - predictions)**2))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    print(f"\nWalk-forward evaluation:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

    return result, predictions


# ARIMAX: ARIMA with exogenous variables
def fit_arimax(train, test, exog_train, exog_test, order, seasonal_order=None):
    """
    ARIMAX model with external regressors.
    Example: Demand forecasting with price and promotion as exogenous variables.
    """
    if seasonal_order:
        model = SARIMAX(train, exog=exog_train, order=order,
                        seasonal_order=seasonal_order, enforce_stationarity=False)
    else:
        model = SARIMAX(train, exog=exog_train, order=order, enforce_stationarity=False)

    result = model.fit(disp=False)
    forecast = result.forecast(steps=len(test), exog=exog_test)

    return result, forecast
```

---

## 5. Exponential Smoothing Family

### 5.1 Simple Exponential Smoothing (SES)

For series with no trend or seasonality. The forecast is a weighted average of all past observations, with exponentially decaying weights:

\[\hat{Y}_{t+1} = \alpha Y_t + (1-\alpha)\hat{Y}_t\]

Expanded: \(\hat{Y}_{t+1} = \alpha Y_t + \alpha(1-\alpha)Y_{t-1} + \alpha(1-\alpha)^2 Y_{t-2} + \cdots\)

where \(\alpha \in (0,1)\) is the smoothing parameter (higher → more weight on recent observations).

### 5.2 Holt's Linear Method (Double Exponential Smoothing)

Handles linear trend with two components: level (\(\ell\)) and trend (\(b\)):

\[\ell_t = \alpha Y_t + (1-\alpha)(\ell_{t-1} + b_{t-1})\]
\[b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)b_{t-1}\]
\[\hat{Y}_{t+h} = \ell_t + h \cdot b_t\]

### 5.3 Holt-Winters Method (Triple Exponential Smoothing)

Handles trend and seasonality:

**Additive seasonality:**
\[\ell_t = \alpha(Y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})\]
\[b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)b_{t-1}\]
\[s_t = \gamma(Y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m}\]
\[\hat{Y}_{t+h} = \ell_t + h \cdot b_t + s_{t+h-m}\]

```python
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                          SimpleExpSmoothing, Holt)

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(train, initialization_method='estimated')
ses_fit = ses_model.fit(optimized=True)  # Optimize alpha
print(f"Optimal alpha: {ses_fit.params['smoothing_level']:.4f}")

# Holt's Linear Method
holt_model = Holt(train, initialization_method='estimated')
holt_fit = holt_model.fit(optimized=True)
print(f"Optimal alpha: {holt_fit.params['smoothing_level']:.4f}")
print(f"Optimal beta: {holt_fit.params['smoothing_trend']:.4f}")

holt_forecast = holt_fit.forecast(steps=len(test))

# Holt-Winters (Triple Exponential Smoothing)
hw_additive = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=12,       # Monthly seasonality
    initialization_method='estimated',
    use_boxcox=True            # Apply Box-Cox for variance stabilization
)
hw_add_fit = hw_additive.fit(optimized=True)
hw_forecast = hw_add_fit.forecast(steps=len(test))

# Multiplicative variant
hw_multiplicative = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='mul',            # Multiplicative seasonality
    seasonal_periods=12,
    damped_trend=True          # Damped trend avoids over-forecasting
)
hw_mul_fit = hw_multiplicative.fit(optimized=True)

# ETS (Error, Trend, Seasonality) state space version
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

ets_model = ETSModel(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    error='add'                # Error type: 'add' or 'mul'
)
ets_fit = ets_model.fit(disp=False)
print(ets_fit.summary())

# Prediction intervals from ETS
forecast_result = ets_fit.get_prediction(
    start=len(train), end=len(train)+len(test)-1
)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.pred_int(alpha=0.05)
```

---

## 6. Vector Autoregression (VAR)

VAR models multiple time series jointly, capturing Granger causality relationships.

\[\mathbf{Y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{Y}_{t-1} + \mathbf{A}_2 \mathbf{Y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{Y}_{t-p} + \boldsymbol{\epsilon}_t\]

where \(\mathbf{Y}_t\) is a \(k\)-dimensional vector, \(\mathbf{A}_i\) are \(k \times k\) coefficient matrices.

**Granger Causality:** Variable \(X\) "Granger causes" \(Y\) if past values of \(X\) help predict \(Y\) beyond what past values of \(Y\) alone can do.

```python
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Example: GDP and unemployment (macroeconomic forecasting)
# Both series must be stationary for VAR
multivariate_df = pd.DataFrame({
    'gdp_growth': np.random.randn(200) + 0.1,
    'unemployment': np.random.randn(200),
    'inflation': np.random.randn(200)
})

# Check for cointegration (if so, use VECM instead of VAR)
johansen_result = coint_johansen(multivariate_df, det_order=0, k_ar_diff=1)
print("Johansen Cointegration Test:")
print(f"Trace statistics: {johansen_result.lr1}")
print(f"95% critical values: {johansen_result.cvt[:, 1]}")

# Fit VAR model
var_model = VAR(multivariate_df)

# Select optimal lag order
lag_order_results = var_model.select_order(maxlags=10)
print(f"\nOptimal lag orders:")
print(lag_order_results.summary())
optimal_lag = lag_order_results.aic  # Or lag_order_results.bic

var_result = var_model.fit(maxlags=optimal_lag, ic='aic')
print(var_result.summary())

# Granger causality test
gc_test = var_result.test_causality('gdp_growth', ['unemployment'], kind='f')
print(f"\nGranger Causality (unemployment -> gdp): p={gc_test.pvalue:.4f}")

# Impulse Response Functions (IRF)
# Shows how shocks to one variable propagate to others
irf = var_result.irf(periods=20)
irf.plot(orth=True)  # Orthogonalized IRF
plt.show()

# Forecast
n_forecast = 12
var_forecast = var_result.forecast(multivariate_df.values[-optimal_lag:], steps=n_forecast)
var_forecast_df = pd.DataFrame(var_forecast, columns=multivariate_df.columns)
print("\nVAR Forecast:")
print(var_forecast_df)

# Forecast Error Variance Decomposition (FEVD)
# Shows what proportion of forecast error variance is explained by each variable
fevd = var_result.fevd(periods=10)
fevd.plot()
plt.show()
```

---

## 7. Prophet: Facebook's Forecasting Tool

Prophet is designed for business forecasting with:
- Automatic trend change detection
- Multiple seasonality handling
- Holiday effects
- Robustness to missing data and outliers

**Mathematical model:**

\[y(t) = g(t) + s(t) + h(t) + \epsilon_t\]

- **g(t):** Trend function (piecewise linear or logistic growth)
- **s(t):** Seasonality (Fourier series decomposition)
- **h(t):** Holiday effects
- **\(\epsilon_t\):** Error term

```python
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
import pandas as pd

# Prepare data (Prophet requires 'ds' and 'y' columns)
prophet_df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=730, freq='D'),
    'y': np.random.randn(730).cumsum() + 100
})

# Basic Prophet model
model = Prophet(
    growth='linear',               # 'linear' or 'logistic' (for bounded growth)
    changepoint_prior_scale=0.05,  # Flexibility of trend changes (higher = more flexible)
    seasonality_prior_scale=10,    # Strength of seasonality
    holidays_prior_scale=10,       # Strength of holiday effects
    seasonality_mode='additive',   # 'additive' or 'multiplicative'
    changepoint_range=0.8,         # Proportion of history to search for changepoints
    n_changepoints=25,             # Number of potential changepoints
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    uncertainty_samples=1000       # Monte Carlo samples for uncertainty intervals
)

# Add custom seasonality (e.g., quarterly)
model.add_seasonality(
    name='quarterly',
    period=91.25,
    fourier_order=5               # Higher = more complex seasonality pattern
)

# Add country holidays
from prophet.make_holidays import make_holidays_df
holidays = make_holidays_df(year_list=[2020, 2021, 2022, 2023], country='US')
model_with_holidays = Prophet(
    holidays=holidays,
    holidays_prior_scale=10
)

# Add custom events/regressors
model.add_regressor('promotion_flag')  # External regressor
prophet_df['promotion_flag'] = 0  # Binary: was there a promotion?

model.fit(prophet_df)

# Create future dataframe
future = model.make_future_dataframe(periods=365, freq='D')
future['promotion_flag'] = 0

forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

# Visualize
fig1 = model.plot(forecast)
add_changepoints_to_plot(fig1.gca(), model, forecast)
plt.title('Prophet Forecast with Changepoints')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

# Cross-validation
cv_results = cross_validation(
    model,
    initial='365 days',    # Training size for first fold
    period='90 days',      # Spacing between cutoff dates
    horizon='30 days',     # Forecast horizon
    parallel='processes'
)
print("\nCross-validation results:")
print(cv_results.head())

metrics_df = performance_metrics(cv_results)
print("\nPerformance metrics:")
print(metrics_df)

fig = plot_cross_validation_metric(cv_results, metric='mape')
plt.show()

# Hyperparameter tuning for Prophet
from itertools import product

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

all_params = [dict(zip(param_grid.keys(), v))
              for v in product(*param_grid.values())]

cv_scores = []
for params in all_params:
    m = Prophet(**params).fit(prophet_df)
    df_cv = cross_validation(m, initial='365 days', period='90 days',
                              horizon='30 days', disable_tqdm=True)
    df_p = performance_metrics(df_cv, rolling_window=1)
    cv_scores.append(df_p['rmse'].values[0])

best_params = all_params[np.argmin(cv_scores)]
print(f"Best params: {best_params}")
```

---

## 8. State Space Models and Kalman Filter

### 8.1 State Space Representation

The general state space model:

**State transition:** \(\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{G} \mathbf{w}_t, \quad \mathbf{w}_t \sim N(\mathbf{0}, \mathbf{Q})\)

**Observation model:** \(\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim N(\mathbf{0}, \mathbf{R})\)

### 8.2 Kalman Filter

The Kalman filter provides optimal linear prediction in two steps:

**Predict (prior):**
\[\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\hat{\mathbf{x}}_{t-1|t-1}\]
\[\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1|t-1}\mathbf{F}^T + \mathbf{Q}\]

**Update (posterior):**
\[\mathbf{K}_t = \mathbf{P}_{t|t-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^T + \mathbf{R})^{-1}\]
\[\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(\mathbf{y}_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})\]
\[\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t\mathbf{H})\mathbf{P}_{t|t-1}\]

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

# Unobserved Components Model (structural time series)
# Decomposes into trend + seasonal + irregular
uc_model = UnobservedComponents(
    train,
    level='local linear trend',   # Smooth trend
    seasonal=12,                   # Monthly seasonality
    autoregressive=1,              # AR(1) component for irregular
    stochastic_level=True,         # Allow level to change stochastically
    stochastic_trend=True,
    stochastic_seasonal=True
)

uc_result = uc_model.fit(disp=False)
print(uc_result.summary())

# Extract components
components = uc_result.components
print(f"Trend: {components.loc[:, 'level'].values[:5]}")

# Forecast with prediction intervals
forecast = uc_result.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Kalman smoother (uses all observations, not just past)
smoothed_states = uc_result.smoother_results
```

---

## 9. Machine Learning for Time Series

### 9.1 Feature Engineering for Time Series

```python
import pandas as pd
import numpy as np

def create_time_series_features(df, target_col, date_col='date',
                                  lags=[1, 7, 14, 21, 28],
                                  rolling_windows=[7, 14, 30, 90]):
    """
    Comprehensive feature engineering for time series ML models.
    Creates lag, rolling, calendar, and target-derived features.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 1. Lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # 2. Rolling statistics (based on past only - no data leakage)
    for window in rolling_windows:
        df[f'{target_col}_rolling_mean_{window}'] = (
            df[target_col].shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'{target_col}_rolling_std_{window}'] = (
            df[target_col].shift(1).rolling(window, min_periods=1).std()
        )
        df[f'{target_col}_rolling_min_{window}'] = (
            df[target_col].shift(1).rolling(window, min_periods=1).min()
        )
        df[f'{target_col}_rolling_max_{window}'] = (
            df[target_col].shift(1).rolling(window, min_periods=1).max()
        )
        df[f'{target_col}_rolling_median_{window}'] = (
            df[target_col].shift(1).rolling(window, min_periods=1).median()
        )

    # 3. Exponentially weighted moving averages
    for span in [7, 30]:
        df[f'{target_col}_ewm_{span}'] = (
            df[target_col].shift(1).ewm(span=span).mean()
        )

    # 4. Calendar features
    dt = df[date_col]
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['dayofweek'] = dt.dt.dayofweek
    df['dayofyear'] = dt.dt.dayofyear
    df['week'] = dt.dt.isocalendar().week.astype(int)
    df['quarter'] = dt.dt.quarter
    df['is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df['is_month_start'] = dt.dt.is_month_start.astype(int)
    df['is_month_end'] = dt.dt.is_month_end.astype(int)
    df['is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)

    # 5. Fourier features (for capturing cyclical patterns)
    # These encode the cyclical nature of time better than raw month/dayofweek
    for period in [7, 30, 365]:
        for order in [1, 2, 3]:
            df[f'sin_{period}_{order}'] = np.sin(2 * np.pi * order * df['dayofyear'] / period)
            df[f'cos_{period}_{order}'] = np.cos(2 * np.pi * order * df['dayofyear'] / period)

    # 6. Target-derived features
    # Year-over-year change
    df[f'{target_col}_yoy_change'] = df[target_col].shift(365)
    df[f'{target_col}_yoy_pct'] = (df[target_col] - df[target_col].shift(365)) / df[target_col].shift(365)

    # Week-over-week change
    df[f'{target_col}_wow_pct'] = df[target_col].pct_change(7)

    # 7. Trend features
    df['time_index'] = (df[date_col] - df[date_col].min()).dt.days
    df['time_index_sq'] = df['time_index'] ** 2

    return df.dropna()


# LightGBM for Time Series
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def lgbm_time_series_forecast(df, target_col, feature_cols,
                                date_col='date', n_splits=5, horizon=30):
    """
    LightGBM with time series cross-validation.
    Uses TimeSeriesSplit to prevent data leakage.
    """
    df = df.sort_values(date_col)
    X = df[feature_cols]
    y = df[target_col]

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=horizon)

    lgb_params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'num_leaves': 63,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    cv_scores = []
    models = []
    feature_importances = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1)
            ]
        )

        val_pred = model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - val_pred)**2))
        mae = np.mean(np.abs(y_val - val_pred))
        mape = np.mean(np.abs((y_val - val_pred) / (y_val + 1e-8))) * 100

        cv_scores.append({'fold': fold+1, 'rmse': rmse, 'mae': mae, 'mape': mape,
                          'n_estimators': model.best_iteration})
        models.append(model)

        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        })
        feature_importances.append(fi)
        print(f"Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%")

    cv_df = pd.DataFrame(cv_scores)
    print(f"\nCV Summary:")
    print(cv_df[['rmse', 'mae', 'mape']].agg(['mean', 'std']))

    # Average feature importances
    fi_avg = pd.concat(feature_importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
    print(f"\nTop 20 Feature Importances:")
    print(fi_avg.head(20))

    return models, cv_df, fi_avg
```

---

## 10. Deep Learning for Time Series

### 10.1 LSTM (Long Short-Term Memory)

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM for time series forecasting.
    Handles univariate and multivariate input.
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM output
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Use last time step output
        out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)        # (batch_size, output_size)
        return out


class BiLSTMForecaster(nn.Module):
    """Bidirectional LSTM - useful for imputation and smoothing."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2):
        super(BiLSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def create_sequences(data, seq_len, horizon=1):
    """Create (X, y) pairs for sequence-to-sequence learning."""
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon])
    return np.array(X), np.array(y)


def train_lstm(train_data, val_data, seq_len=60, horizon=7, epochs=100,
               batch_size=32, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Full LSTM training loop with early stopping, LR scheduling, gradient clipping.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, seq_len, horizon)
    X_val, y_val = create_sequences(val_scaled, seq_len, horizon)

    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, seq_len, 1)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val_t = torch.FloatTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                             batch_size=batch_size)

    model = LSTMForecaster(input_size=1, hidden_size=128, num_layers=2,
                            output_size=horizon).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                             factor=0.5, verbose=True)
    criterion = nn.HuberLoss()  # Robust to outliers vs MSE

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_lstm.pt')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load('best_lstm.pt'))
    return model, scaler, (train_losses, val_losses)
```

### 10.2 Temporal Convolutional Networks (TCN)

```python
class CausalConv1d(nn.Module):
    """Causal convolution ensures no future information leakage."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # Remove future padding


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with residual connection."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network.
    Uses dilated causal convolutions to capture long-range dependencies.
    Receptive field: kernel_size * (2^n_layers - 1)
    """
    def __init__(self, input_size, output_size, hidden_channels=64,
                 kernel_size=3, n_layers=8):
        super().__init__()
        layers = []
        channels = [input_size] + [hidden_channels] * n_layers
        for i in range(n_layers):
            dilation = 2 ** i  # Exponentially increasing dilation
            layers.append(TCNBlock(channels[i], channels[i+1], kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_channels, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> transpose to (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Take last time step
        out = self.fc(out[:, :, -1])
        return out
```

### 10.3 Temporal Fusion Transformer (TFT)

```python
# pip install pytorch-forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl

# TFT expects specific data format
max_encoder_length = 60  # Look-back window
max_prediction_length = 30  # Forecast horizon

training_cutoff = df['time_idx'].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df['time_idx'] <= training_cutoff],
    time_idx='time_idx',
    target='value',
    group_ids=['series_id'],             # For multiple time series
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=['series_id'],   # Time-invariant categorical
    static_reals=[],                     # Time-invariant numerical
    time_varying_known_categoricals=['month', 'dayofweek'],  # Known future
    time_varying_known_reals=['time_idx', 'price_promotion'],  # Known future
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=['value', 'lag_7', 'rolling_mean_30'],  # Unknown future
    target_normalizer='softplus',
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

validation = TimeSeriesDataSet.from_dataset(
    training, df, predict=True, stop_randomization=True
)

train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,                  # Number of quantiles
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4
)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='auto',
    gradient_clip_val=0.1
)
trainer.fit(tft, train_dataloaders=train_dataloader,
             val_dataloaders=val_dataloader)

# Interpret TFT attention patterns
interpretation = tft.interpret_output(
    tft.predict(val_dataloader, mode='raw', return_x=True),
    reduction='sum'
)
tft.plot_interpretation(interpretation)
```

### 10.4 N-BEATS and N-HiTS

**N-BEATS** (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting, Oreshkin et al., 2019) is a pure deep learning architecture with *no* seasonality decomposition, trends, or hand-crafted features. It achieves strong performance through:

1. **Dual residual stacking**: Each block has a *backcast* residual (subtracted from input) and a *forecast* residual (added to output). Blocks are stacked so each sees the residual of the previous—analogous to gradient boosting for time series.

2. **Interpretable basis functions**: Two stack types—*trend* (polynomials) and *seasonality* (Fourier harmonics)—allow decomposition of predictions into trend + seasonal components without explicit modeling.

3. **Fully connected architecture**: No RNNs or convolutions; each block is an MLP that takes the lookback window and outputs expansion coefficients. Simple, fast, and parallelizable.

**N-HiTS** improves N-BEATS with *hierarchical interpolation* and *multi-rate sampling*: different stacks operate at different resolutions (e.g., long-term trend vs. short-term seasonality), reducing parameters and improving long-horizon accuracy.

```python
# pip install neuralforecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, PatchTST
from neuralforecast.losses.pytorch import MAE, MSE

# N-BEATS: Neural Basis Expansion Analysis for Time Series
nbeats = NBEATS(
    h=28,                    # Forecast horizon
    input_size=2 * 28,       # Look-back window (2x horizon is common)
    stack_types=['trend', 'seasonality'],  # Interpretable architecture
    n_blocks=[3, 3],         # Blocks per stack
    mlp_units=[[512, 512], [512, 512]],
    n_harmonics=2,
    n_polynomials=2,
    dropout_prob_theta=0.0,
    max_steps=1000,
    learning_rate=1e-3,
    loss=MAE()
)

# N-HiTS: improved version with multi-rate signal sampling
nhits = NHITS(
    h=28,
    input_size=2 * 28,
    stack_types=['identity', 'identity', 'identity'],
    n_blocks=[1, 1, 1],
    n_freq_downsample=[4, 2, 1],  # Multi-rate downsampling
    interpolation_mode='linear',
    max_steps=1000
)

# PatchTST: Transformer with patch-based tokenization
patchtst = PatchTST(
    h=28,
    input_size=2 * 28,
    patch_len=16,            # Subseries patch length
    stride=8,
    d_model=128,
    n_heads=8,
    d_ff=256,
    dropout=0.1,
    max_steps=500
)

nf = NeuralForecast(
    models=[nbeats, nhits, patchtst],
    freq='D'
)

# Prepare data in long format
df_long = pd.DataFrame({
    'unique_id': ['series_1'] * len(ts),
    'ds': ts.index,
    'y': ts.values
})

nf.fit(df=df_long)
forecasts = nf.predict()
print(forecasts.head(10))
```

### 10.5 Transformer-Based Time Series

Transformers have been adapted for time series via **patch-based tokenization** (treating subseries as tokens) and **temporal positional encoding**. Key architectures:

| Model | Key Idea | Best For |
|-------|----------|----------|
| **PatchTST** | Patches as tokens; channel-independent | Long-horizon, multivariate |
| **Informer** | Prob sparse attention; memory efficient | Very long sequences |
| **Autoformer** | Auto-correlation mechanism | Seasonal patterns |
| **FEDformer** | Frequency-domain attention | Multi-scale seasonality |
| **TimesFM** | Foundation model for TS | Zero-shot, few-shot |

**PatchTST** (Patch Time Series Transformer): Splits the input window into non-overlapping patches (e.g., length 16). Each patch is linearly projected to a token. The transformer attends over patches. Benefits: fewer tokens than pointwise, captures local patterns, scales to long horizons.

```python
# Transformer for time series with PyTorch
import torch
import torch.nn as nn
import math

class PatchTSEncoder(nn.Module):
    """
    Simplified patch-based Transformer for time series forecasting.
    Each patch is a local subsequence; positional encoding handles temporal order.
    """
    def __init__(self, seq_len=96, patch_len=16, n_features=1, d_model=64, n_heads=4, n_layers=2, pred_len=24):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_len * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.n_patches, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        B, T, C = x.shape
        patches = x.unfold(1, self.patch_len, self.patch_len)  # [B, n_patches, patch_len, C]
        patches = patches.reshape(B, self.n_patches, -1)
        tokens = self.patch_embed(patches) + self.pos_embed
        out = self.transformer(tokens)
        out = out.reshape(B, -1)
        return self.head(out)

# Example usage
model = PatchTSEncoder(seq_len=96, patch_len=16, n_features=1, pred_len=24)
x = torch.randn(32, 96, 1)
pred = model(x)  # [32, 24]
```

---

## 11. Evaluation Metrics and Cross-Validation

### 11.1 Forecasting Metrics

```python
import numpy as np
from scipy.stats import norm

def evaluate_forecasts(y_true, y_pred, y_pred_lower=None, y_pred_upper=None,
                        seasonality=None, y_train=None):
    """
    Comprehensive forecasting evaluation metrics.

    Returns dict with: MAE, RMSE, MAPE, sMAPE, MASE, Winkler score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    results = {}

    # MAE: Mean Absolute Error
    results['MAE'] = np.mean(np.abs(y_true - y_pred))

    # RMSE: Root Mean Squared Error (penalizes large errors more)
    results['RMSE'] = np.sqrt(np.mean((y_true - y_pred)**2))

    # MAPE: Mean Absolute Percentage Error (%, not suitable when y_true ≈ 0)
    mask = y_true != 0
    results['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # sMAPE: Symmetric MAPE (handles near-zero values better)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    results['sMAPE'] = np.mean(np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)) * 100

    # MASE: Mean Absolute Scaled Error (scale-free, relative to naive forecast)
    # Naive forecast: previous observation (or seasonal lag)
    if y_train is not None:
        if seasonality:
            naive_errors = np.abs(np.diff(y_train, n=seasonality))
        else:
            naive_errors = np.abs(np.diff(y_train))
        scaling = np.mean(naive_errors)
        if scaling > 0:
            results['MASE'] = results['MAE'] / scaling

    # R² Score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    results['R2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Winkler Score: evaluates prediction interval quality
    # Rewards accurate intervals, penalizes misses by width of interval
    if y_pred_lower is not None and y_pred_upper is not None:
        alpha = 0.05  # 95% PI
        width = y_pred_upper - y_pred_lower
        below = y_true < y_pred_lower
        above = y_true > y_pred_upper

        winkler = width.copy()
        winkler[below] += (2 / alpha) * (y_pred_lower[below] - y_true[below])
        winkler[above] += (2 / alpha) * (y_true[above] - y_pred_upper[above])
        results['Winkler_Score'] = np.mean(winkler)

        # Coverage: % of true values within PI
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        results['PI_Coverage'] = coverage * 100
        results['PI_Width'] = np.mean(width)

    print("\nForecasting Evaluation Metrics:")
    print("=" * 40)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    return results
```

### 11.2 Time Series Cross-Validation

```python
import pandas as pd
import numpy as np

def walk_forward_validation(df, model_func, target_col='y',
                              initial_train_size=0.6,
                              step_size=1, horizon=7,
                              expanding=True):
    """
    Walk-forward (expanding or sliding window) time series cross-validation.

    Types:
    - Expanding window: train on all past data (increases each step)
    - Sliding window: fixed window size (more useful for concept drift)

    Parameters:
    -----------
    model_func: Callable(train_df, horizon) -> predictions
    initial_train_size: fraction or int for initial training set
    step_size: how many steps to advance per fold
    horizon: forecast horizon for each fold
    expanding: True for expanding window, False for sliding window
    """
    n = len(df)
    if isinstance(initial_train_size, float):
        initial_n = int(n * initial_train_size)
    else:
        initial_n = initial_train_size

    all_predictions = []
    all_actuals = []
    fold_metrics = []

    t = initial_n
    fold = 0

    while t + horizon <= n:
        if expanding:
            train = df.iloc[:t]
        else:
            window = initial_n
            train = df.iloc[max(0, t - window):t]

        test = df.iloc[t:t + horizon]

        # Fit model and generate forecast
        predictions = model_func(train, horizon)

        actuals = test[target_col].values
        predictions = np.array(predictions)[:horizon]

        all_predictions.extend(predictions)
        all_actuals.extend(actuals)

        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions)**2))
        fold_metrics.append({'fold': fold, 'cutoff': df.index[t-1],
                             'mae': mae, 'rmse': rmse})

        t += step_size
        fold += 1

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\nWalk-Forward Validation ({fold} folds, {'Expanding' if expanding else 'Sliding'} Window)")
    print(f"MAE: {np.mean(np.abs(np.array(all_actuals) - np.array(all_predictions))):.4f} "
          f"± {np.std([m['mae'] for m in fold_metrics]):.4f}")
    print(f"RMSE: {np.sqrt(np.mean((np.array(all_actuals) - np.array(all_predictions))**2)):.4f}")

    # Plot fold performance over time
    plt.figure(figsize=(12, 5))
    plt.plot(metrics_df['cutoff'], metrics_df['mae'], 'b-o', label='MAE per fold')
    plt.xlabel('Cutoff Date')
    plt.ylabel('MAE')
    plt.title('Fold Performance Over Time (Walk-Forward CV)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return metrics_df, all_actuals, all_predictions
```

---

## 12. Anomaly Detection in Time Series

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.statespace.structural import UnobservedComponents

def detect_time_series_anomalies(ts, methods=['statistical', 'iqr', 'isolation_forest'],
                                   window=30, z_threshold=3.5):
    """
    Multiple methods for time series anomaly detection.

    Methods:
    1. Statistical: Z-score on rolling statistics
    2. IQR: Interquartile range on rolling window
    3. Isolation Forest: Multivariate anomaly detection
    4. STL Residuals: Anomalies in decomposition residuals
    """
    anomaly_scores = pd.DataFrame(index=ts.index)

    # Method 1: Rolling Z-score (modified)
    if 'statistical' in methods:
        rolling_median = ts.rolling(window, center=True, min_periods=1).median()
        rolling_mad = ts.rolling(window, center=True, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        modified_z = 0.6745 * (ts - rolling_median) / (rolling_mad + 1e-8)
        anomaly_scores['statistical'] = (np.abs(modified_z) > z_threshold).astype(int)
        print(f"Statistical anomalies: {anomaly_scores['statistical'].sum()}")

    # Method 2: IQR on rolling window
    if 'iqr' in methods:
        rolling_q1 = ts.rolling(window, min_periods=1).quantile(0.25)
        rolling_q3 = ts.rolling(window, min_periods=1).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        lower = rolling_q1 - 1.5 * rolling_iqr
        upper = rolling_q3 + 1.5 * rolling_iqr
        anomaly_scores['iqr'] = ((ts < lower) | (ts > upper)).astype(int)
        print(f"IQR anomalies: {anomaly_scores['iqr'].sum()}")

    # Method 3: Isolation Forest with lag features
    if 'isolation_forest' in methods:
        features = pd.DataFrame({
            'value': ts,
            'lag_1': ts.shift(1),
            'lag_7': ts.shift(7),
            'rolling_mean': ts.rolling(7).mean(),
            'rolling_std': ts.rolling(7).std()
        }).dropna()

        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        predictions = iso_forest.fit_predict(features)
        iso_anomalies = pd.Series(predictions == -1, index=features.index).astype(int)
        anomaly_scores['isolation_forest'] = iso_anomalies.reindex(ts.index, fill_value=0)
        print(f"Isolation Forest anomalies: {anomaly_scores['isolation_forest'].sum()}")

    # Method 4: STL Residual Analysis
    if 'stl' in methods and len(ts) >= 2 * 365:
        from statsmodels.tsa.seasonal import STL
        stl = STL(ts, period=365, robust=True)
        stl_result = stl.fit()
        residuals = stl_result.resid

        residual_z = np.abs(residuals - residuals.mean()) / residuals.std()
        anomaly_scores['stl'] = (residual_z > 3).astype(int)

    # Ensemble: majority vote
    anomaly_scores['votes'] = anomaly_scores.sum(axis=1)
    n_methods = len([m for m in methods if m in anomaly_scores.columns])
    anomaly_scores['is_anomaly'] = (anomaly_scores['votes'] >= max(1, n_methods // 2)).astype(int)

    print(f"\nEnsemble anomalies: {anomaly_scores['is_anomaly'].sum()}")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].plot(ts, color='blue', alpha=0.7, label='Time Series')
    anomaly_points = ts[anomaly_scores['is_anomaly'] == 1]
    axes[0].scatter(anomaly_points.index, anomaly_points.values,
                    color='red', s=100, zorder=5, label='Anomaly')
    axes[0].legend()
    axes[0].set_title('Time Series with Detected Anomalies')

    axes[1].bar(anomaly_scores.index, anomaly_scores['votes'],
                color='red', alpha=0.5)
    axes[1].set_title('Anomaly Votes (Ensemble Agreement)')
    axes[1].set_ylabel('Number of methods flagging anomaly')
    plt.tight_layout()
    plt.show()

    return anomaly_scores
```

---

## 13. Changepoint Detection

Changepoints are abrupt shifts in the statistical properties of a time series.

```python
# pip install ruptures
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

def detect_changepoints(ts, methods=['pelt', 'binseg', 'window']):
    """
    Multiple changepoint detection methods.

    Methods:
    - PELT: Pruned Exact Linear Time (fast, optimal)
    - BinSeg: Binary Segmentation (greedy, fast)
    - Window: Sliding window approach
    - BOCPD: Bayesian Online CPD (sequential, see below)
    """
    signal = ts.values.reshape(-1, 1)
    results = {}

    # PELT with RBF cost function
    if 'pelt' in methods:
        algo_pelt = rpt.Pelt(model='rbf', min_size=30, jump=5)
        breakpoints = algo_pelt.fit_predict(signal, pen=10)
        results['pelt'] = breakpoints
        print(f"PELT breakpoints: {breakpoints}")

    # Binary Segmentation
    if 'binseg' in methods:
        algo_binseg = rpt.Binseg(model='l2', min_size=30, jump=5)
        breakpoints = algo_binseg.fit_predict(signal, n_bkps=5)
        results['binseg'] = breakpoints
        print(f"BinSeg breakpoints: {breakpoints}")

    # Dynamic Programming (exact, slower)
    if 'dynp' in methods:
        algo_dynp = rpt.Dynp(model='l2', min_size=30, jump=5)
        breakpoints = algo_dynp.fit_predict(signal, n_bkps=3)
        results['dynp'] = breakpoints

    # Visualize changepoints
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for ax, (method, bkps) in zip(axes, results.items()):
        ax.plot(ts.values, color='blue', alpha=0.7)
        for bkp in bkps[:-1]:  # Last element is length of signal
            ax.axvline(x=bkp, color='red', linestyle='--', linewidth=2,
                       label='Changepoint' if bkp == bkps[0] else '')
        ax.set_title(f'Changepoints: {method.upper()}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    return results


# Bayesian Online Changepoint Detection (BOCPD)
class BOCPD:
    """
    Bayesian Online Changepoint Detection.
    Computes P(changepoint at t | data) sequentially.
    """
    def __init__(self, hazard=1/200, mu=0, kappa=1, alpha=1, beta=1):
        self.hazard = hazard  # Prior probability of changepoint at each step
        self.mu0 = mu
        self.kappa0 = kappa
        self.alpha0 = alpha
        self.beta0 = beta

    def update(self, data):
        """Process time series and return changepoint probability at each t."""
        T = len(data)
        run_lengths = np.zeros((T + 1, T + 1))
        run_lengths[0, 0] = 1.0  # Prior: run length 0 at t=0

        changepoint_probs = np.zeros(T)

        # Sufficient statistics
        mu = np.zeros(T + 1)
        kappa = np.zeros(T + 1)
        alpha = np.zeros(T + 1)
        beta = np.zeros(T + 1)

        mu[0] = self.mu0
        kappa[0] = self.kappa0
        alpha[0] = self.alpha0
        beta[0] = self.beta0

        for t in range(1, T + 1):
            x = data[t - 1]

            # Predictive probability under Student-t distribution
            pred_prob = np.zeros(t)
            for r in range(t):
                nu = 2 * alpha[r]
                scale = np.sqrt(beta[r] * (kappa[r] + 1) / (alpha[r] * kappa[r]))
                from scipy.stats import t as t_dist
                pred_prob[r] = t_dist.pdf(x, df=nu, loc=mu[r], scale=scale)

            # Update run length posterior
            run_lengths[t, 1:t + 1] = run_lengths[t - 1, :t] * pred_prob * (1 - self.hazard)
            run_lengths[t, 0] = np.sum(run_lengths[t - 1, :t] * pred_prob * self.hazard)
            run_lengths[t] /= run_lengths[t].sum()  # Normalize

            # Update hyperparameters
            for r in range(t):
                kappa_new = kappa[r] + 1
                mu_new = (kappa[r] * mu[r] + x) / kappa_new
                alpha_new = alpha[r] + 0.5
                beta_new = beta[r] + kappa[r] * (x - mu[r])**2 / (2 * kappa_new)
                if r < t - 1:
                    mu[r + 1] = mu_new
                    kappa[r + 1] = kappa_new
                    alpha[r + 1] = alpha_new
                    beta[r + 1] = beta_new

            # Probability of changepoint at this t
            changepoint_probs[t - 1] = run_lengths[t, 0]

        return changepoint_probs
```

---

## 14. Full End-to-End Examples

### 14.1 ARIMA Forecasting (statsmodels)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate synthetic monthly data (e.g., retail sales)
np.random.seed(42)
dates = pd.date_range('2018-01-01', periods=60, freq='ME')
trend = 100 + np.arange(60) * 0.5
seasonal = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
noise = np.random.normal(0, 5, 60)
sales = trend + seasonal + noise
ts = pd.Series(sales, index=dates, name='sales')

# Train/test split
train = ts[:-12]  # Use last 12 months for testing
test = ts[-12:]

# Step 1: Check stationarity
result = adfuller(train)
print(f"ADF p-value: {result[1]:.4f}")

# Step 2: Auto ARIMA
model = auto_arima(train, seasonal=True, m=12, stepwise=True,
                    information_criterion='aic', trace=False,
                    suppress_warnings=True)
print(f"\nBest model: SARIMA{model.order}{model.seasonal_order}")

# Step 3: Forecast
forecasts, conf_int = model.predict(n_periods=12, return_conf_int=True)
forecasts_series = pd.Series(forecasts, index=test.index)

# Step 4: Evaluate
mae = mean_absolute_error(test, forecasts)
rmse = np.sqrt(mean_squared_error(test, forecasts))
mape = np.mean(np.abs((test - forecasts) / test)) * 100
print(f"\nTest MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

# Step 5: Visualize
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='Train', color='blue')
plt.plot(test.index, test, label='Test (Actual)', color='black')
plt.plot(test.index, forecasts_series, label='SARIMA Forecast', color='red')
plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1],
                 alpha=0.3, color='red', label='95% CI')
plt.title('SARIMA Forecast - Monthly Sales')
plt.legend()
plt.tight_layout()
plt.show()
```

### 14.2 Prophet Forecasting

```python
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import numpy as np

# Prepare data
prophet_df = pd.DataFrame({'ds': dates, 'y': sales})

# Add holiday effects
from prophet.make_holidays import make_holidays_df
holidays = make_holidays_df(year_list=[2018, 2019, 2020, 2021, 2022], country='US')

model = Prophet(
    holidays=holidays,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    yearly_seasonality=True,
    weekly_seasonality=False,  # Monthly data
    daily_seasonality=False
)
model.add_seasonality('monthly', period=30.5, fourier_order=5)
model.fit(prophet_df[:-12])

future = model.make_future_dataframe(periods=12, freq='ME')
forecast = model.predict(future)

# Compare with test
actual = prophet_df[-12:].set_index('ds')['y']
pred = forecast[-12:].set_index('ds')['yhat']
mae = np.mean(np.abs(actual - pred))
print(f"Prophet MAE: {mae:.2f}")

fig = model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

fig = model.plot_components(forecast)
plt.show()
```

### 14.3 LightGBM Time Series Forecasting

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

# Create features
df = pd.DataFrame({'date': dates, 'sales': sales})
df = df.sort_values('date')

# Feature engineering
for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df['sales'].shift(lag)
for w in [3, 6, 12]:
    df[f'rolling_mean_{w}'] = df['sales'].shift(1).rolling(w).mean()
    df[f'rolling_std_{w}'] = df['sales'].shift(1).rolling(w).std()

df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df = df.dropna()

feature_cols = [c for c in df.columns if c not in ['date', 'sales']]
X = df[feature_cols]
y = df['sales']

# Walk-forward CV
tscv = TimeSeriesSplit(n_splits=5, test_size=6)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=5, random_state=42, verbose=-1
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
               eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
               callbacks=[lgb.early_stopping(50, verbose=False)])

    pred = model.predict(X.iloc[val_idx])
    mae = np.mean(np.abs(y.iloc[val_idx] - pred))
    cv_scores.append(mae)
    print(f"Fold {fold+1} MAE: {mae:.2f}")

print(f"\nAvg MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

### 14.4 PyTorch LSTM Forecasting

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Prepare data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()

train_data = scaled[:-12]
test_data = scaled[-12:]

SEQ_LEN = 12  # Use 12 months to predict next month

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return (torch.FloatTensor(np.array(X)).unsqueeze(-1),
            torch.FloatTensor(np.array(y)))

X_train, y_train = make_sequences(train_data, SEQ_LEN)
X_test, y_test = make_sequences(np.concatenate([train_data[-SEQ_LEN:], test_data]), SEQ_LEN)

# Model
model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred.squeeze(), y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_pred = model(X_test).squeeze().numpy()
    test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    test_actual = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()

mae = np.mean(np.abs(test_actual - test_pred))
rmse = np.sqrt(np.mean((test_actual - test_pred)**2))
print(f"\nLSTM Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(14, 5))
plt.plot(dates[-12:], test_actual, 'b-o', label='Actual')
plt.plot(dates[-12:], test_pred, 'r-o', label='LSTM Forecast')
plt.title('LSTM Time Series Forecast')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Quick Reference: Model Selection Guide

| Data Characteristics | Recommended Model |
|---------------------|------------------|
| Univariate, no trend/seasonality | SES, ARMA |
| Univariate, trend only | Holt's Linear, ARIMA(p,1,q) |
| Univariate, trend + seasonality | Holt-Winters, SARIMA |
| Multiple seasonalities | TBATS, Prophet, MSTL+ETS |
| Many related series | LightGBM, XGBoost with lag features |
| Multiple interacting series | VAR |
| Complex patterns, enough data (>1000 pts) | LSTM, TCN |
| Interpretability needed | Prophet (has components), SARIMA |
| High accuracy, tabular features | LightGBM/XGBoost |
| State-of-the-art, large data | TFT, PatchTST, N-HiTS |
| Bounded growth (market saturation) | Prophet with logistic growth |
| Online/streaming | Kalman Filter, BOCPD |

## Cross-Validation Strategy Summary

| Method | Use When |
|--------|----------|
| Walk-Forward (Expanding) | Default choice; good for non-stationary series |
| Walk-Forward (Sliding) | Data distribution shifts over time |
| Blocked CV | Series is stationary; max data usage |
| Prophet cross_validation | Built-in, handles multiple cutoff dates |
| sklearn TimeSeriesSplit | For sklearn-compatible ML models |

---

*This guide covers classical statistical methods through state-of-the-art deep learning for time series. See companion guides for MLOps deployment of time series models.*

---

## References

- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.
- Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale*. The American Statistician (Prophet).
- Oreshkin, B. N., et al. (2019). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*. ICLR.
- Zhou, H., et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*. AAAI.
- Nie, Y., et al. (2022). *Time Series is a Special Sequence: Forecasting with Sample Convolution and Interaction*. NeurIPS (PatchTST).
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
