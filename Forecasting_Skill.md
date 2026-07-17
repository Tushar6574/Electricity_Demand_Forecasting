# Forecasting_Skill — XGBoost Demand Forecasting Pipeline

## Pipeline Steps

### 1. Load data from DuckDB
```python
import duckdb
con = duckdb.connect(r"path\to\electricity_data.db")
data = con.execute("""
    SELECT timestamp, hour, dayofweek, month, year, dayofyear,
           temperature AS Temperature, humidity AS Humidity, demand AS Demand
    FROM demand_records ORDER BY timestamp
""").fetchdf()
con.close()
```

### 2. Feature engineering
```python
def create_features(df, db_path):
    con = duckdb.connect(db_path)
    result = con.execute("""
        SELECT timestamp,
            LAG(demand, 24)  OVER (ORDER BY timestamp) AS Demand_lag_24hr,
            LAG(demand, 168) OVER (ORDER BY timestamp) AS demand_lag_168hr,
            AVG(demand)  OVER (ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS demand_rolling_mean_24hr,
            STDDEV(demand) OVER (ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS demand_rolling_std_24hr
        FROM demand_records ORDER BY timestamp
    """).fetchdf()
    con.close()
    return result
```

### 3. Feature columns (exact order)
```python
features = ['hour', 'dayofweek', 'month', 'year', 'dayofyear',
            'weekofyear', 'quarter', 'is_weekend',
            'Temperature', 'Humidity',
            'Demand_lag_24hr', 'demand_lag_168hr',
            'demand_rolling_mean_24hr', 'demand_rolling_std_24hr']
```

### 4. Train/test split
```python
X_train = X.loc[:'2023-12-31']
Y_train = Y.loc[:'2023-12-31']
X_test  = X.loc['2024-01-01':]
Y_test  = Y.loc['2024-01-01':]
```

### 5. XGBoost model
```python
model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3,
                         subsample=0.5, random_state=42, early_stopping_rounds=10)
model_xgb.fit(X_train, Y_train,
              eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=False)

# Save
joblib.dump(model_xgb, 'electricity_xgb_prediction_model.pkl')
```

### 6. Evaluate
```python
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
mae = mean_absolute_error(Y_test, predictions)
mape = (abs(Y_test - predictions) / Y_test).mean() * 100
```

### 7. Feature importance
```python
importance = model_xgb.feature_importances_
for name, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp:.4f}")
```
