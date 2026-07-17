# ELECTRICITY_SCHEMA_SKILL

## Database Schema — `electricity_data.db`

### Table: `demand_records`

| Column       | Type      | Constraints         |
|-------------|-----------|---------------------|
| timestamp   | TIMESTAMP | PRIMARY KEY         |
| hour        | INTEGER   |                     |
| dayofweek   | INTEGER   |                     |
| month       | INTEGER   |                     |
| year        | INTEGER   |                     |
| dayofyear   | INTEGER   |                     |
| temperature | FLOAT     |                     |
| humidity    | FLOAT     |                     |
| demand      | FLOAT     |                     |

## Operations

### 1. Create table
```sql
CREATE TABLE IF NOT EXISTS demand_records (
    timestamp TIMESTAMP PRIMARY KEY,
    hour INTEGER,
    dayofweek INTEGER,
    month INTEGER,
    year INTEGER,
    dayofyear INTEGER,
    temperature FLOAT,
    humidity FLOAT,
    demand FLOAT
);
```

### 2. Load CSV with timestamp validation
```sql
-- Validate no malformed timestamps before loading
SELECT count(*) FROM read_csv_auto('{CSV_PATH}', strict_mode=false)
WHERE try_cast(
    strptime("Timestamp", '%d-%b-%y') + INTERVAL (CAST(COALESCE(hour, 0) AS INTEGER)) HOUR
    AS TIMESTAMP
) IS NULL;

-- Insert clean data
INSERT INTO demand_records BY NAME
SELECT * FROM read_csv_auto('{CSV_PATH}', strict_mode=false)
WHERE try_cast(
    strptime("Timestamp", '%d-%b-%y') + INTERVAL (CAST(COALESCE(hour, 0) AS INTEGER)) HOUR
    AS TIMESTAMP
) IS NOT NULL;
```

### 3. Feature engineering (window functions)
```sql
SELECT
    timestamp,
    LAG(demand, 24)  OVER (ORDER BY timestamp) AS Demand_lag_24hr,
    LAG(demand, 168) OVER (ORDER BY timestamp) AS demand_lag_168hr,
    AVG(demand)   OVER (ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS demand_rolling_mean_24hr,
    STDDEV(demand)OVER (ORDER BY timestamp ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) AS demand_rolling_std_24hr
FROM demand_records
ORDER BY timestamp;
```

### 4. Connection patterns

**Persistent (writes)** — use for schema setup, data loading:
```python
con = duckdb.connect(r"path\to\electricity_data.db")
```

**In-memory (reads only)** — use for validation queries to avoid file lock:
```python
con = duckdb.connect()
```
