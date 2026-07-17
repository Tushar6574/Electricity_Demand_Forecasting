# Reporting_Skill — Model Performance SOP

## Database: `model_reports.db`

### Table: `model_results`
| Column         | Type    |
|---------------|---------|
| model_name    | TEXT    |
| rmse          | FLOAT   |
| mae           | FLOAT   |
| created_at    | TIMESTAMP (default current_timestamp) |

### Table: `predictions`
| Column         | Type      |
|---------------|-----------|
| timestamp     | TIMESTAMP |
| actual        | FLOAT     |
| predicted     | FLOAT     |
| error         | FLOAT     |
| mape          | FLOAT     |
| created_at    | TIMESTAMP (default current_timestamp) |

## 5-Step Reporting SOP

### Step 1: Source selection
Identify which model version is active (e.g., `electricity_xgb_prediction_model.pkl`).

Insert current model metrics:
```sql
INSERT INTO model_results (model_name, rmse, mae)
VALUES ('XGBoost', 173.84, 123.13);
```

### Step 2: Create `model_performance` view
```sql
CREATE OR REPLACE VIEW model_performance AS
SELECT
    COUNT(*) AS total_predictions,
    AVG(ABS(error)) AS mae,
    AVG(ABS(mape))  AS mape,
    SUM(CASE WHEN actual < predicted THEN 1 ELSE 0 END) AS over_predictions,
    SUM(CASE WHEN actual > predicted THEN 1 ELSE 0 END) AS under_predictions
FROM predictions;
```

### Step 3: Threshold analysis
```sql
SELECT
    CASE
        WHEN actual - predicted > 100 THEN 'High'
        ELSE 'Low'
    END AS demand_category,
    COUNT(*) AS count
FROM predictions
GROUP BY demand_category;
```

### Step 4: Compute MAE and MAPE
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, predictions)
mape = (abs(Y_test - predictions) / Y_test).mean() * 100
```

### Step 5: Operational action plan
- If MAPE < 3%: model is production-ready
- If MAPE 3–5%: consider retraining with more recent data
- If MAPE > 5%: investigate feature drift, data quality issues
