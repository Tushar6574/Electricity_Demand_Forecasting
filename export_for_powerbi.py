"""
Power BI Data Export Script
============================
Exports analysis-ready CSV files from DuckDB databases for Power BI ingestion.

Outputs:
  1. powerbi_demand_history.csv   - Full demand history with weather + derived features
  2. powerbi_model_performance.csv - Model metrics comparison (XGBoost vs TimesFM)
  3. powerbi_predictions.csv       - Predictions vs actuals with error analysis
"""

import duckdb
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE, "data_exports")
ELEC_DB = os.path.join(BASE, "electricity_data.db")
REPORTS_DB = os.path.join(BASE, "model_reports.db")

os.makedirs(EXPORT_DIR, exist_ok=True)


def export_demand_history():
    """Export full demand history with derived columns for Power BI date hierarchies."""
    con = duckdb.connect(ELEC_DB, read_only=True)

    df = con.execute("""
        SELECT
            timestamp,
            CAST(timestamp AS DATE)                       AS date,
            hour,
            dayofweek,
            CASE
                WHEN dayofweek IN (0, 6) THEN 'Weekend'
                ELSE 'Weekday'
            END                                            AS day_type,
            CASE
                WHEN dayofweek = 0 THEN 'Monday'
                WHEN dayofweek = 1 THEN 'Tuesday'
                WHEN dayofweek = 2 THEN 'Wednesday'
                WHEN dayofweek = 3 THEN 'Thursday'
                WHEN dayofweek = 4 THEN 'Friday'
                WHEN dayofweek = 5 THEN 'Saturday'
                WHEN dayofweek = 6 THEN 'Sunday'
            END                                            AS day_name,
            month,
            CASE
                WHEN month IN (12, 1, 2)  THEN 'Winter'
                WHEN month IN (3, 4, 5)   THEN 'Spring'
                WHEN month IN (6, 7, 8)   THEN 'Summer'
                WHEN month IN (9, 10, 11) THEN 'Autumn'
            END                                            AS season,
            year,
            dayofyear,
            temperature,
            humidity,
            demand
        FROM demand_records
        ORDER BY timestamp
    """).fetchdf()

    con.close()

    out = os.path.join(EXPORT_DIR, "powerbi_demand_history.csv")
    df.to_csv(out, index=False)
    print(f"[1/3] powerbi_demand_history.csv  ->  {len(df):,} rows")
    return len(df)


def export_model_performance():
    """Export model comparison metrics."""
    con = duckdb.connect(REPORTS_DB, read_only=True)

    df = con.execute("""
        SELECT
            model_name,
            rmse,
            mae,
            created_at
        FROM model_results
        ORDER BY created_at
    """).fetchdf()

    con.close()

    out = os.path.join(EXPORT_DIR, "powerbi_model_performance.csv")
    df.to_csv(out, index=False)
    print(f"[2/3] powerbi_model_performance.csv  ->  {len(df)} rows")
    return len(df)


def export_predictions():
    """Export prediction-level detail with error percentages."""
    con = duckdb.connect(REPORTS_DB, read_only=True)

    df = con.execute("""
        SELECT
            p.timestamp,
            CAST(p.timestamp AS DATE)   AS date,
            p.actual,
            p.predicted,
            p.error,
            p.mape                       AS mape_pct,
            m.model_name,
            m.rmse,
            m.mae
        FROM predictions p
        CROSS JOIN model_results m
        WHERE m.model_name = 'XGBoost'
        ORDER BY p.timestamp
    """).fetchdf()

    con.close()

    out = os.path.join(EXPORT_DIR, "powerbi_predictions.csv")
    df.to_csv(out, index=False)
    print(f"[3/3] powerbi_predictions.csv  ->  {len(df):,} rows")
    return len(df)


if __name__ == "__main__":
    print("=" * 55)
    print("  Power BI Data Export")
    print("=" * 55)

    if not os.path.exists(ELEC_DB):
        print(f"ERROR: {ELEC_DB} not found.")
        sys.exit(1)
    if not os.path.exists(REPORTS_DB):
        print(f"ERROR: {REPORTS_DB} not found.")
        sys.exit(1)

    n1 = export_demand_history()
    n2 = export_model_performance()
    n3 = export_predictions()

    print("\n" + "=" * 55)
    print(f"  Export complete! {n1 + n2 + n3:,} total rows")
    print(f"  Output: {EXPORT_DIR}")
    print("=" * 55)
