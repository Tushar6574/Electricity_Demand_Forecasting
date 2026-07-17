DuckDB OLAP Initialization
Goal: Setup and maintain the high-performance electricity demand database.

## SOP (Standard Operating Procedure)
1. Database Setup: Connect to `electricity_data.db` using the DuckDB engine.
2. Schema Creation: Ensure a table `raw_demand_data` exists with schema: 
   - `timestamp` (TIMESTAMP), `actual_demand` (DOUBLE).
3. Optimized Ingestion: Use the `read_csv_auto` function for vectorized loading of raw datasets.
4. Validation: Execute a `COUNT(*)` check to confirm data integrity post-load.
5. Vectorized View: Create a view `v_feature_pipeline` that calculates 24h lag and rolling averages using SQL window functions.
