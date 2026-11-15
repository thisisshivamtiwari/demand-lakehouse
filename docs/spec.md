# Spec (written before the build, 15 Nov 2026)

**Problem.** Pricing and supply teams in travel and mobility forecast demand for thousands of location-day series
(zones, properties, routes). A notebook on a laptop handles one series; the hard parts are the pipeline around the
model: messy raw files whose schema drifts month to month, data-quality rules that must stop bad data reaching the
model, a hierarchy (zone, borough, city) whose forecasts should agree, and a backtest that never peeks at the future.

**Data.** NYC TLC yellow-taxi trip records, Oct 2025 to Sep 2026 (12 monthly parquet files, public), plus the TLC
taxi-zone lookup. Demand = pickups per zone per day. 263 zones, 6 boroughs, 1 city.

**Build.**
1. Medallion pipeline on Spark + Delta Lake, run locally:
   - bronze: raw files to one Delta table with a canonical schema (casts recorded as schema drift), idempotent by source month (replaceWhere);
   - silver: data-quality rules as expectations (drop to a quarantine table, fail the run above a threshold);
   - gold: zone-day demand with zero-filled gaps, plus borough-day and city-day.
2. Forecasting on Spark: seasonal-naive and 4-week seasonal-mean baselines; a global gradient-boosted model
   (Spark MLlib GBTRegressor, one model for all zones); a per-zone model trained with `applyInPandas`
   (the "many models" pattern). Rolling-origin backtest: weekly origins over the last 12 weeks, 14-day horizon.
3. Hierarchy: bottom-up sums vs direct forecasts at borough and city level.
4. Orchestration: a small job runner reading `jobs.yaml` (task graph, retries, run log), shaped like a Databricks job.
5. Scaling: pipeline wall time at 1, 2, 4 and 8 cores and with different shuffle-partition counts.

**Measures.** WAPE and MASE by level against seasonal-naive; data-quality drop rates by rule; pipeline runtime by
stage and by core count.

**Gate (CI).** Lint, tests (pipeline on a small real-data fixture), and a replay of the backtest from the committed gold
snapshot that must reproduce results/backtest.json and must beat seasonal-naive at the zone level.
