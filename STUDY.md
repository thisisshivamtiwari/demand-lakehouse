# STUDY: demand-lakehouse

Read this before an interview where this project might come up. All numbers are from `results/*.json` (runs on
15 Nov 2026 on an Apple M2 Mac with 8 cores and 8 GB, Spark 4.2.0 and Delta Lake 4.4.1 in local mode, Java 21; NYC TLC
yellow-taxi records Oct 2025 to Sep 2026).

## 1. The problem in one paragraph
Pricing and supply teams in travel and mobility forecast demand for thousands of location-day series. The model is
the small part. The large part is the pipeline: monthly raw files whose schema drifts, bad rows that must not reach
the model, a hierarchy (zone, borough, city) whose forecasts should agree, and a backtest that never uses the future.
A forecast built on a broken month or a leaky backtest looks good in a notebook and fails in production, and the
price or fleet plan built on it fails with it.

## 2. Design choices (and why)
- **Public taxi data as the stand-in.** 46.5 million real trips, a natural hierarchy (263 zones in 6 boroughs), real
  data problems, and a known strong baseline (weekly seasonality). It is a proxy for travel demand, not bookings.
- **Medallion layers on Delta Lake.** Bronze keeps raw rows in a canonical schema; silver is what passed the rules;
  gold is the modelling table. Each layer can be rebuilt from the one before. Delta gives ACID writes and
  `replaceWhere`, so re-running one month replaces exactly that month.
- **Read each monthly file on its own.** The files disagree on types (integers arriving as bigint) and columns
  (`request_source` appears from June 2026). Reading them together fails or silently widens types; reading each and
  mapping it to a canonical schema makes the drift visible and logged.
- **Expectations with three actions.** Drop (quarantine the row), fail (stop the run if a month's failure rate is
  above the threshold) and warn (count only). Thresholds apply per month so one broken file cannot hide in a year
  of good ones. Thresholds were written before the first run.
- **Zero-fill gold.** A zone with no pickups has demand 0. Without zero-fill, "lag 7 rows" is not "7 days ago".
- **Rolling-origin backtest, features only from the past.** A row is (series, origin, horizon). Same-weekday lags
  are taken at or before the origin (for horizons 8 to 14 that means two weeks back). A unit test builds a series
  where y equals the day number, so every feature names the day it came from.
- **Four methods, from trivial to global.** Seasonal-naive and a 4-week seasonal mean (baselines a forecaster must
  beat), a per-zone ridge regression trained with `applyInPandas` (the many-models pattern used for thousands of
  series on Databricks), and one global MLlib GBT for all zones (shares strength across series).
- **A level-relative target for the global model.** Trees cannot extrapolate. With a plain log target a single model
  has to learn every zone's level; with log(demand / 28-day level) it learns shape, which transfers across zones.
- **Validation chooses, test scores once.** The GBT target (plain log vs relative) was chosen on a validation window
  (origins 1 Apr to 17 Jun 2026) whose actuals stop where the test window starts (origins 1 Jul to 16 Sep 2026).
- **WAPE and MASE.** WAPE is total absolute error over total demand, so big zones dominate, which matches money.
  MASE scales each series by its own naive error, so small zones count equally. Reporting both shows who a model
  helps.
- **Deterministic Spark ML for the CI gate.** MLlib samples rows to pick tree split candidates, and the sample depends
  on partitioning. Training data is hash-partitioned into a fixed 8 partitions and sorted, so a 2-core GitHub runner
  reproduces the numbers from an 8-core laptop.
- **A Databricks-shaped job file.** `jobs.yaml` uses task_key, depends_on and max_retries, and a 60-line runner
  executes it with retries and a run log. It shows the orchestration idea without pretending to be Airflow.

## 3. Results
See the README tables (generated from `results/*.json`). The points to remember:
- Silver gate: the first amounts rule stopped Oct and Nov 2025 (Nov: 9.4% of rows had a negative fare). Card, cash,
  no-charge and dispute negatives were mostly exact mirrors of a positive trip (reversals: dropped). Flex-fare
  (payment type 0) negatives had no mirror (0 of 346,979 in Nov) and a median trip of 2.8 miles (real trips with a
  bad amount: kept and counted). Dropping them would have removed 8.3% of November's trips.
- Validation, zone level: GBT relative 11.4% WAPE vs plain log 14.6%; the 4-week seasonal mean was best (10.9%).
- Test, zone level: GBT relative 14.4% WAPE vs seasonal-naive 14.6%; MASE 0.81 vs 0.91.
- Borough and city: a direct GBT is worse than summing zone GBT forecasts (borough 13.7% vs 11.9% WAPE); the per-series
  ridge is best at the top levels (city 10.7% direct, 10.4% bottom-up).
- Full job run on the laptop (`results/run.json`): bronze about 112 s for 46.5M rows, silver about 106 s, gold 6 s, backtest about 2.3 min.

## 4. What I learned
1. **The data-quality gate earned its place on the first real run.** A rule I thought was obvious ("amounts are not
   negative") was wrong for one payment type, and only failing the run made me look.
2. **A strong seasonal baseline is hard to beat on volume.** Weekly seasonality explains most of taxi demand. The
   model's value showed up on MASE (small zones), not WAPE.
3. **Rankings move between windows.** The seasonal mean won validation; the GBT won test. With a gap of about one point,
   the honest claim is "comparable, with better small-zone accuracy", not "the model is better".
4. **Global tree models need scale-free targets.** Switching to a relative target cut validation WAPE from 14.6% to 11.4%.
5. **Bottom-up beats direct for tree models at the top of a hierarchy**, because a model of 1 or 6 series has almost
   nothing to learn from.

## 5. Failure modes I saw
- A month-level anomaly (flex-fare negative amounts) that a year-level rate would have hidden (1.3% over the year,
  8.3% in Nov 2025).
- Silent drift: a new column (`request_source`) from June 2026; integer columns stored as bigint.
- Trees under-forecast when the test level is outside the training range (direct city GBT with a plain log target).
- Holiday weeks: same-weekday lags carry a holiday into the following week's forecast.
- Spark ML results that change with core count, until partitioning was fixed.
- No speed-up from more cores on the laptop: silver and gold on 3 months took 46.6 s on 1 core and 57.1 s on 8.
  Cause not isolated (writes, not the rule checks; memory pressure is the first suspect).
- Spark at module import: building a `Column` needs a live session, so the rules are built in a function.

## 6. What I would do next
Refit at every origin; add weather and event features; MinT reconciliation; quantile or conformal intervals for
pricing risk; MLflow tracking of each run; and the same job on a multi-node cluster to measure shuffle cost.

## 7. Interview questions and answers
1. **Why Spark for 96k gold rows, and why no speed-up from more cores?** It isn't needed for gold. It is needed for bronze and silver (46.5M rows, schema
   mapping, rule checks, a zone-day aggregation), and the forecasting uses the patterns that scale (a global MLlib
   model, `applyInPandas` per series) so the same code runs on a cluster with 100k series. On one 8 GB laptop,
   8 cores were not faster than 1 for silver and gold; I would measure scaling on a real cluster, where executors
   have their own memory and disks, before claiming anything about it.
2. **How do you stop the backtest from leaking the future?** Every feature is computed at the origin, lags are chosen
   so the day used is at or before the origin, the GBT is trained only on labels up to the first test origin, and a
   unit test proves it on a series where each value is its own day number.
3. **Why did you change the data-quality rule after it failed? Isn't that moving the goalposts?** The rule's purpose
   was to remove reversals, which double count. The investigation showed the flex-fare negatives are not reversals:
   0 of 346,979 had a mirror trip. So the rule was wrong about the world. The new rule still drops reversals with the
   same threshold, and the flex-fare rows are still counted as a warning, so the anomaly stays visible.
4. **WAPE or MASE?** WAPE when the cost of error scales with volume (fleet, revenue); MASE to compare across series of
   different sizes. I report both because they disagree here: GBT barely beats naive on WAPE but clearly on MASE.
5. **Why did the global GBT do badly at city level?** One series, a few hundred training rows, and trees cannot
   extrapolate beyond the training range. Summing zone forecasts is better; the per-series ridge is best there.
6. **How did you choose the model variant without overfitting to the test?** On a separate validation window whose
   actuals end where the test window starts. The choice and the rule are in `results/model_selection.json`.
7. **How does this map to Databricks?** Delta tables become Unity Catalog tables; the silver rules become Delta Live
   Tables expectations; `jobs.yaml` becomes a Databricks job; MLlib and `applyInPandas` run unchanged; MLflow would
   track the runs. None of it ran on Databricks; it ran on Spark and Delta locally.
8. **How is the pipeline idempotent?** Bronze writes each month with `replaceWhere`, silver and gold overwrite, and a
   test runs bronze twice and checks the row counts are unchanged.
9. **What makes CI reproduce the numbers?** The gold snapshot is committed; CI replays the backtest on Spark with fixed
   hash partitioning and seed, compares every metric with `results/backtest.json` within 0.002, and fails if the
   model stops beating seasonal-naive on zone MASE.
10. **What would you add for a pricing team?** Prediction intervals (pricing reacts to risk, not just the mean), event
    and weather features, a refit schedule with drift monitoring on the silver rule rates, and a dashboard of forecast
    error by zone so planners know where not to trust it.
