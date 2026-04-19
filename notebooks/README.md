# GLOBE Curation AI: Execution Notebooks

This directory contains the core computational pipeline for the automated curation of the GLOBE water transparency dataset. The notebooks are structured sequentially to allow researchers and peer reviewers to reproduce the Bayesian Hierarchical Model (BHM) framework—from raw data ingestion to the final manuscript figures—relying exclusively on publicly hosted NASA data.

## Execution Sequence

To perfectly reproduce the findings and figures in the manuscript, please execute the notebooks in the following order:

### `00_quickstart_demo.ipynb`
* **Purpose:** A lightweight, rapid-deployment demonstration of the curation framework. It processes a 10-year subset of the data to verify your local Python environment and visually demonstrates the pipeline's ability to isolate outliers using high-contrast KDE and Rug plots, without requiring the time to process the full historical registry.
* **Expected Execution Time:** < 1 minute

### `01_curation_pipeline.ipynb`
* **Purpose:** The primary data engine. This notebook ingests the raw, publicly available NASA GLOBE hydrology measurements and merges them with the official GLOBE site metadata to construct a normalized, 4-table relational SQLite database. It applies strict heuristic gates to isolate right-censored (clear water) readings, and then executes the PyMC MCMC samplers to isolate systemic noise from natural environmental drift.
* **Expected Execution Time:** 30 - 45 minutes (Hardware dependent; driven by the 4-chain NUTS sampling process across thousands of sites).

> Depending on your environment, the progress meter may remain stagnant at 0% for the entire duration of the execution while the sampler runs in the background. The UI typically jumps directly to 100% only upon completion. **Please allow the process to finish and avoid terminating the session prematurely.**

### `02_model_diagnostics.ipynb`
* **Purpose:** Mathematical validation of the Bayesian framework. This notebook evaluates the health and geometry of the PyMC models. It generates Trace Plots, Divergence Analysis, Gelman-Rubin convergence statistics, and explicitly visualizes **Hierarchical Shrinkage** (Partial Pooling). It also contrasts empirical hardware limits (e.g., the 120cm transparency tube ceiling) against Posterior Predictive Checks (PPCs).
* **Expected Execution Time:** 2 - 5 minutes

### `03_exploratory_analysis.ipynb`
* **Purpose:** Supplemental statistical analysis and the primary visualization suite for the manuscript. This notebook reads directly from the curated SQLite database to produce the final publication figures, including:
  * Curation Cascade proportions (Donut Charts)
  * Pre vs. Post-Curation Density Distributions
  * Global Spatial Footprints (High-resolution Cartopy mapping)
  * Seasonal Distributions over time
  * "Representative Site" Time-Series anomaly isolation
* **Expected Execution Time:** ~5 minutes

## Environment Setup

**Please refer to the [main README.md](../README.md) at the root of this repository for comprehensive installation instructions.**

To guarantee perfect parity with the manuscript's exact system libraries (specifically required for `cartopy` and `netCDF4`), we strongly recommend using the Docker container as described in the primary documentation.

*Note: All setup commands (like `docker build` or `pip install`) and Jupyter server executions should be run from the **project root folder**, not from this `/notebooks` directory.