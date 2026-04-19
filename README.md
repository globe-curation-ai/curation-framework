# GLOBE Curation AI: Automated Framework for Continuous Bayesian Curation

This repository hosts a Python-based curation pipeline for harmonizing dispersed citizen science water quality data through artificial intelligence. Designed specifically for the GLOBE (Global Learning and Observations to Benefit the Environment) Program, this framework curates three decades of water transparency data collected via Secchi Disks and Transparency Tubes. 

To address the inherent variability in citizen science data and the dynamic nature of environmental systems, this pipeline utilizes a Bayesian Hierarchical Model (BHM) as a probabilistic validation layer.

## The Recursive Curation Cycle

The framework operates as a continuous, automated loop consisting of six distinct stages of data processing and parameter refinement:

1. **Ingest & Expand:** Raw data is associated with multi-source site metadata.
2. **Validate Samples:** Incoming observations are evaluated via a Posterior Predictive Check (PPC) against site-specific baselines.
3. **Merge to Registry:** All observations are merged with the Master Registry. Statistical outliers are flagged.
4. **Update Master Registry:** The normalized site metadata and time-series measurements are persistently stored in an SQLite database.
5. **Audit & Review:** Copies of flagged statistical outliers are exported to flat CSV files for extreme-event verification by human reviewers, while remaining in the registry.
6. **Retrain Model & Publish:** The BHM is recalibrated using the curated observations in the Master Registry (excluding the flagged statistical outliers), and the authoritative dataset is published.

## Handling Environmental Concept Drift

Water transparency is subject to environmental changes across multiple timescales. A static validation model trained on data from 2010 might incorrectly flag valid 2026 measurements as outliers due to "concept drift" driven by seasonal phenology or stochastic events.

To mitigate this, the framework employs a continuous Temporal Weighting Mechanism that prioritizes recent observations while maintaining a decade-long historical baseline. The weight of an observation is determined by its age:

$$w(t) = \max\left(0, 1 - \frac{t}{10}\right)$$

where t is the continuous sample age in years. The statistical influence of an observation decays linearly, reaching zero exactly 10 years from the dynamic anchor date.

## Project Structure

    curation-framework/
    │
    ├── main.py                        # The master orchestrator script
    ├── Dockerfile                     # Containerized environment
    ├── README.md                      # Project documentation
    ├── requirements.txt               # Python dependencies
    │
    ├── config/
    │   └── settings.yaml              # Pipeline configuration and column mappings
    │
    ├── data/
    │   ├── observations/              # GLOBE measurement CSVs will be downloaded here
    │   ├── site_info/                 # GLOBE site info CSV will be downloaded here
    │   ├── master_registry.sqlite     # The persistent SQLite Master Registry
    │   └── flagged/                   # Audit logs for flagged anomalies
    │
    ├── output/                        # Pipeline generated artifacts
    │   ├── traces/                    # Saved PyMC NetCDF traces
    │   ├── audit/                     # Final diagnostic CSV summaries
    │   └── figures/                   # Plots and spatial maps
    │
    ├── notebooks/
    │   ├── README.md                  # Detailed execution guide
    │   ├── 00_quickstart_demo.ipynb   # Interactive pipeline testing
    │   ├── 01_curation_pipeline.ipynb # Primary data engine
    │   ├── 02_model_diagnostics.ipynb # MCMC validation and PPCs
    │   └── 03_exploratory_analysis.ipynb # Visualization and manuscript figures
    │
    ├── src/
    │   ├── database/
    │   │   └── manager.py             # Heuristic routing and SQLite ingestion
    │   │
    │   └── curation/
    │       ├── disk_validator.py      # Secchi Disk heuristic rules
    │       ├── tube_validator.py      # Transparency Tube heuristic rules
    │       ├── disk_model.py          # PyMC Log-Normal BHM with Temporal Weighting
    │       └── tube_model.py          # PyMC Truncated-Normal BHM with Temporal Weighting
    └── tests/
        ├── test_bayesian_models.py    # Mathematical and MCMC validation
        ├── test_data_ingestion.py     # Data ingestion and validation tests
        └── test_integrity.py          # Relational database and SQLite schema checks

## Quick Start

To ensure the scientific integrity and reproducibility of the recursive curation cycle, we provide both an interactive demonstration and a full pipeline execution script.

### 1. Environment Setup (Recommended: Docker)
Because geospatial mapping (`cartopy`) and Bayesian trace storage (`netCDF4`) rely on complex system-level C-libraries, we strongly recommend using the provided Docker container to guarantee execution on any machine.

    docker build -t globe-curation .
    docker run -p 8888:8888 -v $(pwd):/app globe-curation

*(For local installation without Docker: `pip install -r requirements.txt`)*

### 2. Interactive Demonstration
To instantly verify the framework's ingestion, temporal joining, and heuristic curation capabilities without waiting for the full Bayesian model to sample, run the quickstart notebook.

If using the Docker container, the Jupyter Lab server is already running. Access it securely at:
`http://localhost:8888/lab?token=globe2026`

If running locally without Docker, launch the notebook server manually:

    jupyter notebook notebooks/00_quickstart_demo.ipynb
   
*This self-contained demo automatically fetches the last 10 years of GLOBE data, builds a local SQLite registry, and generates before-and-after density plots of the curated output.*

### 3. Configure Settings (For Full Pipeline)
Review `config/settings.yaml` to adjust the number of Bayesian draws and tuning iterations for testing versus production runs.

### 4. Run the Full Pipeline
Manual data downloads are not required. Execute the master recursive curation framework:

    python main.py
   
*Note: The first execution acts as a "Bootstrap" run to build the initial Master Registry, generate the normalized SQLite tables, and establish the first set of site-specific Bayesian boundaries.*

## License
All code is released under the MIT License, facilitating open modification and integration into broader scientific monitoring and decision-support systems.