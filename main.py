import os
import warnings

import pandas as pd
import yaml

from src.database.manager import DatabaseManager
from src.curation.disk_validator import validate_secchi_data
from src.curation.tube_validator import validate_tube_data
from src.curation.disk_model import BayesianDiskModel
from src.curation.tube_model import BayesianTubeModel
from src.curation.tube_length_estimator import apply_all_tube_estimations
from src.curation.distance_to_water import compute_water_distances

warnings.filterwarnings("ignore", category=FutureWarning, module="arviz")


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Loads pipeline configuration parameters."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def print_summary(instrument_name: str, df: pd.DataFrame) -> None:
    """Calculates and prints a standardized, mutually exclusive summary of the curation results."""
    total = len(df)

    passed_h = df['passed_heuristics'].astype(bool)
    is_censored = df.get('is_censored', pd.Series(False, index=df.index)).astype(bool)

    # Ternary outlier status: True = outlier, False = not outlier, None = censored (not evaluated)
    outlier_col = df.get('is_statistical_outlier', pd.Series(None, index=df.index))
    is_outlier = outlier_col.eq(True)  # Only True matches, None and False do not

    # Calculate mutually exclusive subsets so they sum exactly to the total
    heuristic_errors = (~passed_h).sum()
    censored_in_model = (passed_h & is_censored).sum()
    outliers = (passed_h & is_outlier).sum()
    valid = (passed_h & ~is_outlier & ~is_censored).sum()

    print(f"{instrument_name.upper()} SUMMARY:")
    print(f"  Total Observations: {total}")
    print(f"  Valid Samples:      {valid}")
    print(f"  Right-Censored:     {censored_in_model} (Lower bounds, included in model)")
    print(f"  Bayesian Outliers:  {outliers} (Statistical anomalies)")
    print(f"  Heuristic Errors:   {heuristic_errors} (Typos, bounds, contradictions)")
    print("-" * 50)

    # Sanity check for the logs
    if (valid + censored_in_model + outliers + heuristic_errors) != total:
        print(
            f"  [WARNING]: Summary buckets "
            f"({valid + censored_in_model + outliers + heuristic_errors}) "
            f"do not equal total observations ({total}). Check boolean logic."
        )


def main():
    """
    Executes the primary data curation pipeline.
    Orchestrates normalized data ingestion, instrument routing, and Bayesian evaluation.
    """
    print("Starting Bayesian Hierarchical Curation Pipeline...")

    config = load_config()
    db_path = config.get("paths", {}).get("database", "data/master_registry.sqlite")
    flagged_dir = config.get("paths", {}).get("flagged_output", "data/flagged/")
    window_years = config.get("drift", {}).get("window_years", 10)

    db_manager = DatabaseManager(db_path=db_path, overwrite=True)

    print("Initializing normalized data ingestion...")
    df_merged = db_manager.load_and_merge_data(config)
    print(f"Total merged records available for processing: {len(df_merged)}")

    if config.get('distance_to_water', {}).get('enabled', False):
        print("\nEnriching observations with water distance metadata...")
        df_merged = compute_water_distances(df_merged, config)

    print("Routing data streams by instrument type...")
    df_disk, df_tube = db_manager.split_by_instrument(df_merged)

    curated_disk = None
    curated_tube = None

    if not df_disk.empty:
        print(f"\n--- Processing {len(df_disk)} Secchi Disk records ---")
        validated_disk = validate_secchi_data(df_disk, config)

        print("  -> Initializing Bayesian inference model...")
        disk_model = BayesianDiskModel(config)
        curated_disk = disk_model.evaluate(validated_disk, window_years=window_years)

        # Maintain naming symmetry between disk and tube tables.
        db_manager.export_curated_data(curated_disk, "measurements_disk")
        db_manager.export_audit_log(curated_disk, os.path.join(flagged_dir, "flagged_disk.csv"))

    if not df_tube.empty:
        print(f"\n--- Processing {len(df_tube)} Transparency Tube records ---")
        print("  -> Applying tube length estimations...")
        df_tube = apply_all_tube_estimations(df_tube)

        validated_tube = validate_tube_data(df_tube, config)

        print("  -> Initializing Bayesian inference model...")
        tube_model = BayesianTubeModel(config)
        curated_tube = tube_model.evaluate(validated_tube, window_years=window_years)

        db_manager.export_curated_data(curated_tube, "measurements_tube")
        db_manager.export_audit_log(curated_tube, os.path.join(flagged_dir, "flagged_tube.csv"))

    print("\n" + "=" * 50)
    print("             PIPELINE EXECUTION SUMMARY")
    print("=" * 50)

    if curated_disk is not None:
        print_summary("Secchi Disk", curated_disk)

    if curated_tube is not None:
        print_summary("Transparency Tube", curated_tube)

    print("\nPipeline execution complete. Normalized database and audit logs generated.")


if __name__ == "__main__":
    main()