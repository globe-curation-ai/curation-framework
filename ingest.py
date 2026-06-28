"""
Incremental Curation Pipeline — Continuous Ingestion Entry Point

Processes new CSV files dropped into the inbox/ directory:
  1. Scans for *.csv files
  2. Generates deterministic sample tracking IDs (usid)
  3. Refreshes site info and rejects samples with unknown site_ids
  4. Enriches with water distance metadata
  5. Applies heuristic validation
  6. Applies Bayesian validation using the saved model trace
  7. Upserts results into the master registry (overwriting amendments)
  8. Exports audit logs
  9. Retrains the Bayesian model on the full curated dataset
  10. Archives processed inbox files

Usage:
    python ingest.py
"""

import os
import glob
import shutil
import hashlib
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


def load_inbox(inbox_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Reads all CSV files from the inbox directory, concatenates them,
    and generates deterministic usid hashes.

    Returns (df_inbox, list_of_csv_paths).
    """
    csv_files = sorted(glob.glob(os.path.join(inbox_dir, "*.csv")))

    if not csv_files:
        return pd.DataFrame(), []

    print(f" -> Found {len(csv_files)} CSV file(s) in inbox:")
    for f in csv_files:
        print(f"    - {os.path.basename(f)}")

    df_list = [
        pd.read_csv(f, low_memory=False, dtype={'userid': str, 'site_id': str})
        for f in csv_files
    ]
    df_inbox = pd.concat(df_list, ignore_index=True)
    df_inbox.columns = df_inbox.columns.str.strip()

    # Generate deterministic USIDs (same logic as DatabaseManager.load_and_merge_data)
    composite_keys = (
        df_inbox['site_id'].fillna('').astype(str) + "_" +
        df_inbox['measured_at'].fillna('').astype(str) + "_" +
        df_inbox['userid'].fillna('').astype(str)
    )
    df_inbox.insert(0, 'usid', composite_keys.apply(
        lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()
    ))

    df_inbox['measured_on'] = pd.to_datetime(df_inbox['measured_on'], errors='coerce')
    df_inbox = df_inbox.dropna(subset=['measured_on']).sort_values('measured_on')

    return df_inbox, csv_files


def report_duplicates(df_inbox: pd.DataFrame, existing_disk: set, existing_tube: set):
    """Logs how many incoming samples have USIDs already in the registry."""
    known = existing_disk | existing_tube
    dup_mask = df_inbox['usid'].isin(known)
    n_dup = dup_mask.sum()
    if n_dup > 0:
        print(f" -> {n_dup} incoming sample(s) match existing USIDs "
              f"(will overwrite as amended measurements).")
    else:
        print(" -> No duplicate USIDs detected — all samples are new.")
    return n_dup


def archive_inbox(csv_files: list[str], processed_dir: str):
    """Moves processed CSV files to the archive directory."""
    os.makedirs(processed_dir, exist_ok=True)
    for f in csv_files:
        dest = os.path.join(processed_dir, os.path.basename(f))
        # Avoid overwriting previous archives with the same filename
        if os.path.exists(dest):
            base, ext = os.path.splitext(os.path.basename(f))
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(processed_dir, f"{base}_{counter}{ext}")
                counter += 1
        shutil.move(f, dest)
        print(f"    Archived: {os.path.basename(f)} -> {dest}")


def print_summary(instrument_name: str, df: pd.DataFrame) -> None:
    """Prints a standardized summary of curation results for new samples."""
    total = len(df)

    passed_h = df['passed_heuristics'].astype(bool)
    is_censored = df.get('is_censored', pd.Series(False, index=df.index)).astype(bool)
    outlier_col = df.get('is_statistical_outlier', pd.Series(None, index=df.index))
    is_outlier = outlier_col.eq(True)

    heuristic_errors = (~passed_h).sum()
    censored_in_model = (passed_h & is_censored).sum()
    outliers = (passed_h & is_outlier).sum()
    valid = (passed_h & ~is_outlier & ~is_censored).sum()

    print(f"{instrument_name.upper()} SUMMARY (NEW SAMPLES):")
    print(f"  Total Ingested:     {total}")
    print(f"  Valid Samples:      {valid}")
    print(f"  Right-Censored:     {censored_in_model} (Lower bounds, included in model)")
    print(f"  Bayesian Outliers:  {outliers} (Statistical anomalies)")
    print(f"  Heuristic Errors:   {heuristic_errors} (Typos, bounds, contradictions)")
    print("-" * 50)


def main():
    """
    Executes the incremental curation pipeline.
    Processes new samples from the inbox using the existing Bayesian model,
    then retrains the model on the updated registry.
    """
    print("=" * 60)
    print("  INCREMENTAL CURATION PIPELINE — Continuous Ingestion")
    print("=" * 60)

    config = load_config()

    # Resolve paths
    db_path = config.get("paths", {}).get("database", "data/master_registry.sqlite")
    flagged_dir = config.get("paths", {}).get("flagged_output", "data/flagged/")
    window_years = config.get("drift", {}).get("window_years", 10)

    inc_config = config.get("incremental", {})
    inbox_dir = inc_config.get("inbox", "inbox/")
    processed_dir = inc_config.get("processed", "inbox/processed/")
    trace_dir = inc_config.get("trace_dir", "output/traces/")
    disk_trace_file = inc_config.get("disk_trace_file", "disk_trace.nc")
    tube_trace_file = inc_config.get("tube_trace_file", "tube_trace.nc")
    retrain = inc_config.get("retrain_after_ingest", True)

    disk_trace_path = os.path.join(trace_dir, disk_trace_file)
    tube_trace_path = os.path.join(trace_dir, tube_trace_file)

    # Verify prerequisites
    if not os.path.exists(db_path):
        print(f"\n[ERROR] Master registry not found at '{db_path}'.")
        print("Run 'python main.py' first to bootstrap the registry and traces.")
        return

    has_disk_trace = os.path.exists(disk_trace_path)
    has_tube_trace = os.path.exists(tube_trace_path)

    if not has_disk_trace and not has_tube_trace:
        print(f"\n[ERROR] No saved model traces found in '{trace_dir}'.")
        print("Run 'python main.py' first to bootstrap the registry and traces.")
        return

    # ------------------------------------------------------------------
    # Step 1: Scan Inbox
    # ------------------------------------------------------------------
    print("\n--- Step 1: Scanning Inbox ---")
    df_inbox, csv_files = load_inbox(inbox_dir)

    if df_inbox.empty:
        print(f"No CSV files found in '{inbox_dir}'. Nothing to process.")
        return

    print(f"Total raw records from inbox: {len(df_inbox)}")

    # ------------------------------------------------------------------
    # Step 2: Report Duplicates (informational — we overwrite amendments)
    # ------------------------------------------------------------------
    print("\n--- Step 2: Checking for Amended Measurements ---")
    db_manager = DatabaseManager(db_path=db_path, overwrite=False)
    existing_disk_usids = db_manager.get_existing_usids("measurements_disk")
    existing_tube_usids = db_manager.get_existing_usids("measurements_tube")
    report_duplicates(df_inbox, existing_disk_usids, existing_tube_usids)

    # ------------------------------------------------------------------
    # Step 3: Refresh Site Info & Validate Site IDs
    # ------------------------------------------------------------------
    print("\n--- Step 3: Refreshing Site Registry ---")
    df_sites, df_versions = db_manager.refresh_site_info(config)
    known_site_ids = set(df_sites['site_id'].astype(str).unique())

    df_merged, df_rejected = db_manager.merge_inbox_with_sites(
        df_inbox, df_versions, known_site_ids
    )

    if not df_rejected.empty:
        print(f"\n  [WARNING] {len(df_rejected)} sample(s) REJECTED — "
              f"site_id not found in the GLOBE site registry:")
        rejected_sites = df_rejected['site_id'].unique()
        for sid in rejected_sites[:10]:  # Show first 10
            count = (df_rejected['site_id'] == sid).sum()
            print(f"    - site_id '{sid}': {count} sample(s)")
        if len(rejected_sites) > 10:
            print(f"    ... and {len(rejected_sites) - 10} more sites.")

        # Export rejected samples for the user
        rejected_path = os.path.join(flagged_dir, "rejected_unknown_sites.csv")
        os.makedirs(flagged_dir, exist_ok=True)
        df_rejected.to_csv(rejected_path, index=False)
        print(f"  -> Rejected samples saved to: {rejected_path}")

    if df_merged.empty:
        print("\nNo valid samples remaining after site validation. Exiting.")
        archive_inbox(csv_files, processed_dir)
        return

    print(f"\n  {len(df_merged)} sample(s) matched to the site registry.")

    # ------------------------------------------------------------------
    # Step 4: Enrich with Water Distances
    # ------------------------------------------------------------------
    if config.get('distance_to_water', {}).get('enabled', False):
        print("\n--- Step 4: Enriching with Water Distance Metadata ---")
        df_merged = compute_water_distances(df_merged, config)
    else:
        print("\n--- Step 4: Water Distance Enrichment (Disabled) ---")

    # ------------------------------------------------------------------
    # Step 5: Route by Instrument & Process
    # ------------------------------------------------------------------
    print("\n--- Step 5: Routing by Instrument & Curating ---")
    df_disk, df_tube = db_manager.split_by_instrument(df_merged)

    curated_disk = None
    curated_tube = None

    # --- Secchi Disk ---
    if not df_disk.empty:
        print(f"\n  Processing {len(df_disk)} Secchi Disk records...")
        validated_disk = validate_secchi_data(df_disk, config)

        if has_disk_trace:
            print("  -> Evaluating against saved Bayesian model (fast path)...")
            disk_model = BayesianDiskModel(config)
            curated_disk = disk_model.evaluate_from_trace(
                validated_disk, disk_trace_path, window_years=window_years
            )
        else:
            print("  -> [WARNING] No disk trace found. Running full MCMC...")
            disk_model = BayesianDiskModel(config)
            curated_disk = disk_model.evaluate(validated_disk, window_years=window_years)

        db_manager.upsert_curated_data(curated_disk, "measurements_disk")
        db_manager.export_audit_log(
            curated_disk, os.path.join(flagged_dir, "flagged_disk_incremental.csv")
        )

    # --- Transparency Tube ---
    if not df_tube.empty:
        print(f"\n  Processing {len(df_tube)} Transparency Tube records...")
        print("  -> Applying tube length estimations...")
        df_tube = apply_all_tube_estimations(df_tube)

        validated_tube = validate_tube_data(df_tube, config)

        if has_tube_trace:
            print("  -> Evaluating against saved Bayesian model (fast path)...")
            tube_model = BayesianTubeModel(config)
            curated_tube = tube_model.evaluate_from_trace(
                validated_tube, tube_trace_path, window_years=window_years
            )
        else:
            print("  -> [WARNING] No tube trace found. Running full MCMC...")
            tube_model = BayesianTubeModel(config)
            curated_tube = tube_model.evaluate(validated_tube, window_years=window_years)

        db_manager.upsert_curated_data(curated_tube, "measurements_tube")
        db_manager.export_audit_log(
            curated_tube, os.path.join(flagged_dir, "flagged_tube_incremental.csv")
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("         INCREMENTAL INGESTION SUMMARY")
    print("=" * 50)

    if curated_disk is not None:
        print_summary("Secchi Disk", curated_disk)
    if curated_tube is not None:
        print_summary("Transparency Tube", curated_tube)

    # ------------------------------------------------------------------
    # Step 6: Retrain Model
    # ------------------------------------------------------------------
    if retrain:
        print("\n--- Step 6: Retraining Bayesian Models ---")
        print("  Loading full curated datasets from master registry...")
        os.makedirs(trace_dir, exist_ok=True)

        # Retrain Disk Model
        full_disk = db_manager.load_curated_data("measurements_disk")
        if not full_disk.empty:
            # Filter out flagged observations for retraining
            clean_disk = full_disk[
                full_disk.get('passed_heuristics', True).astype(bool)
                & ~full_disk.get('is_statistical_outlier', False).eq(True)
            ].copy()

            print(f"  Disk: Retraining on {len(clean_disk)} clean records "
                  f"(of {len(full_disk)} total)...")

            disk_model = BayesianDiskModel(config)
            disk_model.evaluate(clean_disk, window_years=window_years)
            disk_model.save_trace(disk_trace_path)
        else:
            print("  Disk: No data in registry. Skipping retrain.")

        # Retrain Tube Model
        full_tube = db_manager.load_curated_data("measurements_tube")
        if not full_tube.empty:
            clean_tube = full_tube[
                full_tube.get('passed_heuristics', True).astype(bool)
                & ~full_tube.get('is_statistical_outlier', False).eq(True)
            ].copy()

            print(f"  Tube: Retraining on {len(clean_tube)} clean records "
                  f"(of {len(full_tube)} total)...")

            tube_model = BayesianTubeModel(config)
            tube_model.evaluate(clean_tube, window_years=window_years)
            tube_model.save_trace(tube_trace_path)
        else:
            print("  Tube: No data in registry. Skipping retrain.")

        print("  Model retraining complete.")

    # ------------------------------------------------------------------
    # Step 7: Archive Inbox
    # ------------------------------------------------------------------
    print("\n--- Step 7: Archiving Processed Files ---")
    archive_inbox(csv_files, processed_dir)

    print("\n" + "=" * 60)
    print("  Incremental ingestion complete.")
    print("  Master registry and model traces have been updated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
