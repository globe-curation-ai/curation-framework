import pandas as pd
import numpy as np


def validate_secchi_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Applies heuristic rules to crowdsourced Secchi Disk data using native NASA schema.
    Explicitly isolates right-censored (clear water hitting the bottom) data from
    physical impossibilities and data-entry errors.
    """
    df_flagged = df.copy()
    df_flagged['passed_heuristics'] = True
    df_flagged['is_censored'] = False  # Explicit tracking for clear water

    val_col = config.get('bayesian_model', {}).get('disk_target_col', 'transparency_disk_image_disappearance_m')

    if val_col not in df_flagged.columns:
        print("  -> No Secchi Disk target column found. Bypassing validation.")
        return df_flagged

    state_col = 'water_body_state'
    saturated_col = 'transparency_disk_does_not_disappear'
    depth_col = 'water_body_depth_m'

    # 1. Enforce Numeric Types
    df_flagged[val_col] = pd.to_numeric(df_flagged[val_col], errors='coerce')
    missing_mask = df_flagged[val_col].isna()
    df_flagged.loc[missing_mask, 'passed_heuristics'] = False

    # 2. Track Right-Censored Data (Perfectly clear water reaching the bottom)
    if saturated_col in df_flagged.columns:
        true_conditions = [True, 'True', 'true', '1', 1, 'Yes', 'yes', 'T', 't']
        censored_mask = df_flagged[saturated_col].isin(true_conditions)

        # Mark as censored so clear water can be analyzed independently later
        df_flagged.loc[censored_mask, 'is_censored'] = True

        # Censored data now flows into the Bayesian model via a custom pm.Potential likelihood.
        # The is_censored flag tells the model to use the survival function instead of the density.

    # 3. Catch Zeros and Negative Values (LogNormal models fail on <= 0)
    invalid_range_mask = (df_flagged[val_col] <= 0)
    df_flagged.loc[invalid_range_mask, 'passed_heuristics'] = False

    # 4. Catch Environmental Contradictions
    if state_col in df_flagged.columns:
        invalid_states = ['frozen', 'dry', 'unreachable']
        current_states = df_flagged[state_col].fillna('').astype(str).str.lower().str.strip()
        state_mask = current_states.isin(invalid_states)
        df_flagged.loc[state_mask, 'passed_heuristics'] = False
        df_flagged.loc[state_mask, 'is_censored'] = False

    # 5. Saturation and Physical Mismatch Logic
    if depth_col in df_flagged.columns:
        tolerance = config.get('validation', {}).get('disk_tolerance_m', 0.5)

        temp_depth = pd.to_numeric(df_flagged[depth_col], errors='coerce')
        has_depth_mask = temp_depth.notna()

        # Mismatch A: User claimed disk didn't disappear, but the reading is much shallower than the known water body depth
        invalid_saturation = has_depth_mask & df_flagged['is_censored'] & (
                    df_flagged[val_col] < (temp_depth - tolerance))

        # Mismatch B: User claimed disk DID disappear, but the reading is physically deeper than the water body itself
        invalid_unsaturation = has_depth_mask & (~df_flagged['is_censored']) & (
                    df_flagged[val_col] >= (temp_depth + tolerance))

        # Revoke censored status for logical mismatches so they count as true errors
        df_flagged.loc[invalid_saturation, 'is_censored'] = False
        df_flagged.loc[invalid_saturation | invalid_unsaturation, 'passed_heuristics'] = False

    # 6. Distance to Water Check
    no_water_count = 0
    if 'water_detected' in df_flagged.columns:
        # We are only confident there is NO water if:
        # 1. water_detected is False (OSM and GEE NDWI failed to find water)
        # 2. ESA WorldCover explicitly classifies the pixel as Bare/Sparse (60), Snow/Ice (70), or Moss/Lichen (100)
        # We DO NOT drop Tree Cover (10) since canopy hides water from satellites.
        # We DO NOT drop Built-up (50) since urban canals and park ponds exist.
        confident_no_water_mask = (
            ~df_flagged['water_detected'].astype(bool) & 
            df_flagged['land_cover_class'].isin([60.0, 70.0, 100.0])
        )
        no_water_count = confident_no_water_mask.sum()
        df_flagged.loc[confident_no_water_mask, 'passed_heuristics'] = False
        df_flagged.loc[confident_no_water_mask, 'is_censored'] = False

    # Guarantee that no record failing heuristics retains valid censored status
    df_flagged.loc[~df_flagged['passed_heuristics'], 'is_censored'] = False

    # Summarize results cleanly in the terminal
    total_failed = (~df_flagged['passed_heuristics']).sum()
    censored_count = df_flagged['is_censored'].sum()
    true_errors = total_failed - no_water_count

    print(f"  -> Rejected {total_failed} records (excluded from Bayesian model):")
    print(f"     - {no_water_count} Confident No Water (Bare land / Snow + >1000m from known waterbody)")
    print(f"     - {true_errors} Heuristic Failures (Typos, negatives, depth contradictions)")
    print(f"  -> {censored_count} Right-Censored samples included in model (as lower bounds)")
    print(f"  -> {(df_flagged['passed_heuristics']).sum()} total records entering Bayesian model")

    return df_flagged