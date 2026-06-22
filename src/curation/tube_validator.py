import pandas as pd
import numpy as np


def validate_tube_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Applies heuristic rules to crowdsourced Transparency Tube data using native NASA schema.
    Explicitly isolates right-censored (clear water) data from physical impossibilities and typos.
    """
    df_flagged = df.copy()
    df_flagged['passed_heuristics'] = True
    df_flagged['is_censored'] = False  # Explicit tracking for clear water

    val_col = config.get('bayesian_model', {}).get(
        'tube_target_col', 'tube_image_disappearance_cm'
    )

    if val_col not in df_flagged.columns:
        print("  -> No Transparency Tube target column found. Bypassing validation.")
        return df_flagged

    state_col = 'water_body_state'
    saturated_col = 'tube_image_does_not_disappear'

    # 1. Enforce Numeric Types
    df_flagged[val_col] = pd.to_numeric(df_flagged[val_col], errors='coerce')
    missing_mask = df_flagged[val_col].isna()
    df_flagged.loc[missing_mask, 'passed_heuristics'] = False

    # Get dynamic tube lengths if available
    active_tube_length_col = config.get('bayesian_model', {}).get(
        'active_tube_length_col', 'tube_len_mle_site'
    )
    if active_tube_length_col in df_flagged.columns:
        raw_tube_lengths = pd.to_numeric(df_flagged[active_tube_length_col], errors='coerce').replace(0.0, np.nan)
        tube_lengths = raw_tube_lengths.fillna(config.get('validation', {}).get('max_tube_cm', 122.0))
        has_estimate_mask = raw_tube_lengths.notna()
    else:
        tube_lengths = config.get('validation', {}).get('max_tube_cm', 122.0)
        has_estimate_mask = pd.Series(False, index=df_flagged.index)

    # 2. Track Right-Censored Data (Perfectly clear water)
    if saturated_col in df_flagged.columns:
        true_conditions = [True, 'True', 'true', '1', 1, 'Yes', 'yes', 'T', 't']
        censored_mask = df_flagged[saturated_col].isin(true_conditions)

        # Mark as censored so you can analyze clear water later
        df_flagged.loc[censored_mask, 'is_censored'] = True

    # Also infer censorship if the reading hits or exceeds the estimated hardware limit
    hit_limit_mask = df_flagged[val_col] >= tube_lengths
    df_flagged.loc[hit_limit_mask, 'is_censored'] = True

    # Censored data now flows into the Bayesian model via a custom pm.Potential likelihood.
    # The is_censored flag tells the model to use the log-survival function instead of the log-density.

    # 3. Catch Negative Values, Zeros, and Extreme Typos
    max_cm = config.get('validation', {}).get('max_tube_cm', 122.0)
    invalid_range_mask = (df_flagged[val_col] <= 0) | (df_flagged[val_col] > max_cm)
    df_flagged.loc[invalid_range_mask, 'passed_heuristics'] = False
    # If it's physically impossible (> max_cm), it shouldn't be considered valid clear water
    df_flagged.loc[invalid_range_mask, 'is_censored'] = False

    # 4. Catch Environmental Contradictions
    if state_col in df_flagged.columns:
        invalid_states = ['frozen', 'dry', 'unreachable']
        current_states = df_flagged[state_col].fillna('').astype(str).str.lower().str.strip()
        state_mask = current_states.isin(invalid_states)
        df_flagged.loc[state_mask, 'passed_heuristics'] = False
        df_flagged.loc[state_mask, 'is_censored'] = False

    # 5. Logical Mismatch (Claimed clear water, but reading is far below the estimated tube length)
    if saturated_col in df_flagged.columns:
        # A "does not disappear" reading should be near the full tube length.
        # If the reading is significantly lower than the estimated tube length, it's a mismatch.
        # We only apply this check if we actually have a confident MLE estimate for the site.
        invalid_saturation = (
            df_flagged['is_censored'] & 
            has_estimate_mask & 
            (df_flagged[val_col] < (tube_lengths - 5.0))
        )

        # Explicitly revoke censored status for logical mismatches so they count as true errors
        df_flagged.loc[invalid_saturation, 'is_censored'] = False
        df_flagged.loc[invalid_saturation, 'passed_heuristics'] = False

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
    print(f"     - {true_errors} Heuristic Failures (Typos, bounds, contradictions)")
    print(f"  -> {censored_count} Right-Censored samples included in model (as lower bounds)")
    print(f"  -> {(df_flagged['passed_heuristics']).sum()} total records entering Bayesian model")

    return df_flagged