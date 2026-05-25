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

    val_col = config.get('bayesian_model', {}).get('tube_target_col', 'tube_image_disappearance_cm')

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
    active_tube_length_col = config.get('bayesian_model', {}).get('active_tube_length_col', 'tube_len_mle_site')
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

    # Censored data MUST fail heuristics so it doesn't break the continuous Bayesian model
    df_flagged.loc[df_flagged['is_censored'], 'passed_heuristics'] = False

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

    # 5. Logical Mismatch (Claimed clear water, but reading is far below the estimated tube length)
    if saturated_col in df_flagged.columns:
        # A "does not disappear" reading should be near the full tube length.
        # If the reading is significantly lower than the estimated tube length, it's a mismatch.
        # We only apply this check if we actually have a confident MLE estimate for the site.
        mismatch_mask = df_flagged['is_censored'] & has_estimate_mask & (df_flagged[val_col] < (tube_lengths - 5.0))

        # Revoke the censored status so it gets counted properly as a heuristic error
        df_flagged.loc[mismatch_mask, 'is_censored'] = False
        df_flagged.loc[mismatch_mask, 'passed_heuristics'] = False

    # Summarize results cleanly in the terminal
    total_failed = (~df_flagged['passed_heuristics']).sum()
    censored_count = df_flagged['is_censored'].sum()
    true_errors = total_failed - censored_count

    print(f"  -> Flagged {total_failed} records bypassing Bayesian evaluation:")
    print(f"     - {censored_count} Right-Censored (Perfectly clear water)")
    print(f"     - {true_errors} Heuristic Failures (Typos, negatives, logical mismatches)")

    return df_flagged