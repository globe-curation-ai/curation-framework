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

        # Censored data MUST fail heuristics so it doesn't skew the continuous Bayesian model
        df_flagged.loc[censored_mask, 'passed_heuristics'] = False

    # 3. Catch Zeros and Negative Values (LogNormal models fail on <= 0)
    invalid_range_mask = (df_flagged[val_col] <= 0)
    df_flagged.loc[invalid_range_mask, 'passed_heuristics'] = False

    # 4. Catch Environmental Contradictions
    if state_col in df_flagged.columns:
        invalid_states = ['frozen', 'dry', 'unreachable']
        current_states = df_flagged[state_col].fillna('').astype(str).str.lower().str.strip()
        state_mask = current_states.isin(invalid_states)
        df_flagged.loc[state_mask, 'passed_heuristics'] = False

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

    # Summarize results cleanly in the terminal
    total_failed = (~df_flagged['passed_heuristics']).sum()
    censored_count = df_flagged['is_censored'].sum()
    true_errors = total_failed - censored_count

    print(f"  -> Flagged {total_failed} records bypassing Bayesian evaluation:")
    print(f"     - {censored_count} Right-Censored (Disk reached bottom visibly)")
    print(f"     - {true_errors} Heuristic Failures (Typos, negatives, depth contradictions)")

    return df_flagged