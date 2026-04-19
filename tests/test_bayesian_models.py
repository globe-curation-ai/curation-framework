import pandas as pd
import numpy as np
from src.curation.disk_model import BayesianDiskModel


def test_continuous_linear_decay_weights():
    """
    Tests the fading memory architecture to ensure the mathematical
    temporal weights are calculated correctly over a 10-year rolling window.
    """
    config = {
        'bayesian_model': {'disk_target_col': 'transparency_disk_image_disappearance_m'},
        'date_col': 'measured_on',
        'site_col': 'site_id',
        'water_type_col': 'water_body_source'
    }

    model = BayesianDiskModel(config)

    # Establish a dynamic "Current Date" as the anchor point
    base_date = pd.Timestamp.now().normalize()

    # Create a synthetic dataset testing all temporal edge cases
    df = pd.DataFrame({
        'site_id': ['SITE_1'] * 5,
        'transparency_disk_image_disappearance_m': [2.5, 2.5, 2.5, 2.5, 2.5],
        'passed_heuristics': [True, True, True, True, True],
        'measured_on': [
            base_date,                           # Exactly today
            base_date - pd.DateOffset(years=5),  # Exactly 5 years old
            base_date - pd.DateOffset(years=10), # Exactly 10 years old (Boundary)
            base_date - pd.DateOffset(years=15), # 15 years old
            base_date + pd.DateOffset(days=14)   # Typo: 2 weeks in the future
        ]
    })

    model.df_full = df.copy()
    model.df_valid = model.df_full[model.df_full['passed_heuristics']].copy()

    # Execute the data preparation phase with a 10-year rolling window
    model._prepare_data(window_years=10)

    processed_df = model.df_valid

    # A. Future Data Safeguard: Future typos should cap at Age 0 (Weight 1.0)
    assert processed_df.loc[4, 'weight'] == 1.0, "Future dates failed to cap at weight 1.0"

    # B. Current Data: Should have maximum influence (Weight 1.0)
    assert processed_df.loc[0, 'weight'] == 1.0, "Current date failed to receive weight 1.0"

    # C. Midpoint Decay: 5 years into a 10-year window should have ~0.5 weight
    assert np.isclose(processed_df.loc[1, 'weight'], 0.5, atol=0.01), "Linear decay failed at midpoint"

    # D. Historical Exclusion: Data >= 10 years old should be completely dropped
    assert 2 not in processed_df[processed_df['weight'] > 0.001].index, "10-year-old boundary data was not dropped"
    assert 3 not in processed_df.index, "15-year-old historical data was not dropped"


def test_lognormal_boundary_enforcement():
    """
    Tests that the _prepare_data step successfully enforces the strictly
    positive boundary required by the LogNormal PyMC likelihood.
    """
    config = {
        'bayesian_model': {'disk_target_col': 'transparency_disk_image_disappearance_m'},
        'date_col': 'measured_on',
        'site_col': 'site_id'
    }
    model = BayesianDiskModel(config)

    df = pd.DataFrame({
        'site_id': ['SITE_1'] * 3,
        'measured_on': [pd.Timestamp.now().normalize()] * 3,
        'passed_heuristics': [True, True, True],
        'transparency_disk_image_disappearance_m': [1.5, 0.0, -0.5]
    })

    model.df_full = df.copy()
    model.df_valid = model.df_full[model.df_full['passed_heuristics']].copy()
    model._prepare_data(window_years=10)

    # Ensure zeros and negatives were bumped to the 0.01 epsilon buffer
    assert model.df_valid.loc[1, 'transparency_disk_image_disappearance_m'] == 0.01
    assert model.df_valid.loc[2, 'transparency_disk_image_disappearance_m'] == 0.01


def test_empty_valid_bypass():
    """
    Ensures the model gracefully bypasses the PyMC sampler if no records
    pass the heuristic gates, returning the dataframe unmodified.
    """
    config = {
        'bayesian_model': {'disk_target_col': 'transparency_disk_image_disappearance_m'},
        'date_col': 'measured_on',
        'site_col': 'site_id'
    }
    model = BayesianDiskModel(config)

    df = pd.DataFrame({
        'site_id': ['SITE_1', 'SITE_2'],
        'measured_on': [pd.Timestamp.now().normalize()] * 2,
        'transparency_disk_image_disappearance_m': [1.5, 2.0],
        'passed_heuristics': [False, False]  # All records fail heuristics
    })

    result_df = model.evaluate(df)

    assert model.trace is None, "Sampler executed despite empty valid dataframe"
    assert 'is_statistical_outlier' in result_df.columns, "Bypass failed to append outlier column"
    assert not result_df['is_statistical_outlier'].any(), "Bypass generated false positive outliers"


def test_historical_relative_anchoring():
    """
    Tests that the temporal anchor dynamically adjusts to historical archives.
    If a dataset ends in 2010, the 10-year window should be relative to 2010,
    preventing the entire archive from being deleted by the present-day clock.
    """
    config = {
        'bayesian_model': {'disk_target_col': 'transparency_disk_image_disappearance_m'},
        'date_col': 'measured_on',
        'site_col': 'site_id',
        'water_type_col': 'water_body_source'
    }
    model = BayesianDiskModel(config)

    # An archive from 2010. The anchor should become 2010, dropping 1995 but keeping 2005.
    df = pd.DataFrame({
        'site_id': ['SITE_1', 'SITE_2', 'SITE_3'],
        'water_body_source': ['lake', 'lake', 'lake'],
        'measured_on': [
            pd.to_datetime('2010-01-01'),  # Age 0 (Relative Anchor)
            pd.to_datetime('2005-01-01'),  # Age 5 (Kept)
            pd.to_datetime('1995-01-01')   # Age 15 (Dropped)
        ],
        'transparency_disk_image_disappearance_m': [1.5, 2.0, 2.5],
        'passed_heuristics': [True, True, True]
    })

    model.df_full = df.copy()
    model.df_valid = model.df_full[model.df_full['passed_heuristics']].copy()
    model._prepare_data(window_years=10)

    processed_df = model.df_valid

    # Prove the anchor shifted to 2010 and applied relative decay
    assert processed_df.loc[0, 'weight'] == 1.0, "Historical anchor failed to adjust"
    assert 1 in processed_df.index, "Data within historical window was incorrectly dropped"
    assert 2 not in processed_df.index, "Data outside historical window was kept"


def test_pymc_engine_smoke_test():
    """
    A full integration test forcing the PyMC engine to compile the computational
    graph and run a micro-chain (5 draws). Proves that dynamic coordinates,
    hierarchies, and custom Potentials function without tensor dimension errors.
    """
    config = {
        'date_col': 'measured_on',
        'site_col': 'site_id',
        'water_type_col': 'water_body_source',
        'bayesian_model': {
            'disk_target_col': 'transparency_disk_image_disappearance_m',
            'draws': 5,
            'tune': 5,
            'chains': 1
        }
    }
    model = BayesianDiskModel(config)

    # Provide a minimal, clean dataset
    df = pd.DataFrame({
        'site_id': ['SITE_1', 'SITE_1', 'SITE_2'],
        'water_body_source': ['lake', 'lake', 'river'],
        'measured_on': [pd.Timestamp.now().normalize()] * 3,
        'transparency_disk_image_disappearance_m': [2.0, 2.2, 1.5],
        'passed_heuristics': [True, True, True]
    })

    result_df = model.evaluate(df, window_years=10)

    # 1. Assert the sampler actually ran and generated a trace
    assert model.trace is not None, "PyMC sampler failed to generate a trace"

    # 2. Assert the custom Potential compiled and didn't crash
    potential_names = [p.name for p in model.model.potentials]
    assert "weighted_logp" in potential_names, "Custom PyMC Potential was not registered"

    # 3. Assert downstream processing successfully mapped the flags back to the dataframe
    assert 'is_statistical_outlier' in result_df.columns
    assert len(result_df) == 3, "Output dataframe length mismatch"