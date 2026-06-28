"""
Tests for the distance-to-water heuristic module.

All OSM and GEE calls are mocked to ensure tests run without network
dependencies. Tests verify the core logic: haversine formula, source
prioritization, temporal merge, and edge cases.
"""

import math
import os
import sqlite3
import datetime

import pytest
import pandas as pd
import numpy as np

from src.curation.distance_to_water import (
    haversine_distance,
    compute_distance_for_location,
    compute_water_distances,
    load_water_distance_db,
    save_water_distance_db,
    query_osm_water,
)


# ---------------------------------------------------------------------------
# Haversine Distance Tests
# ---------------------------------------------------------------------------

class TestHaversineDistance:
    """Tests for the haversine great-circle distance formula."""

    def test_known_distance_london_to_paris(self):
        """London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ≈ 343.5 km."""
        dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        # Allow 1% tolerance for spherical approximation
        assert abs(dist - 343_500) < 5_000

    def test_same_point_is_zero(self):
        """Distance from a point to itself should be 0."""
        dist = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert dist == 0.0

    def test_antipodal_points(self):
        """Opposite sides of Earth ≈ half circumference ≈ 20,037 km."""
        dist = haversine_distance(0.0, 0.0, 0.0, 180.0)
        assert abs(dist - 20_037_500) < 100_000

    def test_small_distance_meters(self):
        """Two points ~111m apart (0.001 degree latitude at equator)."""
        dist = haversine_distance(0.0, 0.0, 0.001, 0.0)
        assert 100 < dist < 120  # ~111 meters

    def test_symmetry(self):
        """Distance A->B should equal B->A."""
        d1 = haversine_distance(37.7749, -122.4194, 34.0522, -118.2437)
        d2 = haversine_distance(34.0522, -118.2437, 37.7749, -122.4194)
        assert d1 == pytest.approx(d2)

    def test_negative_coordinates(self):
        """Works correctly with southern/western hemisphere coordinates."""
        dist = haversine_distance(-33.8688, 151.2093, -37.8136, 144.9631)
        assert 700_000 < dist < 800_000  # Sydney to Melbourne ≈ 714 km


# ---------------------------------------------------------------------------
# Single Location Distance Computation Tests
# ---------------------------------------------------------------------------

class TestComputeDistanceForLocation:
    """Tests for the two-source distance computation logic."""

    @pytest.fixture
    def base_config(self):
        return {
            'distance_to_water': {
                'osm_radius_m': 300,
                'max_distance_m': 1000,
                'ndwi_threshold': 0.3,
                'cloud_cover_max_pct': 5,
                'sentinel_pixel_m': 10,
            }
        }

    def test_osm_finds_water_returns_osm_source(self, base_config, monkeypatch):
        """When OSM finds water within radius, source should be 'osm'."""
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_osm_water',
            lambda lat, lon, radius_m: 150.0
        )
        result = compute_distance_for_location(
            40.0, -74.0, base_config
        )
        assert result['distance_combined_source'] == 'osm'
        assert result['distance_combined_m'] == 150.0

    def test_osm_finds_nothing_no_gee_returns_none(self, base_config,
                                                    monkeypatch):
        """When OSM finds nothing and GEE is unavailable, source is 'none'."""
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_osm_water',
            lambda lat, lon, radius_m: None
        )
        result = compute_distance_for_location(
            40.0, -74.0, base_config, gee_available=False
        )
        assert result['distance_combined_source'] == 'none'
        assert result['distance_combined_m'] == 1000.0

    def test_osm_beyond_radius_gee_finds_water(self, base_config,
                                                monkeypatch):
        """When OSM finds water beyond the 300m radius but GEE finds closer
        water, the GEE result should be preferred."""
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_osm_water',
            lambda lat, lon, radius_m: 500.0  # Beyond 300m radius
        )
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_gee_environmental_features',
            lambda lat, lon, **kwargs: {'distance_m': 200.0, 'land_cover': 10, 'precip_14d_mm': 5.0}
        )
        result = compute_distance_for_location(
            40.0, -74.0, base_config, gee_available=True
        )
        assert result['distance_combined_source'] == 'gee'
        assert result['distance_combined_m'] == 200.0

    def test_osm_exception_is_graceful(self, base_config, monkeypatch):
        """OSM throwing an exception should not crash — treated as not found."""
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_osm_water',
            lambda lat, lon, radius_m: None
        )
        result = compute_distance_for_location(
            40.0, -74.0, base_config, gee_available=False
        )
        assert result['distance_combined_source'] == 'none'

    def test_max_distance_sentinel_value(self, base_config, monkeypatch):
        """When no water is found, distance should be exactly max_distance_m."""
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_osm_water',
            lambda lat, lon, radius_m: None
        )
        result = compute_distance_for_location(
            40.0, -74.0, base_config, gee_available=False
        )
        assert result['distance_combined_m'] == base_config['distance_to_water']['max_distance_m']

    def test_osm_returns_zero_distance(self, base_config, monkeypatch):
        """Site directly on water (distance=0) should work correctly."""
        monkeypatch.setattr(
            'src.curation.distance_to_water.query_osm_water',
            lambda lat, lon, radius_m: 0.0
        )
        result = compute_distance_for_location(
            40.0, -74.0, base_config
        )
        assert result['distance_combined_m'] == 0.0
        assert result['distance_combined_source'] == 'osm'


# ---------------------------------------------------------------------------
# Pre-computed Database I/O Tests
# ---------------------------------------------------------------------------

class TestDatabaseIO:
    """Tests for saving and loading the water distance SQLite database."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'site_id': ['SITE_1', 'SITE_2', 'SITE_3'],
            'version_date': ['2023-01-01', '2023-06-15', '2024-01-01'],
            'latitude': [40.0, 41.0, 42.0],
            'longitude': [-74.0, -73.0, -72.0],
            'distance_osm_m': [150.0, 1000.0, 1000.0],
            'distance_gee_m': [1000.0, 50.0, 1000.0],
            'distance_combined_m': [150.0, 50.0, 1000.0],
            'distance_combined_source': ['osm', 'gee', 'none'],
            'land_cover_class': [10, 20, 30],
            'precip_14d_mm': [0.0, 5.0, 10.0],
            'computed_at': ['2024-01-01T00:00:00', '2024-01-01T00:00:00',
                            '2024-01-01T00:00:00'],
        })

    def test_roundtrip_save_and_load(self, tmp_path, sample_df):
        """Data survives a save/load cycle without corruption."""
        db_path = str(tmp_path / "test_water.sqlite")
        save_water_distance_db(sample_df, db_path)

        loaded = load_water_distance_db(db_path)
        assert len(loaded) == 3
        assert set(loaded['site_id']) == {'SITE_1', 'SITE_2', 'SITE_3'}
        assert loaded['distance_combined_m'].tolist() == [150.0, 50.0,
                                                             1000.0]

    def test_missing_database_returns_empty(self, tmp_path):
        """Loading from a nonexistent path returns empty DataFrame."""
        loaded = load_water_distance_db(
            str(tmp_path / "nonexistent.sqlite")
        )
        assert loaded.empty
        assert 'site_id' in loaded.columns

    def test_database_directory_created(self, tmp_path, sample_df):
        """Saving to a nested path creates intermediate directories."""
        db_path = str(tmp_path / "nested" / "dir" / "water.sqlite")
        save_water_distance_db(sample_df, db_path)
        assert os.path.exists(db_path)


# ---------------------------------------------------------------------------
# Pipeline Integration Tests (Temporal Merge)
# ---------------------------------------------------------------------------

class TestComputeWaterDistances:
    """Tests for the main pipeline integration function."""

    @pytest.fixture
    def water_db(self, tmp_path):
        """Creates a temporary water distance database."""
        db_path = str(tmp_path / "water_distances_osm.sqlite")
        df = pd.DataFrame({
            'site_id': ['SITE_A', 'SITE_A', 'SITE_B'],
            'version_date': ['2020-01-01', '2023-01-01', '2022-06-01'],
            'latitude': [40.0, 40.001, 41.0],
            'longitude': [-74.0, -74.001, -73.0],
            'distance_osm_m': [100.0, 120.0, 500.0],
            'distance_gee_m': [1000.0, 1000.0, 1000.0],
            'distance_combined_m': [100.0, 120.0, 500.0],
            'distance_combined_source': ['osm', 'osm', 'osm'],
            'land_cover_class': [10, 10, 10],
            'precip_14d_mm': [0.0, 0.0, 0.0],
            'computed_at': ['2024-01-01'] * 3,
        })
        save_water_distance_db(df, db_path)
        return db_path

    def test_output_columns_added(self, water_db):
        """After enrichment, the 3 water distance columns must exist."""
        config = {
            'distance_to_water': {
                'enabled': True,
                'database': water_db,
                'max_distance_m': 1000,
            }
        }
        df = pd.DataFrame({
            'site_id': ['SITE_A'],
            'measured_on': ['2022-06-15'],
        })
        result = compute_water_distances(df, config)
        assert 'distance_from_water_m' in result.columns
        assert 'distance_from_water_source' in result.columns
        assert 'water_detected' in result.columns

    def test_temporal_merge_selects_correct_version(self, water_db):
        """
        SITE_A has two versions (2020 and 2023). An observation in 2021
        should match the 2020 version (backward merge), and an observation
        in 2024 should match the 2023 version.
        """
        config = {
            'distance_to_water': {
                'enabled': True,
                'database': water_db,
                'max_distance_m': 1000,
            }
        }
        df = pd.DataFrame({
            'site_id': ['SITE_A', 'SITE_A'],
            'measured_on': ['2021-06-01', '2024-01-15'],
        })
        result = compute_water_distances(df, config)
        result = result.sort_values('measured_on').reset_index(drop=True)

        # 2021 observation → 2020 version (100m)
        assert result.loc[0, 'distance_from_water_m'] == 100.0
        # 2024 observation → 2023 version (120m)
        assert result.loc[1, 'distance_from_water_m'] == 120.0

    def test_unmatched_site_gets_defaults(self, water_db):
        """Sites not in the database get the sentinel max_distance value."""
        config = {
            'distance_to_water': {
                'enabled': True,
                'database': water_db,
                'max_distance_m': 1000,
            }
        }
        df = pd.DataFrame({
            'site_id': ['UNKNOWN_SITE'],
            'measured_on': ['2023-01-01'],
        })
        result = compute_water_distances(df, config)
        assert result.loc[0, 'distance_from_water_m'] == 1000.0
        assert result.loc[0, 'distance_from_water_source'] == 'none'
        assert result.loc[0, 'water_detected'] == False

    def test_disabled_config_returns_unchanged(self):
        """When enabled=False, the input DataFrame is returned unchanged."""
        config = {'distance_to_water': {'enabled': False}}
        df = pd.DataFrame({
            'site_id': ['SITE_A'],
            'measured_on': ['2023-01-01'],
        })
        result = compute_water_distances(df, config)
        assert 'distance_from_water_m' not in result.columns
        assert len(result) == len(df)

    def test_missing_database_adds_default_columns(self, tmp_path):
        """When the database file doesn't exist, defaults are applied
        and a warning is emitted."""
        config = {
            'distance_to_water': {
                'enabled': True,
                'database': str(tmp_path / 'nonexistent.sqlite'),
                'max_distance_m': 1000,
            }
        }
        df = pd.DataFrame({
            'site_id': ['SITE_A'],
            'measured_on': ['2023-01-01'],
        })
        result = compute_water_distances(df, config)
        assert result.loc[0, 'distance_from_water_m'] == 1000.0
        assert result.loc[0, 'distance_from_water_source'] == 'none'
        assert result.loc[0, 'water_detected'] == False

    def test_row_count_preserved(self, water_db):
        """The merge should not create or lose rows."""
        config = {
            'distance_to_water': {
                'enabled': True,
                'database': water_db,
                'max_distance_m': 1000,
            }
        }
        df = pd.DataFrame({
            'site_id': ['SITE_A', 'SITE_B', 'SITE_A', 'UNKNOWN'],
            'measured_on': ['2022-01-01', '2023-01-01',
                            '2024-01-01', '2023-06-01'],
        })
        result = compute_water_distances(df, config)
        assert len(result) == 4
