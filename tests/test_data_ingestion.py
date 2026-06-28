import hashlib
import pandas as pd

from src.database.manager import DatabaseManager


def test_public_dataset_merge_integrity():
    """
    Verifies hydrology measurements merge with site metadata without dropping records.
    """

    # Synthetic hydrology measurements (3 records)
    hydro_df = pd.DataFrame({
        'site_id': ['101', '102', '103'],
        'disk_disappear_m': [2.5, 1.8, 4.0]
    })

    # Synthetic siteinfo metadata (Missing site 103)
    siteinfo_df = pd.DataFrame({
        'site_id': ['101', '102'],
        'water_body_source': ['lake', 'river'],
        'water_body_type': ['fresh', 'fresh']
    })

    # A standard left merge ensures we don't lose hydrology data if siteinfo is missing
    merged_df = hydro_df.merge(siteinfo_df, on='site_id', how='left')

    assert len(merged_df) == len(hydro_df), "Merging siteinfo must not drop hydrology records."
    assert 'water_body_source' in merged_df.columns, "Site metadata columns must be present after the merge."

    # Check that missing siteinfo is handled gracefully (Site 103 should have NaN/None for water_body_source)
    assert pd.isna(merged_df.loc[merged_df['site_id'] == '103', 'water_body_source'].iloc[0]), \
        "Unmatched sites should retain their hydrology data but have nulls for missing siteinfo fields."


def test_temporal_site_versioning():
    """
    Verifies that observations are merged with the correct historical site version 
    based on the measurement date using a backward merge_asof.
    """
    # Observations on different dates
    df_obs = pd.DataFrame({
        'site_id': ['101', '101', '101'],
        'measured_on': pd.to_datetime(['2020-01-15', '2021-06-15', '2022-12-01']),
        'disk_disappear_m': [2.5, 3.0, 2.8]
    }).sort_values('measured_on')

    # Site 101 changes characteristics over time
    df_versions = pd.DataFrame({
        'site_id': ['101', '101'],
        'version_id': ['v1', 'v2'],
        'version_date': pd.to_datetime(['2019-01-01', '2022-01-01']),
        'site_status': ['active', 'relocated']
    }).sort_values('version_date')

    # Replicate the DatabaseManager temporal join
    df_merged = pd.merge_asof(
        df_obs,
        df_versions,
        left_on='measured_on',
        right_on='version_date',
        by='site_id',
        direction='backward'
    )

    # 2020 and 2021 measurements should hit v1, 2022 should hit v2
    expected_versions = ['v1', 'v1', 'v2']
    assert df_merged['version_id'].tolist() == expected_versions, \
        "Temporal join failed to map observations to correct historical site versions."


def test_split_by_instrument():
    """
    Verifies that the merged dataset safely splits into disk and tube datasets
    based on non-null values of the respective target columns.
    """
    manager = DatabaseManager(db_path=":memory:")
    
    df_mixed = pd.DataFrame({
        'site_id': ['1', '2', '3', '4'],
        'transparency_disk_image_disappearance_m': [1.2, None, 3.4, None],
        'tube_image_disappearance_cm': [None, 45.0, 12.5, None] # Site 3 has both, Site 4 has neither
    })
    
    df_disk, df_tube = manager.split_by_instrument(
        df_mixed, 
        disk_col='transparency_disk_image_disappearance_m',
        tube_col='tube_image_disappearance_cm'
    )
    
    # Disk should have 2 records (sites 1 and 3)
    assert len(df_disk) == 2
    assert list(df_disk['site_id']) == ['1', '3']
    
    # Tube should have 2 records (sites 2 and 3)
    assert len(df_tube) == 2
    assert list(df_tube['site_id']) == ['2', '3']


def test_usid_deterministic_generation():
    """
    Verifies that the composite key hashing logic (site + date + userid) 
    generates deterministic tracking IDs.
    """
    df_obs = pd.DataFrame({
        'site_id': ['101', '101', None],
        'measured_at': ['2022-01-01 12:00:00', '2022-01-01 12:00:00', '2022-01-01 12:00:00'],
        'userid': ['99', '99', 'something_else']
    })
    
    # Simulate the hashing code inside load_and_merge_data
    composite_keys = (
        df_obs['site_id'].fillna('').astype(str) + "_" +
        df_obs['measured_at'].fillna('').astype(str) + "_" +
        df_obs['userid'].fillna('').astype(str)
    )
    df_obs.insert(0, 'usid', composite_keys.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()))
    
    # Assert deterministic output (first two rows should have identical usids)
    assert df_obs['usid'].iloc[0] == df_obs['usid'].iloc[1], "Identical records yield different hashes"
    
    # Assert missing fields handle gracefully without erroring or matching populated fields
    assert df_obs['usid'].iloc[0] != df_obs['usid'].iloc[2], "Distinct records hash collided"


def test_export_normalization_stripping():
    """
    Verifies that redundant site version columns are stripped out prior to export,
    leaving only foreign keys.
    """
    manager = DatabaseManager(db_path=":memory:")
    
    # Simulate manager state after processing site versions
    manager.version_columns = ['version_id', 'version_date', 'version_comments', 'site_id']
    
    # Dataframe ready for export
    df_export = pd.DataFrame({
        'usid': ['hash1', 'hash2'],
        'site_id': ['101', '102'],
        'disk_val': [1.0, 2.0],
        'version_id': ['v1', 'v2'],
        'version_date': ['2020-01-01', '2021-01-01'], # Should be dropped
        'version_comments': ['Testing', 'Notes']       # Should be dropped
    })
    
    protected_columns = {'version_id', 'site_id'}
    cols_to_drop = [col for col in manager.version_columns if 
                    col in df_export.columns and col not in protected_columns]
    
    df_stripped = df_export.drop(columns=cols_to_drop, errors='ignore')
    
    assert 'version_date' not in df_stripped.columns
    assert 'version_comments' not in df_stripped.columns
    assert 'version_id' in df_stripped.columns
    assert 'site_id' in df_stripped.columns
    assert 'disk_val' in df_stripped.columns


def test_get_existing_usids_empty_table():
    """
    Verifies that get_existing_usids returns an empty set when the
    target table does not exist yet (pre-bootstrap state).
    """
    manager = DatabaseManager(db_path=":memory:")
    result = manager.get_existing_usids("measurements_disk")
    assert result == set(), "Should return empty set for non-existent table"


def test_get_existing_usids_returns_stored_ids():
    """
    Verifies that get_existing_usids correctly returns all USIDs
    stored in a curated table.
    """
    import sqlite3

    manager = DatabaseManager(db_path=":memory:")
    # Manually create a table with some USIDs
    with sqlite3.connect(":memory:") as conn:
        pass  # :memory: is ephemeral — use the manager's path instead

    # Use a temp file for this test
    import tempfile, os
    db_file = os.path.join(tempfile.mkdtemp(), "test.sqlite")
    manager = DatabaseManager(db_path=db_file)
    manager.version_columns = []

    df = pd.DataFrame({
        'usid': ['aaa', 'bbb', 'ccc'],
        'site_id': ['1', '2', '3'],
        'value': [1.0, 2.0, 3.0]
    })

    import sqlite3
    with sqlite3.connect(db_file) as conn:
        df.to_sql("measurements_disk", conn, if_exists='replace', index=False)

    result = manager.get_existing_usids("measurements_disk")
    assert result == {'aaa', 'bbb', 'ccc'}, f"Expected 3 USIDs, got {result}"


def test_upsert_overwrites_amended_measurements():
    """
    Verifies that upsert_curated_data overwrites rows with matching USIDs
    (GLOBE allows measurement amendments) while preserving non-conflicting rows.
    """
    import tempfile, os, sqlite3

    db_file = os.path.join(tempfile.mkdtemp(), "test_upsert.sqlite")
    manager = DatabaseManager(db_path=db_file)
    manager.version_columns = []

    # Initial batch
    df_initial = pd.DataFrame({
        'usid': ['aaa', 'bbb'],
        'site_id': ['1', '2'],
        'value': [1.0, 2.0],
        'passed_heuristics': [True, True]
    })
    manager.upsert_curated_data(df_initial, "measurements_disk")

    # Amended batch — 'aaa' is updated, 'ccc' is new
    df_amended = pd.DataFrame({
        'usid': ['aaa', 'ccc'],
        'site_id': ['1', '3'],
        'value': [9.99, 3.0],  # aaa changed from 1.0 to 9.99
        'passed_heuristics': [True, True]
    })
    manager.upsert_curated_data(df_amended, "measurements_disk")

    # Read back
    with sqlite3.connect(db_file) as conn:
        df_result = pd.read_sql("SELECT * FROM measurements_disk", conn)

    assert len(df_result) == 3, f"Expected 3 total rows, got {len(df_result)}"

    # Verify the amended value was overwritten
    aaa_row = df_result[df_result['usid'] == 'aaa']
    assert len(aaa_row) == 1, "Should have exactly one row for USID 'aaa'"
    assert aaa_row['value'].iloc[0] == 9.99, (
        f"Amended value should be 9.99, got {aaa_row['value'].iloc[0]}"
    )

    # Verify original non-conflicting row is preserved
    bbb_row = df_result[df_result['usid'] == 'bbb']
    assert len(bbb_row) == 1
    assert bbb_row['value'].iloc[0] == 2.0


def test_merge_inbox_rejects_unknown_sites():
    """
    Verifies that merge_inbox_with_sites rejects observations whose
    site_id is absent from the known site registry.
    """
    manager = DatabaseManager(db_path=":memory:")

    df_obs = pd.DataFrame({
        'site_id': ['101', '102', '999'],  # 999 is unknown
        'measured_on': pd.to_datetime(['2023-01-15', '2023-06-01', '2023-03-10']),
        'value': [2.5, 3.0, 1.0]
    })

    df_versions = pd.DataFrame({
        'site_id': ['101', '102'],
        'version_id': ['v1', 'v2'],
        'version_date': pd.to_datetime(['2020-01-01', '2020-01-01']),
    }).sort_values('version_date')

    known_site_ids = {'101', '102'}

    df_merged, df_rejected = manager.merge_inbox_with_sites(
        df_obs, df_versions, known_site_ids
    )

    assert len(df_rejected) == 1, f"Expected 1 rejected, got {len(df_rejected)}"
    assert df_rejected['site_id'].iloc[0] == '999'
    assert len(df_merged) == 2, f"Expected 2 merged, got {len(df_merged)}"


def test_merge_inbox_temporal_join_correctness():
    """
    Verifies that merge_inbox_with_sites performs a correct backward
    temporal join, attaching the right site version to each observation.
    """
    manager = DatabaseManager(db_path=":memory:")

    df_obs = pd.DataFrame({
        'site_id': ['101', '101'],
        'measured_on': pd.to_datetime(['2020-06-01', '2023-06-01']),
        'value': [2.5, 3.0]
    })

    df_versions = pd.DataFrame({
        'site_id': ['101', '101'],
        'version_id': ['v1', 'v2'],
        'version_date': pd.to_datetime(['2019-01-01', '2022-01-01']),
    }).sort_values('version_date')

    known_site_ids = {'101'}

    df_merged, df_rejected = manager.merge_inbox_with_sites(
        df_obs, df_versions, known_site_ids
    )

    assert len(df_rejected) == 0
    assert len(df_merged) == 2
    assert df_merged['version_id'].tolist() == ['v1', 'v2'], (
        "2020 observation should hit v1, 2023 observation should hit v2"
    )


def test_load_curated_data_roundtrip():
    """
    Verifies that data written via upsert_curated_data can be read
    back via load_curated_data with the same content.
    """
    import tempfile, os

    db_file = os.path.join(tempfile.mkdtemp(), "test_roundtrip.sqlite")
    manager = DatabaseManager(db_path=db_file)
    manager.version_columns = []

    df_original = pd.DataFrame({
        'usid': ['x1', 'x2', 'x3'],
        'site_id': ['A', 'B', 'C'],
        'value': [10.0, 20.0, 30.0],
        'passed_heuristics': [True, True, False],
        'is_statistical_outlier': [False, True, None]
    })

    manager.upsert_curated_data(df_original, "measurements_disk")
    df_loaded = manager.load_curated_data("measurements_disk")

    assert len(df_loaded) == 3, f"Expected 3 rows, got {len(df_loaded)}"
    assert set(df_loaded['usid']) == {'x1', 'x2', 'x3'}
    assert df_loaded[df_loaded['usid'] == 'x2']['value'].iloc[0] == 20.0


def test_load_curated_data_nonexistent_table():
    """
    Verifies that load_curated_data returns an empty DataFrame when
    the requested table does not exist (pre-bootstrap).
    """
    import tempfile, os

    db_file = os.path.join(tempfile.mkdtemp(), "test_empty.sqlite")
    manager = DatabaseManager(db_path=db_file)

    # Create the database file but don't create any tables
    import sqlite3
    with sqlite3.connect(db_file) as conn:
        pass

    df = manager.load_curated_data("nonexistent_table")
    assert df.empty, "Should return empty DataFrame for non-existent table"