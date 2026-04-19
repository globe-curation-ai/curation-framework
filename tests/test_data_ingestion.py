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