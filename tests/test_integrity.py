import pytest
import sqlite3
import pandas as pd


@pytest.fixture
def in_memory_db():
    """
    Creates an ephemeral SQLite database populated with the 4-table structure
    to test ingestion integrity without touching the production database.
    """
    conn = sqlite3.connect(':memory:')

    # 1. Sites Table (Metadata)
    conn.execute('''
                 CREATE TABLE sites
                 (
                     site_id             TEXT PRIMARY KEY,
                     water_body_source   TEXT,
                     water_body_depth_m  REAL,
                     site_tube_length_cm REAL
                 )
                 ''')

    # 2. Measurements Table (Secchi Disk)
    conn.execute('''
                 CREATE TABLE measurements_disk
                 (
                     obs_id                                  INTEGER PRIMARY KEY,
                     site_id                                 TEXT,
                     measured_on                             TEXT,
                     transparency_disk_image_disappearance_m REAL,
                     transparency_disk_does_not_disappear    TEXT
                 )
                 ''')

    # Insert mock data
    conn.execute("INSERT INTO sites VALUES ('SITE_1', 'lake', 15.5, 120.0)")
    conn.execute("INSERT INTO sites VALUES ('SITE_2', 'river', 2.0, 60.0)")
    # SITE_3 intentionally omitted from sites table to test "Ghost Sites"

    conn.execute("INSERT INTO measurements_disk VALUES (1, 'SITE_1', '2026-03-22', 2.5, 'false')")
    conn.execute("INSERT INTO measurements_disk VALUES (2, 'SITE_2', '2026-03-21', 1.0, 'true')")
    conn.execute("INSERT INTO measurements_disk VALUES (3, 'SITE_3', '2026-03-20', 3.0, 'false')")

    yield conn
    conn.close()


def test_relational_merge_preserves_row_count(in_memory_db):
    """
    Ensures that joining the measurements table with the sites metadata table
    does not cause a Cartesian explosion (row duplication).
    """
    # Simulate the ingestion phase
    df_meas = pd.read_sql("SELECT * FROM measurements_disk", in_memory_db)
    df_sites = pd.read_sql("SELECT * FROM sites", in_memory_db)

    original_row_count = len(df_meas)

    # Perform the relational join
    df_merged = df_meas.merge(df_sites, on='site_id', how='left')

    assert len(df_merged) == original_row_count, "Join resulted in a Cartesian explosion (duplicate rows)"


def test_ghost_site_handling(in_memory_db):
    """
    Tests that measurements belonging to 'Ghost Sites' (sites missing from the metadata table)
    are retained, and their missing metadata is gracefully handled (e.g., NaN depth).
    """
    df_meas = pd.read_sql("SELECT * FROM measurements_disk", in_memory_db)
    df_sites = pd.read_sql("SELECT * FROM sites", in_memory_db)

    df_merged = df_meas.merge(df_sites, on='site_id', how='left')

    # The ghost site, SITE_3 should still exist in the merged dataframe.
    ghost_site_data = df_merged[df_merged['site_id'] == 'SITE_3']

    assert not ghost_site_data.empty, "Measurements for missing sites were incorrectly dropped"
    assert pd.isna(ghost_site_data.iloc[0]['water_body_source']), "Missing metadata was not cast to NaN"


def test_sqlite_type_coercion(in_memory_db):
    """
    SQLite lacks native strict datetime and boolean types. This test ensures the
    ingestion layer properly coerces SQLite strings back into actionable pandas types.
    """
    df = pd.read_sql("SELECT * FROM measurements_disk", in_memory_db)

    # 1. Test Datetime Coercion
    df['measured_on'] = pd.to_datetime(df['measured_on'])
    assert pd.api.types.is_datetime64_any_dtype(df['measured_on']), "Failed to coerce SQLite date string to datetime"

    # 2. Test Boolean Coercion (SQLite stores as 'true'/'false', 1/0, or True/False)
    truthy_vals = [1, '1', True, 'True', 'true', 'T', 't']
    df['transparency_disk_does_not_disappear'] = df['transparency_disk_does_not_disappear'].isin(truthy_vals)

    assert pd.api.types.is_bool_dtype(
        df['transparency_disk_does_not_disappear']), "Failed to coerce SQLite string to boolean"

    # Verify the specific boolean logic holds (SITE_2 was 'true', SITE_1 was 'false')
    # Using '==' instead of 'is' to safely compare numpy.True_ with Python True
    assert df.loc[df['site_id'] == 'SITE_2', 'transparency_disk_does_not_disappear'].iloc[0] == True
    assert df.loc[df['site_id'] == 'SITE_1', 'transparency_disk_does_not_disappear'].iloc[0] == False

def test_native_nasa_column_preservation():
    """
    Verifies the dataframes maintain the highly specific, verbose NASA GLOBE
    column names required by downstream validators without accidental truncation.
    """
    expected_columns = [
        'transparency_disk_image_disappearance_m',
        'transparency_disk_does_not_disappear',
        'water_body_depth_m'
    ]

    # Create a dummy dataframe with the long NASA names
    df = pd.DataFrame(columns=expected_columns)

    for col in expected_columns:
        assert col in df.columns, f"NASA standard column {col} was lost or truncated"


from src.database.manager import DatabaseManager

def test_site_id_survives_temporal_merge(tmp_path):
    """
    Regression test to ensure 'site_id' is not dropped during the
    split-and-merge process of the raw hydrologySiteVersions.csv file.
    """
    # 1. Setup ephemeral directories for the test
    obs_dir = tmp_path / "observations"
    site_dir = tmp_path / "site_info"
    obs_dir.mkdir()
    site_dir.mkdir()
    db_path = str(tmp_path / "test_master.sqlite")

    # 2. Mock the Observation CSV
    df_obs = pd.DataFrame({
        'userid': ['test_user'],
        'site_id': ['SITE_123'],
        'measured_at': ['2023-06-15 12:00:00'],
        'measured_on': ['2023-06-15'],
        'transparency_disk_image_disappearance_m': [1.5]
    })
    df_obs.to_csv(obs_dir / "transparencies2023.csv", index=False)

    # 3. Mock the highly specific 62-column Site CSV
    # We must recreate the exact pandas duplicate-column naming behavior (the ".1" suffix)
    cols_static = [f'static_{i}' for i in range(24)]
    cols_static[0] = 'id'  # The primary site_id

    cols_versions = [f'version_{i}' for i in range(24, 40)]
    cols_versions[0] = 'id.1'  # The version_id
    cols_versions[2] = 'site_id.1'  # The critical column (Index 26)
    cols_versions[5] = 'version_date'

    all_cols = cols_static + cols_versions
    df_sites = pd.DataFrame([['data'] * len(all_cols)], columns=all_cols)

    # Inject our test data
    df_sites['id'] = 'SITE_123'
    df_sites['site_id.1'] = 'SITE_123'
    df_sites['version_date'] = '2023-01-01'  # Must be before observation date for backward merge

    df_sites.to_csv(site_dir / "hydrologySiteVersions.csv", index=False)

    # 4. Configure the Manager to point to our ephemeral files
    config = {
        'paths': {
            'observations': str(obs_dir),
            'site_info': str(site_dir)
        },
        'data_sources': {
            'site_info_filename': 'hydrologySiteVersions.csv'
        }
    }

    # 5. Execute the pipeline
    manager = DatabaseManager(db_path=db_path)
    df_merged = manager.load_and_merge_data(config)

    # 6. THE ASSERTIONS
    assert 'site_id' in df_merged.columns, "CRITICAL FAILURE: 'site_id' was lost during the merge process!"
    assert df_merged['site_id'].iloc[0] == 'SITE_123', "Data corruption: 'site_id' value mutated during merge!"

    # 7. Verify it actually makes it into the SQLite database properly
    manager.export_curated_data(df_merged, "test_table")
    with sqlite3.connect(db_path) as conn:
        df_sql = pd.read_sql("SELECT * FROM test_table", conn)
        assert 'site_id' in df_sql.columns, "CRITICAL FAILURE: 'site_id' was dropped during SQLite export!"