import os
import glob
import sqlite3
import pandas as pd
import requests
import hashlib
import datetime


class DatabaseManager:
    """
    Manages data ingestion, cache synchronization, and temporal site-registry merging.
    Constructs a normalized relational database schema while providing flattened
    data views for downstream curation models.
    """

    def __init__(self, db_path: str, overwrite: bool = False):
        self.db_path = db_path
        self.version_columns = []
        if overwrite and os.path.exists(self.db_path):
            os.remove(self.db_path)
            print(f" -> Wiped existing database at {self.db_path}")

    def _download_file(self, url: str, dest_path: str):
        """Helper method to download a file with streaming to handle large datasets safely."""
        print(f"    Downloading missing asset: {url}")
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"    Saved to: {dest_path}")
        except requests.exceptions.RequestException as e:
            print(f"    Failed to download {url}: {e}")

    def sync_local_data(self, config: dict, start_year: int = 1995, end_year: int = None):
        """Ensures all necessary CSV files exist locally in the data/ cache."""
        if end_year is None:
            end_year = datetime.datetime.now().year - 1

        print(f" -> Synchronizing local data cache for years {start_year}-{end_year}...")
        obs_dir = config.get('paths', {}).get('observations', 'data/observations/')
        site_dir = config.get('paths', {}).get('site_info', 'data/site_info/')
        os.makedirs(obs_dir, exist_ok=True)
        os.makedirs(site_dir, exist_ok=True)

        data_sources = config.get('data_sources', {})
        base_url = data_sources.get('observations_base_url')
        if base_url:
            for year in range(start_year, end_year + 1):
                file_name = f"transparencies{year}.csv"
                local_path = os.path.join(obs_dir, file_name)
                if not os.path.exists(local_path):
                    self._download_file(base_url.format(year=year), local_path)

        site_url = data_sources.get('site_info_url')
        site_filename = data_sources.get('site_info_filename', 'hydrologySiteVersions.csv')
        if site_url:
            local_site_path = os.path.join(site_dir, site_filename)
            if not os.path.exists(local_site_path):
                self._download_file(site_url, local_site_path)

    def load_and_merge_data(self, config: dict, start_year: int = 1995, end_year: int = None) -> pd.DataFrame:
        """Loads synchronized data, normalizes the registry, and creates the working dataframe."""
        if end_year is None:
            end_year = datetime.datetime.now().year - 1

        self.sync_local_data(config, start_year=start_year, end_year=end_year)

        obs_dir = config.get('paths', {}).get('observations', 'data/observations/')
        site_dir = config.get('paths', {}).get('site_info', 'data/site_info/')
        site_filename = config.get('data_sources', {}).get('site_info_filename', 'hydrologySiteVersions.csv')

        # 1. Load and clean observations (Targeted to the specific years)
        csv_files = [os.path.join(obs_dir, f"transparencies{year}.csv") for year in range(start_year, end_year + 1)]
        csv_files = [f for f in csv_files if os.path.exists(f)]

        if not csv_files:
            raise FileNotFoundError(f"No observation files found for years {start_year}-{end_year} in: {obs_dir}")

        print(f" -> Loading observations from {len(csv_files)} files...")
        df_list = [pd.read_csv(f, low_memory=False, dtype={'userid': str, 'site_id': str}) for f in csv_files]
        df_obs = pd.concat(df_list, ignore_index=True)
        df_obs.columns = df_obs.columns.str.strip()

        # Assign deterministic sample tracking ID (usid)
        print(" -> Generating deterministic sample tracking IDs (usid)...")
        composite_keys = (
                df_obs['site_id'].fillna('').astype(str) + "_" +
                df_obs['measured_at'].fillna('').astype(str) + "_" +
                df_obs['userid'].fillna('').astype(str)
        )
        df_obs.insert(0, 'usid', composite_keys.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()))

        df_obs['measured_on'] = pd.to_datetime(df_obs['measured_on'], errors='coerce')
        df_obs = df_obs.dropna(subset=['measured_on']).sort_values('measured_on')

        # 2. Normalize site data
        site_path = os.path.join(site_dir, site_filename)
        print(" -> Normalizing site registry into relational tables...")
        df_raw_sites = pd.read_csv(site_path, low_memory=False)

        # Split at the index 24 seam
        df_sites = df_raw_sites.iloc[:, :24].copy()
        df_versions = df_raw_sites.iloc[:, 24:].copy()

        # Format static site data
        df_sites.rename(columns={'id': 'site_id'}, inplace=True)
        df_sites['site_id'] = df_sites['site_id'].astype(str)
        df_sites = df_sites.drop_duplicates(subset=['site_id'])

        # Format version history (resolving pandas '.1' duplicate assignment)
        rename_map = {
            'id.1': 'version_id',
            'site_id.1': 'site_id',
            'activated_at.1': 'version_activation',
            'comments.1': 'version_comments',
            'created_at.1': 'version_created_at',
            'updated_at.1': 'version_updated_at',
            'old_schoolid.1': 'version_old_schoolid',
            'old_siteid.1': 'version_old_siteid'
        }
        df_versions.rename(columns=rename_map, inplace=True)
        df_versions['site_id'] = df_versions['site_id'].astype(str)
        df_versions['version_date'] = pd.to_datetime(df_versions['version_date'], errors='coerce')
        df_versions = df_versions.dropna(subset=['version_date']).sort_values('version_date')

        # Export reference tables immediately to the database
        with sqlite3.connect(self.db_path) as conn:
            df_sites.to_sql("sites", conn, if_exists='replace', index=False)
            df_versions.to_sql("site_versions", conn, if_exists='replace', index=False)

        # Track version columns to strip them out later for normalized export
        self.version_columns = df_versions.columns.tolist()

        # 3. Perform temporal point-in-time join
        # Links the exact site version logic specifically for the validation models
        print(f" -> Executing temporal join across {len(df_obs)} observations...")
        df_merged = pd.merge_asof(
            df_obs,
            df_versions,
            left_on='measured_on',
            right_on='version_date',
            by='site_id',
            direction='backward',
            suffixes=('_sample', '_site')
        )

        return df_merged

    def split_by_instrument(self, df: pd.DataFrame, disk_col: str = 'transparency_disk_image_disappearance_m',
                            tube_col: str = 'tube_image_disappearance_cm') -> tuple[pd.DataFrame, pd.DataFrame]:
        """Routes the merged data stream into instrument-specific subsets based on NASA column names."""
        df_disk = df[df[disk_col].notna()].copy() if disk_col in df.columns else pd.DataFrame()
        df_tube = df[df[tube_col].notna()].copy() if tube_col in df.columns else pd.DataFrame()
        return df_disk, df_tube

    def export_curated_data(self, df: pd.DataFrame, table_name: str):
        """Exports curated observations to SQLite while maintaining strict normalization."""
        export_df = df.copy()

        # Keep the foreign key linking to the exact site state
        if 'version_id' not in export_df.columns:
            export_df['version_id'] = None

        # Strip out all other site version columns to eliminate redundancy
        # Protect both version_id and the primary site_id from deletion
        protected_columns = {'version_id', 'site_id'}
        cols_to_drop = [col for col in self.version_columns if
                        col in export_df.columns and col not in protected_columns]

        export_df = export_df.drop(columns=cols_to_drop, errors='ignore')

        with sqlite3.connect(self.db_path) as conn:
            export_df.to_sql(table_name, conn, if_exists='replace', index=False)

    def export_audit_log(self, df: pd.DataFrame, output_path: str):
        """Exports rejected or flagged observations for human review."""
        if 'passed_heuristics' in df.columns and 'is_statistical_outlier' in df.columns:
            flagged_mask = (~df['passed_heuristics']) | (df['is_statistical_outlier'])
            df_flagged = df[flagged_mask]
        else:
            df_flagged = df
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_flagged.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Incremental / Continuous Curation Methods
    # ------------------------------------------------------------------

    def refresh_site_info(self, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Re-downloads the latest hydrologySiteVersions.csv and reloads the
        sites and site_versions tables in the master registry.

        Returns (df_sites, df_versions).
        """
        data_sources = config.get('data_sources', {})
        site_dir = config.get('paths', {}).get('site_info', 'data/site_info/')
        site_url = data_sources.get('site_info_url')
        site_filename = data_sources.get('site_info_filename', 'hydrologySiteVersions.csv')
        os.makedirs(site_dir, exist_ok=True)

        local_site_path = os.path.join(site_dir, site_filename)

        # Re-download to obtain the latest site metadata
        if site_url:
            if os.path.exists(local_site_path):
                os.remove(local_site_path)
            self._download_file(site_url, local_site_path)
        elif not os.path.exists(local_site_path):
            raise FileNotFoundError(
                f"No site info URL configured and no local file at {local_site_path}"
            )

        print(" -> Normalizing refreshed site registry into relational tables...")
        df_raw_sites = pd.read_csv(local_site_path, low_memory=False)

        df_sites = df_raw_sites.iloc[:, :24].copy()
        df_versions = df_raw_sites.iloc[:, 24:].copy()

        df_sites.rename(columns={'id': 'site_id'}, inplace=True)
        df_sites['site_id'] = df_sites['site_id'].astype(str)
        df_sites = df_sites.drop_duplicates(subset=['site_id'])

        rename_map = {
            'id.1': 'version_id',
            'site_id.1': 'site_id',
            'activated_at.1': 'version_activation',
            'comments.1': 'version_comments',
            'created_at.1': 'version_created_at',
            'updated_at.1': 'version_updated_at',
            'old_schoolid.1': 'version_old_schoolid',
            'old_siteid.1': 'version_old_siteid'
        }
        df_versions.rename(columns=rename_map, inplace=True)
        df_versions['site_id'] = df_versions['site_id'].astype(str)
        df_versions['version_date'] = pd.to_datetime(df_versions['version_date'], errors='coerce')
        df_versions = df_versions.dropna(subset=['version_date']).sort_values('version_date')

        # Update the registry's reference tables
        with sqlite3.connect(self.db_path) as conn:
            df_sites.to_sql("sites", conn, if_exists='replace', index=False)
            df_versions.to_sql("site_versions", conn, if_exists='replace', index=False)

        self.version_columns = df_versions.columns.tolist()
        return df_sites, df_versions

    def load_site_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads the sites and site_versions tables from the existing master
        registry without re-downloading.

        Returns (df_sites, df_versions).
        """
        with sqlite3.connect(self.db_path) as conn:
            df_sites = pd.read_sql("SELECT * FROM sites", conn)
            df_versions = pd.read_sql("SELECT * FROM site_versions", conn)

        df_sites['site_id'] = df_sites['site_id'].astype(str)
        df_versions['site_id'] = df_versions['site_id'].astype(str)
        df_versions['version_date'] = pd.to_datetime(df_versions['version_date'], errors='coerce')
        df_versions = df_versions.dropna(subset=['version_date']).sort_values('version_date')
        self.version_columns = df_versions.columns.tolist()
        return df_sites, df_versions

    def get_existing_usids(self, table_name: str) -> set:
        """
        Returns the set of all usid values currently stored in the
        given curated table.  Returns an empty set if the table does
        not yet exist.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT usid FROM {table_name}")
                return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            # Table does not exist yet (first run before bootstrap)
            return set()

    def upsert_curated_data(self, df: pd.DataFrame, table_name: str):
        """
        Inserts new rows into an existing curated table, REPLACING any
        rows whose usid already exists (GLOBE allows measurement amendments).

        Creates the table with a UNIQUE constraint on usid if it does not
        already exist.
        """
        export_df = df.copy()

        if 'version_id' not in export_df.columns:
            export_df['version_id'] = None

        protected_columns = {'version_id', 'site_id'}
        cols_to_drop = [col for col in self.version_columns if
                        col in export_df.columns and col not in protected_columns]
        export_df = export_df.drop(columns=cols_to_drop, errors='ignore')

        with sqlite3.connect(self.db_path) as conn:
            # Ensure the table exists with a UNIQUE constraint on usid
            # by first checking if the table is present
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if cursor.fetchone() is None:
                # Table doesn't exist — create it via pandas, then add the constraint
                export_df.head(0).to_sql(table_name, conn, if_exists='fail', index=False)
                try:
                    conn.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_usid "
                        f"ON {table_name} (usid)"
                    )
                except sqlite3.OperationalError:
                    pass  # Index may already exist

            # Delete existing rows that would conflict, then insert
            usids = export_df['usid'].tolist()
            # Batch delete in chunks of 500 to stay within SQLite variable limits
            for i in range(0, len(usids), 500):
                chunk = usids[i:i + 500]
                placeholders = ','.join('?' * len(chunk))
                conn.execute(
                    f"DELETE FROM {table_name} WHERE usid IN ({placeholders})",
                    chunk
                )

            export_df.to_sql(table_name, conn, if_exists='append', index=False)

        amended = len(usids)
        print(f"  -> Upserted {len(export_df)} records into '{table_name}' "
              f"(overwriting any amended measurements).")

    def load_curated_data(self, table_name: str) -> pd.DataFrame:
        """
        Reads the full curated table from the master registry.
        Used during retrain to reconstruct the complete dataset.
        Returns an empty DataFrame if the table does not exist.
        """
        try:
            query = f"""
                SELECT m.*, v.water_body_source 
                FROM {table_name} m
                LEFT JOIN site_versions v ON m.version_id = v.version_id
            """
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql(query, conn)
        except (sqlite3.OperationalError, pd.io.sql.DatabaseError):
            return pd.DataFrame()

    def merge_inbox_with_sites(
        self,
        df_obs: pd.DataFrame,
        df_versions: pd.DataFrame,
        known_site_ids: set,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs the temporal point-in-time join between new observations
        and the refreshed site registry.

        Observations whose site_id is absent from the known site registry
        are separated out and returned as rejected.

        Returns (df_merged, df_rejected).
        """
        df_obs['site_id'] = df_obs['site_id'].astype(str)
        unknown_mask = ~df_obs['site_id'].isin(known_site_ids)
        df_rejected = df_obs[unknown_mask].copy()
        df_valid = df_obs[~unknown_mask].copy()

        if df_valid.empty:
            return df_valid, df_rejected

        df_valid['measured_on'] = pd.to_datetime(df_valid['measured_on'], errors='coerce')
        df_valid = df_valid.dropna(subset=['measured_on']).sort_values('measured_on')

        df_merged = pd.merge_asof(
            df_valid,
            df_versions,
            left_on='measured_on',
            right_on='version_date',
            by='site_id',
            direction='backward',
            suffixes=('_sample', '_site')
        )

        return df_merged, df_rejected