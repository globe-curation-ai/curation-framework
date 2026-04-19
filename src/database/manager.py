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