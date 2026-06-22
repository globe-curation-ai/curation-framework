"""
Distance to Water Body — Heuristic Metadata Layer

Computes the distance between each GLOBE observation's geolocation and the
nearest natural water body. Uses a two-source verification approach:

1. OpenStreetMap: Queries OSM for water features within a
   configurable radius using the Overpass API via osmnx.
2. Sentinel-2 satellite imagery: Computes the Normalized Difference
   Water Index (NDWI) from cloud-free composites via Google Earth
   Engine. Requires a GEE account and project ID.

By default, both sources are evaluated equally and the pipeline selects
whichever reports the shortest distance to water ("minimum distance wins").

The builder script (build_water_distances.py) pre-computes distances for all
known site coordinates and stores them in a SQLite database committed to the
repo. At pipeline runtime, this module performs a fast lookup from that
database — no API calls are made during curation.
"""

import os
import math
import sqlite3
import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_OSM_RADIUS_M = 300
_DEFAULT_MAX_DISTANCE_M = 1000
_DEFAULT_NDWI_THRESHOLD = 0.3
_DEFAULT_CLOUD_COVER_MAX_PCT = 5
_DEFAULT_SENTINEL_PIXEL_M = 10
_WATER_DB_TABLE = "water_distances"


# ---------------------------------------------------------------------------
# Haversine Distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two geographic points
    using the haversine formula.

    Args:
        lat1, lon1: Coordinates of point 1 in decimal degrees.
        lat2, lon2: Coordinates of point 2 in decimal degrees.

    Returns:
        Distance in meters.
    """
    R = 6_371_000  # Earth radius in meters

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# OSM Water Query
# ---------------------------------------------------------------------------

def query_osm_water(lat: float, lon: float,
                    radius_m: float = _DEFAULT_OSM_RADIUS_M) -> Optional[float]:
    """
    Queries OpenStreetMap for the nearest water feature within `radius_m`
    of (lat, lon) using osmnx.

    Returns the haversine distance in meters to the nearest water geometry
    centroid, or None if no water feature is found.
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import Point
        ox.settings.log_console = False
        # Use an independent Swiss mirror and disable the status slot checker
        ox.settings.overpass_url = "https://overpass.osm.ch/api"
        ox.settings.overpass_rate_limit = False
    except ImportError:
        warnings.warn(
            "osmnx and/or geopandas not installed. "
            "Install with: pip install osmnx geopandas"
        )
        return None

    # Define the water-related OSM tags to query
    water_tags = {
        'natural': ['water', 'wetland', 'bay', 'strait'],
        'waterway': ['river', 'stream', 'canal', 'drain', 'ditch',
                     'riverbank'],
        'water': True,  # Catches all water=* values
        'landuse': ['reservoir', 'basin'],
    }

    try:
        gdf = ox.features_from_point(
            center_point=(lat, lon),
            tags=water_tags,
            dist=radius_m
        )
    except Exception:
        # Overpass may return empty results or timeout — treat as "not found"
        return None

    if gdf.empty:
        return None

    # Compute haversine distance from the query point to each feature centroid
    obs_point = Point(lon, lat)
    min_distance = float('inf')

    for _, feature in gdf.iterrows():
        try:
            centroid = feature.geometry.centroid
            dist = haversine_distance(lat, lon, centroid.y, centroid.x)
            min_distance = min(min_distance, dist)
        except Exception:
            continue

    return min_distance if min_distance < float('inf') else None


# ---------------------------------------------------------------------------
# Sentinel-2 / NDWI Water Query (Google Earth Engine)
# ---------------------------------------------------------------------------

def query_gee_environmental_features(
    lat: float, lon: float,
    obs_date_str: Optional[str] = None,
    max_distance_m: float = _DEFAULT_MAX_DISTANCE_M,
    ndwi_threshold: float = _DEFAULT_NDWI_THRESHOLD,
    cloud_cover_max_pct: float = _DEFAULT_CLOUD_COVER_MAX_PCT,
    pixel_size_m: float = _DEFAULT_SENTINEL_PIXEL_M,
) -> dict:
    """
    Queries Google Earth Engine (GEE) for environmental metadata around a specific geographic point.
    
    This function extracts three main features:
    1. Land Cover: Extracted from ESA WorldCover 2021 (10m resolution).
    2. Precipitation: 14-day cumulative rainfall prior to the observation date using CHIRPS Daily.
    3. Distance to Water:
       - Uses JRC Monthly Water History for observations in or before 2021.
       - Falls back to calculating the Normalized Difference Water Index (NDWI) from 
         Sentinel-2 surface reflectance composites (±30 days) for newer observations.
       - Computes the Euclidean distance from the target point to the nearest detected water pixel
         using GEE's fastDistanceTransform.

    Args:
        lat, lon: The target coordinates in decimal degrees.
        obs_date_str: ISO format date string of the observation (e.g., 'YYYY-MM-DD').
        max_distance_m: Maximum search radius for water detection.
        ndwi_threshold: Threshold above which a pixel is considered water (for Sentinel-2).
        cloud_cover_max_pct: Maximum allowed cloud cover for Sentinel-2 composites.
        pixel_size_m: The spatial resolution used for the distance transform reducer.

    Returns:
        dict: A dictionary containing:
            - 'distance_m' (float | None): Distance to the nearest water body in meters.
            - 'land_cover' (int | None): ESA WorldCover classification code.
            - 'precip_14d_mm' (float | None): Total precipitation in the preceding 14 days.
    """
    result = {'distance_m': None, 'land_cover': None, 'precip_14d_mm': None}
    try:
        import ee
    except ImportError:
        return result

    try:
        point = ee.Geometry.Point([lon, lat])
        bbox = point.buffer(max_distance_m).bounds()
        
        # 1. LAND COVER (ESA WorldCover 2021)
        try:
            worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
            lc_val = worldcover.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=10
            ).get('Map').getInfo()
            if lc_val is not None:
                result['land_cover'] = int(lc_val)
        except Exception:
            pass

        # 2. PRECIPITATION (CHIRPS Daily) and WATER DISTANCE
        water_mask = None
        if obs_date_str:
            import pandas as pd
            from datetime import datetime
            obs_date_str = obs_date_str.split('T')[0]
            dt = datetime.fromisoformat(obs_date_str)
            year = dt.year
            month = dt.month

            # Precipitation (14 days prior to obs_date)
            try:
                start_precip = (dt - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
                end_precip = dt.strftime('%Y-%m-%d')
                chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")\
                           .filterDate(start_precip, end_precip)\
                           .sum()
                precip_val = chirps.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=5566
                ).get('precipitation').getInfo()
                if precip_val is not None:
                    result['precip_14d_mm'] = float(precip_val)
            except Exception:
                pass

            # Water Distance
            if year <= 2021:
                jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
                jrc_month = jrc.filter(ee.Filter.calendarRange(year, year, 'year'))\
                               .filter(ee.Filter.calendarRange(month, month, 'month')).first()
                if jrc_month.getInfo() is not None:
                    water_mask = jrc_month.eq(2)
            
            if water_mask is None:
                start_date = (dt - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = (dt + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                s2 = (
                    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(bbox)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max_pct))
                    .median()
                )
                ndwi = s2.normalizedDifference(['B3', 'B8']).rename('ndwi')
                water_mask = ndwi.gt(ndwi_threshold)
        else:
            s2 = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(bbox)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max_pct))
                .median()
            )
            ndwi = s2.normalizedDifference(['B3', 'B8']).rename('ndwi')
            water_mask = ndwi.gt(ndwi_threshold)

        if water_mask is not None:
            water_count = water_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=bbox,
                scale=int(pixel_size_m),
                maxPixels=1e6
            ).getInfo()

            if water_count and list(water_count.values())[0] > 0:
                distance_img = (
                    water_mask.Not()
                    .fastDistanceTransform(256)
                    .sqrt()
                    .multiply(pixel_size_m)
                )
                distance_at_point = distance_img.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=int(pixel_size_m)
                ).getInfo()

                dist_val = list(distance_at_point.values())[0] if distance_at_point else None
                if dist_val is not None and dist_val >= 0:
                    result['distance_m'] = float(dist_val)

        return result
    except Exception:
        return result



# ---------------------------------------------------------------------------
# Earth Engine Initialization
# ---------------------------------------------------------------------------

def init_earth_engine(config: dict) -> bool:
    """
    Attempts to initialize the Google Earth Engine client.

    Uses one of:
      - Service account credentials via GEE_SERVICE_ACCOUNT_KEY env var
      - Interactive OAuth token (from ~/.config/earthengine/)

    Returns True if initialization succeeded, False otherwise.
    """
    dtw_config = config.get('distance_to_water', {})
    project = dtw_config.get('gee_project')

    try:
        import ee
    except ImportError:
        warnings.warn("earthengine-api not installed. GEE fallback disabled.")
        return False

    try:
        key_path = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
        if key_path and os.path.exists(key_path):
            credentials = ee.ServiceAccountCredentials('', key_file=key_path)
            if project:
                ee.Initialize(credentials=credentials, project=project)
            else:
                ee.Initialize(credentials=credentials)
        else:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
        return True
    except Exception as e:
        warnings.warn(f"Earth Engine initialization failed: {e}. "
                      f"GEE fallback disabled.")
        return False


# ---------------------------------------------------------------------------
# Single-Location Distance Computation (used by build_water_distances.py)
# ---------------------------------------------------------------------------

def compute_distance_for_location(
    lat: float, lon: float,
    config: dict,
    gee_available: bool = False,
    obs_date: Optional[str] = None,
) -> dict:
    """
    Computes the distance from (lat, lon) to the nearest water body.

    Returns a dict with:
        - distance_osm_m (float): Raw OSM distance (max_distance if not found)
        - distance_gee_m (float): Raw GEE distance (max_distance if not found)
        - distance_combined_m (float): min(osm, gee)
        - distance_combined_source (str): 'osm', 'gee', or 'none'
        - land_cover_class (int|None): ESA WorldCover class from GEE
        - precip_14d_mm (float|None): 14-day precipitation sum from GEE
    """
    dtw_config = config.get('distance_to_water', {})
    max_distance = dtw_config.get('max_distance_m', _DEFAULT_MAX_DISTANCE_M)
    osm_radius = dtw_config.get('osm_radius_m', _DEFAULT_OSM_RADIUS_M)

    result = {
        'distance_osm_m': float(max_distance),
        'distance_gee_m': float(max_distance),
        'distance_combined_m': float(max_distance),
        'distance_combined_source': 'none',
        'land_cover_class': None,
        'precip_14d_mm': None,
    }

    # --- Source 1: OpenStreetMap ---
    osm_val = query_osm_water(lat, lon, radius_m=osm_radius)
    if osm_val is not None and osm_val < max_distance:
        result['distance_osm_m'] = round(osm_val, 2)

    # --- Source 2: Google Earth Engine ---
    if gee_available:
        gee_features = query_gee_environmental_features(
            lat, lon,
            obs_date_str=obs_date,
            max_distance_m=max_distance,
            ndwi_threshold=dtw_config.get('ndwi_threshold',
                                          _DEFAULT_NDWI_THRESHOLD),
            cloud_cover_max_pct=dtw_config.get('cloud_cover_max_pct',
                                               _DEFAULT_CLOUD_COVER_MAX_PCT),
            pixel_size_m=dtw_config.get('sentinel_pixel_m',
                                        _DEFAULT_SENTINEL_PIXEL_M),
        )

        result['land_cover_class'] = gee_features.get('land_cover')
        result['precip_14d_mm'] = gee_features.get('precip_14d_mm')

        gee_val = gee_features.get('distance_m')
        if gee_val is not None and gee_val < max_distance:
            result['distance_gee_m'] = round(gee_val, 2)

    # --- The Contest (Minimum Distance Wins) ---
    osm_d = result['distance_osm_m']
    gee_d = result['distance_gee_m']

    if osm_d < max_distance and gee_d < max_distance:
        if osm_d <= gee_d:
            result['distance_combined_m'] = osm_d
            result['distance_combined_source'] = 'osm'
        else:
            result['distance_combined_m'] = gee_d
            result['distance_combined_source'] = 'gee'
    elif osm_d < max_distance:
        result['distance_combined_m'] = osm_d
        result['distance_combined_source'] = 'osm'
    elif gee_d < max_distance:
        result['distance_combined_m'] = gee_d
        result['distance_combined_source'] = 'gee'

    return result



# ---------------------------------------------------------------------------
# Pre-computed Database I/O
# ---------------------------------------------------------------------------

def load_water_distance_db(db_path: str) -> pd.DataFrame:
    """
    Loads the pre-computed water distance database from SQLite.
    Returns an empty DataFrame with the correct schema if the file
    doesn't exist.
    """
    empty = pd.DataFrame(columns=[
        'site_id', 'version_date', 'latitude', 'longitude',
        'distance_osm_m', 'distance_gee_m',
        'distance_combined_m', 'distance_combined_source',
        'land_cover_class', 'precip_14d_mm', 'computed_at'
    ])

    if not os.path.exists(db_path):
        warnings.warn(
            f"Water distance database not found at {db_path}. "
            f"Run build_water_distances.py to generate it."
        )
        return empty

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {_WATER_DB_TABLE}", conn)
        df['version_date'] = pd.to_datetime(df['version_date'],
                                            errors='coerce', format='mixed', utc=True).dt.tz_localize(None)
        return df
    except Exception as e:
        warnings.warn(f"Failed to load water distance database: {e}")
        return empty


def save_water_distance_db(df: pd.DataFrame, db_path: str) -> None:
    """
    Saves the computed water distances to a SQLite database.
    Creates the database and directory if they don't exist.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Normalize datetime columns to strings to prevent SQLite binding errors
    df_save = df.copy()
    for col in ['version_date', 'computed_at']:
        if col in df_save.columns:
            df_save[col] = pd.to_datetime(df_save[col], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
            df_save[col] = df_save[col].where(pd.notnull(df_save[col]), None)

    with sqlite3.connect(db_path) as conn:
        df_save.to_sql(_WATER_DB_TABLE, conn, if_exists='replace', index=False)
    print(f"  -> Water distance database saved to {db_path} "
          f"({len(df_save)} records)")


# ---------------------------------------------------------------------------
# Pipeline Integration — Lookup from Pre-computed DB
# ---------------------------------------------------------------------------

def compute_water_distances(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Enriches the pipeline DataFrame with pre-computed water distance metadata.

    Performs a temporal merge_asof on (site_id, version_date) to attach the
    correct water distance for each observation based on the site version
    that was active at measurement time.

    The ``distance_method`` config option (default: ``combined``) controls
    which distance column is surfaced as the canonical distance:
        - ``combined``: min(OSM, GEE) — the "minimum distance wins" contest
        - ``osm_first``: OSM-first, with GEE as a fallback if OSM fails
        - ``gee_first``: GEE-first, with OSM as a fallback if GEE fails
        - ``osm_only``: OpenStreetMap vector database only
        - ``gee_only``: Google Earth Engine satellite imagery only

    Adds these columns:
        - distance_from_water_m (float): Distance in meters (selected method)
        - distance_from_water_source (str): 'osm', 'gee', or 'none'
        - water_detected (bool): True if water was found within max_distance_m
        - land_cover_class (int|None): ESA WorldCover class
        - precip_14d_mm (float|None): 14-day precipitation sum

    If the pre-computed database is missing, columns are added with NaN/default
    values and a warning is emitted.
    """
    dtw_config = config.get('distance_to_water', {})

    if not dtw_config.get('enabled', False):
        return df

    db_path = dtw_config.get('database', 'data/water_distances.sqlite')
    max_distance = dtw_config.get('max_distance_m', _DEFAULT_MAX_DISTANCE_M)
    distance_method = dtw_config.get('distance_method', 'combined')

    print(f"  -> Loading pre-computed water distance database "
          f"(method: {distance_method})...")
    df_water = load_water_distance_db(db_path)

    if df_water.empty:
        print("  -> WARNING: No pre-computed water distances available. "
              "Adding default columns.")
        df['distance_from_water_m'] = float(max_distance)
        df['distance_from_water_source'] = 'none'
        df['water_detected'] = False
        df['land_cover_class'] = None
        df['precip_14d_mm'] = None
        return df



    # --- Select the active distance column based on method ---
    if distance_method == 'osm_first':
        # OSM-first, GEE failover
        df_water['distance_computed_m'] = np.where(
            df_water['distance_osm_m'] < max_distance,
            df_water['distance_osm_m'],
            df_water['distance_gee_m']
        )
        df_water['distance_computed_source'] = np.where(
            df_water['distance_osm_m'] < max_distance,
            'osm',
            np.where(df_water['distance_gee_m'] < max_distance, 'gee', 'none')
        )
        dist_col = 'distance_computed_m'
        src_val = None  # will use distance_computed_source
    elif distance_method == 'gee_first':
        # GEE-first, OSM failover
        df_water['distance_computed_m'] = np.where(
            df_water['distance_gee_m'] < max_distance,
            df_water['distance_gee_m'],
            df_water['distance_osm_m']
        )
        df_water['distance_computed_source'] = np.where(
            df_water['distance_gee_m'] < max_distance,
            'gee',
            np.where(df_water['distance_osm_m'] < max_distance, 'osm', 'none')
        )
        dist_col = 'distance_computed_m'
        src_val = None  # will use distance_computed_source
    elif distance_method == 'osm_only':
        # OSM only
        dist_col = 'distance_osm_m'
        src_val = 'osm'
    elif distance_method == 'gee_only':
        # GEE only
        dist_col = 'distance_gee_m'
        src_val = 'gee'
    else:  # 'combined' (default)
        dist_col = 'distance_combined_m'
        src_val = None  # will use distance_combined_source

    # Build a clean lookup table with standardized output column names
    df_water = df_water.sort_values('version_date')
    df_water_lookup = df_water[['site_id', 'version_date',
                                dist_col, 'land_cover_class',
                                'precip_14d_mm']].copy()

    # Map the chosen column to the canonical output name
    df_water_lookup['distance_from_water_m'] = df_water_lookup[dist_col]

    if src_val is not None:
        df_water_lookup['distance_from_water_source'] = np.where(
            df_water_lookup['distance_from_water_m'] < max_distance,
            src_val,
            'none'
        )
    else:
        # For combined or osm_first modes, use the dynamically computed source
        source_col = 'distance_combined_source' if distance_method == 'combined' else 'distance_computed_source'
        df_water_lookup['distance_from_water_source'] = (
            df_water[source_col].values
        )

    df_water_lookup['water_detected'] = (
        df_water_lookup['distance_from_water_m'] < max_distance
    )

    # Drop the raw distance column to avoid duplication
    df_water_lookup.drop(columns=[dist_col], inplace=True)

    # Prepare the observations for merge
    df_result = df.copy()

    # Ensure measured_on is datetime for merge_asof
    date_col = 'measured_on'
    if date_col in df_result.columns:
        df_result[date_col] = pd.to_datetime(df_result[date_col],
                                             errors='coerce', format='mixed', utc=True).dt.tz_localize(None)
        df_result = df_result.sort_values(date_col)

        # Pass 1: Standard backward temporal merge (point-in-time)
        # e.g., A 2014 observation maps to a 2010 site version, not a 2015 one.
        df_back = pd.merge_asof(
            df_result,
            df_water_lookup,
            left_on=date_col,
            right_on='version_date',
            by='site_id',
            direction='backward',
            suffixes=('', '_water')
        )
        
        # Pass 2: Forward temporal merge (fallback for back-filled data)
        # e.g., A 1995 observation for a site registered in 1996 will fall back
        # to the 1996 version, rather than dropping to NaN.
        df_fwd = pd.merge_asof(
            df_result,
            df_water_lookup,
            left_on=date_col,
            right_on='version_date',
            by='site_id',
            direction='forward',
            suffixes=('', '_water')
        )
        
        # Combine: fill missing values in the backward merge with the forward merge
        for col in df_water_lookup.columns:
            if col != 'site_id':
                col_name = f"{col}_water" if col in df_result.columns else col
                df_back[col_name] = df_back[col_name].fillna(df_fwd[col_name])
                
        df_result = df_back
    else:
        # Fallback: simple merge on site_id (take most recent version)
        latest_water = (df_water_lookup
                        .sort_values('version_date')
                        .drop_duplicates(subset='site_id', keep='last')
                        .drop(columns='version_date', errors='ignore'))
        df_result = df_result.merge(latest_water, on='site_id', how='left')

    # Fill unmatched sites with defaults
    df_result['distance_from_water_m'] = (
        df_result['distance_from_water_m'].fillna(float(max_distance))
    )
    df_result['distance_from_water_source'] = (
        df_result['distance_from_water_source'].fillna('none')
    )
    df_result['water_detected'] = (
        df_result['water_detected'].fillna(False).astype(bool)
    )

    # Drop the merge key artifact if present
    if 'version_date_water' in df_result.columns:
        df_result.drop(columns='version_date_water', inplace=True)

    matched = df_result['distance_from_water_source'].ne('none').sum()
    total = len(df_result)
    print(f"  -> Water distance enrichment complete: "
          f"{matched}/{total} observations matched "
          f"(method: {distance_method}).")

    return df_result

