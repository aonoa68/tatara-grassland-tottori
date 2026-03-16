# -*- coding: utf-8 -*-
"""grass_tatara_evolutionary_extensions.py

# Grass–Tatara Evolutionary Anthropology Extensions

This notebook extends the existing grass–tatara analysis with additional analyses
from the perspectives of evolutionary anthropology, cultural evolution, and niche
construction theory.

## Analyses included
1. Generate slope and curvature from DEM
2. Aggregate terrain indicators at the municipality level
3. Grassland distribution model including terrain variables
4. Interaction terms: tatara × slope and tatara × geology
5. DBSCAN spatial clustering of tatara point data
6. Environmental summary by cluster
7. Template for time-series comparison: 1950 / 1965 / 1975
8. Template for adding historical population / density
9. Classification analysis including terrain and population variables
10. Final integrated model from an evolutionary anthropology perspective

## Theoretical background
The additional analyses in this notebook are motivated by the following questions:

- Do environmental conditions constrain cultural activities?
- Do cultural activities modify landscapes?
- Is this relationship condition-dependent?
- Is there evidence of cultural clustering?
- Does population density relate to industrial specialization?

## Prerequisites
- Compatible with `agg`, `gdf_muni`, `gdf_sites`, `EXTRACT_DIR`, and `DEM_PATH`
  from the base notebook
- Assumes all input data are stored on Google Drive
"""

# Uncomment and run first if packages are not installed
# !pip -q install geopandas rasterio rasterstats scipy scikit-learn statsmodels

import os
import warnings
from math import log

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

import rasterio
from rasterstats import zonal_stats
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.max_columns", 300)
pd.set_option("display.max_rows", 200)

from google.colab import drive
drive.mount('/content/drive')

EXTRACT_DIR = "/content/drive/MyDrive/鳥大地理情報研卒論2025/B22A5163M_BITO"

CSV_PATH = os.path.join(
    EXTRACT_DIR,
    "02_世界農業センサス_公私有牧野統計表CSV",
    "鳥取県",
    "世界農業センサス",
    "1950",
    "世界農業センサス＋たたら",
    "tottori20251216.csv"
)

FGB_PATH = os.path.join(
    EXTRACT_DIR,
    "06_混合効果モデル",
    "「流域」「地質」に基づくグループ分け",
    "鳥取県",
    "結果",
    "3guru-pu",
    "「流域」「地質」",
    "basin_geology_group",
    "admin_grass_tatara_basin_geology.fgb"
)

TATARA_POINT_CSV = os.path.join(
    EXTRACT_DIR,
    "04_たたら遺構位置情報CSV",
    "鳥取県",
    "tottoritataraitizyouhou20251228.csv"
)

DEM_PATH = "/content/drive/MyDrive/鳥大地理情報研卒論2025/add_analysis/tottori_Z13DEM_6673.tif"

print("CSV:", os.path.exists(CSV_PATH))
print("FGB:", os.path.exists(FGB_PATH))
print("Tatara CSV:", os.path.exists(TATARA_POINT_CSV))
print("DEM:", os.path.exists(DEM_PATH))

# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------

def read_csv_multi_encoding(path, encodings=("utf-8-sig", "utf-8", "cp932", "shift_jis")):
    """Try multiple encodings to read a CSV file."""
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print("CSV encoding:", enc)
            return df
        except Exception as e:
            last_error = e
    raise last_error

def pick_first_existing(columns, candidates):
    """Return the first candidate column name that exists in the given column list."""
    for c in candidates:
        if c in columns:
            return c
    return None

def canonicalize_main_df(df_raw):
    """Standardise column names in the main census DataFrame."""
    muni_col = pick_first_existing(df_raw.columns, [
        "municipality", "N03_004", "市町村名", "世界農業センサス1950 - 1950年csv用整形データ_結合用都市町村"
    ])
    grass_col = pick_first_existing(df_raw.columns, [
        "grass", "世界農業センサス1950 - 1950年csv用整形データ_採草地・放牧地・貸付地面積"
    ])
    count_col = pick_first_existing(df_raw.columns, ["count", "No_count"])
    assert muni_col and grass_col and count_col, "Required columns not found"

    out = df_raw.copy().rename(columns={
        muni_col: "municipality",
        grass_col: "grass",
        count_col: "count"
    })
    out["grass"] = pd.to_numeric(out["grass"], errors="coerce")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    return out.dropna(subset=["municipality", "grass", "count"]).copy()

def canonicalize_geo_gdf(gdf_raw):
    """Standardise column names in the spatial GeoDataFrame."""
    muni_col = pick_first_existing(gdf_raw.columns, [
        "municipality", "N03_004", "市町村名", "世界農業センサス1950 - 1950年csv用整形データ_結合用都市町村"
    ])
    basin_col = pick_first_existing(gdf_raw.columns, ["basin", "W07_004"])
    geology_col = pick_first_existing(gdf_raw.columns, ["geology", "legend_group_ja", "legend_lithology_ja"])
    rename_map = {}
    if muni_col: rename_map[muni_col] = "municipality"
    if basin_col: rename_map[basin_col] = "basin"
    if geology_col: rename_map[geology_col] = "geology"
    gdf = gdf_raw.copy().rename(columns=rename_map)
    keep = [c for c in ["municipality", "basin", "geology", "geometry"] if c in gdf.columns]
    return gdf[keep].dropna(subset=["municipality", "geometry"]).copy()

def safe_area(gdf):
    """Compute polygon area, falling back to alternative CRS if needed."""
    try:
        return gdf.to_crs(6677).geometry.area
    except Exception:
        try:
            return gdf.to_crs(3857).geometry.area
        except Exception:
            return pd.Series(np.nan, index=gdf.index)

def entropy_from_shares(shares):
    """Compute normalised Shannon entropy from a list of area shares."""
    shares = [s for s in shares if s > 0]
    if len(shares) <= 1:
        return 0.0
    H = -sum(s * log(s) for s in shares)
    Hmax = log(len(shares))
    return H / Hmax if Hmax > 0 else 0.0

def rock_family(label):
    """Classify a Japanese geological label into a broad rock family."""
    if pd.isna(label):
        return "unknown"
    s = str(label)
    if any(k in s for k in ["火成", "花崗", "安山岩", "玄武岩", "流紋岩", "閃緑"]):
        return "igneous"
    if any(k in s for k in ["堆積", "砂岩", "泥岩", "頁岩", "石灰岩", "礫岩"]):
        return "sedimentary"
    if any(k in s for k in ["変成", "片岩", "片麻岩", "結晶片岩", "ホルンフェルス"]):
        return "metamorphic"
    return "other"

def canonicalize_tatara_point_csv(df_raw):
    """Standardise and convert a tatara site CSV to a GeoDataFrame."""
    lon_col = pick_first_existing(df_raw.columns, ["経度", "lon", "longitude", "x"])
    lat_col = pick_first_existing(df_raw.columns, ["緯度", "lat", "latitude", "y"])
    assert lon_col and lat_col, "Longitude/latitude columns not found"

    out = df_raw.copy().rename(columns={lon_col: "lon", lat_col: "lat"})
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out = out.dropna(subset=["lon", "lat"]).copy()
    return gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["lon"], out["lat"]), crs="EPSG:4326")


# ================================================================
# 1. Reconstruct base data
#    Recreate agg, gdf_muni, and gdf_sites from source files.
# ================================================================

df = canonicalize_main_df(read_csv_multi_encoding(CSV_PATH))
gdf = canonicalize_geo_gdf(gpd.read_file(FGB_PATH))
gdf["poly_area"] = safe_area(gdf).fillna(0)

# Municipality polygons
gdf_muni = gdf[["municipality", "geometry"]].dissolve(by="municipality").reset_index()
if gdf_muni.crs is None:
    gdf_muni = gdf_muni.set_crs(gdf.crs)
gdf_muni = gdf_muni.to_crs(6677)

# Dominant basin per municipality
if "basin" in gdf.columns:
    basin_area = gdf.groupby(["municipality", "basin"], dropna=False)["poly_area"].sum().reset_index()
    basin_main = basin_area.sort_values(["municipality", "poly_area"], ascending=[True, False]).drop_duplicates("municipality")
    basin_main = basin_main.rename(columns={"basin": "basin_main"})
else:
    basin_main = pd.DataFrame({"municipality": gdf["municipality"].unique(), "basin_main": "NA"})

# Geology summary: entropy and rock-family shares
if "geology" in gdf.columns:
    geo_area = gdf.groupby(["municipality", "geology"], dropna=False)["poly_area"].sum().reset_index()
    total_geo = geo_area.groupby("municipality")["poly_area"].sum().rename("total_geo_area").reset_index()
    geo_area = geo_area.merge(total_geo, on="municipality", how="left")
    geo_area["share"] = np.where(geo_area["total_geo_area"] > 0, geo_area["poly_area"] / geo_area["total_geo_area"], np.nan)
    geo_area["rock_family"] = geo_area["geology"].map(rock_family)

    entropy_tbl = geo_area.groupby("municipality")["share"].apply(
        lambda s: entropy_from_shares(s.dropna().tolist())
    ).rename("geology_entropy").reset_index()

    fam_tbl = geo_area.groupby(["municipality", "rock_family"])["share"].sum().reset_index().pivot(
        index="municipality", columns="rock_family", values="share"
    ).fillna(0).reset_index()

    for col in ["igneous", "sedimentary", "metamorphic", "other", "unknown"]:
        if col not in fam_tbl.columns:
            fam_tbl[col] = 0.0

    fam_tbl = fam_tbl.rename(columns={
        "igneous": "igneous_share",
        "sedimentary": "sedimentary_share",
        "metamorphic": "metamorphic_share",
        "other": "other_rock_share",
        "unknown": "unknown_rock_share",
    })
else:
    entropy_tbl = pd.DataFrame({"municipality": gdf["municipality"].unique(), "geology_entropy": np.nan})
    fam_tbl = pd.DataFrame({"municipality": gdf["municipality"].unique()})

# Basin entropy
if "basin" in gdf.columns:
    basin_area2 = gdf.groupby(["municipality", "basin"], dropna=False)["poly_area"].sum().reset_index()
    total_basin = basin_area2.groupby("municipality")["poly_area"].sum().rename("total_basin_area").reset_index()
    basin_area2 = basin_area2.merge(total_basin, on="municipality", how="left")
    basin_area2["share"] = np.where(basin_area2["total_basin_area"] > 0, basin_area2["poly_area"] / basin_area2["total_basin_area"], np.nan)
    basin_entropy_tbl = basin_area2.groupby("municipality")["share"].apply(
        lambda s: entropy_from_shares(s.dropna().tolist())
    ).rename("basin_entropy").reset_index()
else:
    basin_entropy_tbl = pd.DataFrame({"municipality": gdf["municipality"].unique(), "basin_entropy": np.nan})

# Merge all tables
agg = (
    df.merge(basin_main[["municipality", "basin_main"]], on="municipality", how="left")
      .merge(entropy_tbl, on="municipality", how="left")
      .merge(fam_tbl, on="municipality", how="left")
      .merge(basin_entropy_tbl, on="municipality", how="left")
)

for c in ["geology_entropy", "basin_entropy", "igneous_share", "sedimentary_share", "metamorphic_share",
          "other_rock_share", "unknown_rock_share"]:
    if c in agg.columns:
        agg[c] = agg[c].fillna(0)

# Dominant geology group per municipality
if "geology" in gdf.columns:
    geo_area_main = gdf.groupby(["municipality", "geology"], dropna=False)["poly_area"].sum().reset_index()
    geo_main = geo_area_main.sort_values(["municipality", "poly_area"], ascending=[True, False]).drop_duplicates("municipality")
    geo_main = geo_main.rename(columns={"geology": "geology_main"})
    agg = agg.merge(geo_main[["municipality", "geology_main"]], on="municipality", how="left")
    vc = agg["geology_main"].astype(str).value_counts()
    top_levels = set(vc.head(6).index)
    agg["geology_main_grp"] = agg["geology_main"].astype(str).where(
        agg["geology_main"].astype(str).isin(top_levels), "Other"
    )
else:
    agg["geology_main_grp"] = "NA"

# Tatara site point data
gdf_sites = canonicalize_tatara_point_csv(read_csv_multi_encoding(TATARA_POINT_CSV)).to_crs(6677)

display(agg.head())
print("agg shape:", agg.shape)
print("gdf_sites shape:", gdf_sites.shape)


# ================================================================
# 2. Generate slope and curvature from DEM
# ================================================================

SLOPE_TIF = os.path.join(EXTRACT_DIR, "DEM", "tottori_slope.tif")
CURV_TIF  = os.path.join(EXTRACT_DIR, "DEM", "tottori_curvature.tif")

def make_dem_derivatives(dem_path, slope_tif, curv_tif, smooth_sigma=1.0):
    """Derive slope (degrees) and curvature rasters from a DEM."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).astype("float64")
        profile = src.profile.copy()
        transform = src.transform

        dx = transform.a
        dy = abs(transform.e)

        arr = dem.filled(np.nan)
        arr_s = gaussian_filter(np.nan_to_num(arr, nan=np.nanmedian(arr)), sigma=smooth_sigma)

        dz_dy, dz_dx = np.gradient(arr_s, dy, dx)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        d2z_dx2 = np.gradient(dz_dx, dx, axis=1)
        d2z_dy2 = np.gradient(dz_dy, dy, axis=0)
        curvature = d2z_dx2 + d2z_dy2

        profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)

        with rasterio.open(slope_tif, "w", **profile) as dst:
            dst.write(slope_deg.astype("float32"), 1)

        with rasterio.open(curv_tif, "w", **profile) as dst:
            dst.write(curvature.astype("float32"), 1)

    print("saved:", slope_tif)
    print("saved:", curv_tif)

if os.path.exists(DEM_PATH):
    make_dem_derivatives(DEM_PATH, SLOPE_TIF, CURV_TIF)
else:
    print("DEM not found; skip derivative generation.")


# ================================================================
# 3. Aggregate terrain indicators at the municipality level
# ================================================================

for c in ["elev_mean", "elev_sd", "elev_min", "elev_max", "slope_mean", "slope_sd", "slope_max", "curv_mean", "curv_sd"]:
    if c not in agg.columns:
        agg[c] = np.nan

if os.path.exists(DEM_PATH):
    with rasterio.open(DEM_PATH) as src:
        dem_crs = src.crs

    gdf_muni_dem = gdf_muni.to_crs(dem_crs)

    dem_zs   = zonal_stats(gdf_muni_dem, DEM_PATH,   stats=["mean", "std", "min", "max"], nodata=np.nan)
    slope_zs = zonal_stats(gdf_muni_dem, SLOPE_TIF,  stats=["mean", "std", "max"], nodata=np.nan) if os.path.exists(SLOPE_TIF) else None
    curv_zs  = zonal_stats(gdf_muni_dem, CURV_TIF,   stats=["mean", "std"],        nodata=np.nan) if os.path.exists(CURV_TIF)  else None

    dem_df   = pd.DataFrame(dem_zs).rename(columns={"mean":"elev_mean","std":"elev_sd","min":"elev_min","max":"elev_max"})
    slope_df = pd.DataFrame(slope_zs).rename(columns={"mean":"slope_mean","std":"slope_sd","max":"slope_max"}) if slope_zs is not None else pd.DataFrame()
    curv_df  = pd.DataFrame(curv_zs).rename(columns={"mean":"curv_mean","std":"curv_sd"})                       if curv_zs  is not None else pd.DataFrame()

    terrain_df = pd.concat(
        [gdf_muni[["municipality"]].reset_index(drop=True),
         dem_df.reset_index(drop=True),
         slope_df.reset_index(drop=True),
         curv_df.reset_index(drop=True)],
        axis=1
    )

    agg = agg.drop(columns=["elev_mean", "elev_sd", "elev_min", "elev_max",
                             "slope_mean", "slope_sd", "slope_max", "curv_mean", "curv_sd"], errors="ignore")
    agg = agg.merge(terrain_df, on="municipality", how="left")

display(agg[["municipality", "elev_mean", "slope_mean", "slope_sd", "curv_mean"]].head())


# ================================================================
# 4. Grassland distribution model including terrain variables
# ================================================================

m_terrain_1 = smf.ols(
    '''
    grass ~ geology_entropy
          + igneous_share
          + sedimentary_share
          + metamorphic_share
          + basin_entropy
          + elev_mean
          + elev_sd
          + slope_mean
          + slope_sd
          + curv_mean
    ''',
    data=agg
).fit()

m_terrain_2 = smf.ols(
    '''
    grass ~ geology_entropy
          + igneous_share
          + sedimentary_share
          + metamorphic_share
          + basin_entropy
          + elev_mean
          + elev_sd
          + slope_mean
          + slope_sd
          + curv_mean
          + count
    ''',
    data=agg
).fit()

cmp_terrain = pd.DataFrame({
    "model": ["terrain_base", "terrain_plus_tatara"],
    "AIC": [m_terrain_1.aic, m_terrain_2.aic],
    "R2": [m_terrain_1.rsquared, m_terrain_2.rsquared]
})

display(cmp_terrain)
print(m_terrain_2.summary())


# ================================================================
# 5. Interaction terms: tatara × slope and tatara × geology
#
#    From an evolutionary anthropology perspective, these
#    interactions capture the condition-dependence of cultural
#    adaptation — i.e., whether the effect of iron production
#    on grassland area is moderated by local terrain or geology.
# ================================================================

m_inter_slope = smf.ols(
    '''
    grass ~ geology_entropy
          + basin_entropy
          + elev_mean
          + slope_mean
          + curv_mean
          + count
          + count:slope_mean
    ''',
    data=agg
).fit()

print("=== tatara x slope ===")
print(m_inter_slope.summary())

m_inter_geo = smf.ols(
    '''
    grass ~ geology_entropy
          + basin_entropy
          + elev_mean
          + slope_mean
          + curv_mean
          + count
          + count:geology_entropy
    ''',
    data=agg
).fit()

print("=== tatara x geology_entropy ===")
print(m_inter_geo.summary())


# ================================================================
# 6. DBSCAN spatial clustering of tatara point data
#
#    Tests for cultural clustering — geographically concentrated
#    iron production districts rather than a random distribution.
#    eps and min_samples should be tuned to the study region.
# ================================================================

pts = gdf_sites.to_crs(6677).copy()
coords = np.array([(geom.x, geom.y) for geom in pts.geometry])

db = DBSCAN(eps=3000, min_samples=5)
labels = db.fit_predict(coords)
pts["cluster_id"] = labels

print(pts["cluster_id"].value_counts(dropna=False).sort_index())

cluster_summary = (
    pts[pts["cluster_id"] >= 0]
    .groupby("cluster_id")
    .agg(
        n_sites=("cluster_id", "size"),
        mean_x=("geometry", lambda s: np.mean([g.x for g in s])),
        mean_y=("geometry", lambda s: np.mean([g.y for g in s])),
    )
    .reset_index()
    .sort_values("n_sites", ascending=False)
)

display(cluster_summary)

fig, ax = plt.subplots(figsize=(8, 8))
gdf_muni.boundary.plot(ax=ax, color="gray", linewidth=0.5)
pts.plot(column="cluster_id", ax=ax, categorical=True, legend=True, markersize=8)
ax.set_title("Tatara site clusters (DBSCAN)")
ax.axis("off")
plt.show()


# ================================================================
# 7. Environmental summary by cluster
# ================================================================

cluster_hulls = (
    pts[pts["cluster_id"] >= 0]
    .dissolve(by="cluster_id")
    .convex_hull
)

cluster_gdf = gpd.GeoDataFrame(
    {"cluster_id": cluster_hulls.index},
    geometry=cluster_hulls.values,
    crs=pts.crs
)

muni_env = gdf_muni.merge(
    agg[["municipality", "grass", "count", "basin_main", "geology_main_grp", "elev_mean", "slope_mean", "curv_mean"]],
    on="municipality",
    how="left"
)

cluster_env = gpd.overlay(cluster_gdf, muni_env, how="intersection")

cluster_env_summary = (
    cluster_env.groupby("cluster_id")
    .agg(
        n_muni=("municipality", "nunique"),
        grass_mean=("grass", "mean"),
        tatara_mean=("count", "mean"),
        elev_mean=("elev_mean", "mean"),
        slope_mean=("slope_mean", "mean"),
        curv_mean=("curv_mean", "mean"),
    )
    .reset_index()
)

display(cluster_env_summary)


# ================================================================
# 8. Time-series comparison: 1950 / 1960 / 1975
#    Fill in the CSV paths below and the block runs as-is.
# ================================================================

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ----------------------------
# 1. Paths to uploaded CSVs
# ----------------------------
CSV_1950 = "/content/drive/MyDrive/鳥大地理情報研卒論2025/B22A5163M_BITO/02_世界農業センサス_公私有牧野統計表CSV/鳥取県/世界農業センサス/1950/世界農業センサス1950 - 1950年csv用整形データ/世界農業センサス1950 - 1950年csv用整形データ.csv"
CSV_1960 = "/content/drive/MyDrive/鳥大地理情報研卒論2025/B22A5163M_BITO/02_世界農業センサス_公私有牧野統計表CSV/鳥取県/世界農業センサス/1960/世界農業センサス1960 - 1960年csv用整形データ/世界農業センサス1960 - 1960年csv用整形データ全地域版.csv"
CSV_1975 = "/content/drive/MyDrive/鳥大地理情報研卒論2025/B22A5163M_BITO/02_世界農業センサス_公私有牧野統計表CSV/鳥取県/世界農業センサス/1975/世界農業センサス1975　 - 1975年csv用整形データ/世界農業センサス1975　 - 1975年csv用整形データ.csv"

# Verify files exist
for p in [CSV_1950, CSV_1960, CSV_1975]:
    print(p, "=>", os.path.exists(p))


# ----------------------------
# 2. Multi-encoding CSV reader
# ----------------------------
def read_csv_multi_encoding(path, encodings=("utf-8-sig", "utf-8", "cp932", "shift_jis")):
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded {os.path.basename(path)} with encoding={enc}")
            return df
        except Exception as e:
            last_error = e
    raise last_error


# ----------------------------
# 3. Year-specific loaders
#    Normalise grass_area to a common column name.
# ----------------------------
def load_census_1950(path):
    df = read_csv_multi_encoding(path).copy()
    df = df.rename(columns={
        "結合用都市町村": "municipality",
        "地域": "region",
        "採草地・放牧地・貸付地面積": "grass_area"
    })
    keep = ["municipality", "region", "grass_area"]
    df = df[keep].copy()
    df["grass_area"] = pd.to_numeric(df["grass_area"], errors="coerce")
    df["year"] = 1950
    return df.dropna(subset=["municipality", "grass_area"])


def load_census_1960(path):
    df = read_csv_multi_encoding(path).copy()
    df = df.rename(columns={
        "結合用都市町村": "municipality",
        "地域名": "region",
        "採草地": "grass_cut",
        "放牧地": "grass_grazing"
    })
    df["grass_cut"]     = pd.to_numeric(df["grass_cut"],     errors="coerce")
    df["grass_grazing"] = pd.to_numeric(df["grass_grazing"], errors="coerce")
    df["grass_area"]    = df[["grass_cut", "grass_grazing"]].sum(axis=1, min_count=1)
    keep = ["municipality", "region", "grass_area"]
    df = df[keep].copy()
    df["year"] = 1960
    return df.dropna(subset=["municipality", "grass_area"])


def load_census_1975(path):
    df = read_csv_multi_encoding(path).copy()
    df = df.rename(columns={
        "結合用都市町村": "municipality",
        "牧草専用地面積": "grass_area",
        "Unnamed: 1": "region"
    })
    keep = ["municipality", "region", "grass_area"]
    df = df[keep].copy()
    df["grass_area"] = pd.to_numeric(df["grass_area"], errors="coerce")
    df["year"] = 1975
    return df.dropna(subset=["municipality", "grass_area"])


# ----------------------------
# 4. Load all years
# ----------------------------
frames = []
for year, path, loader in [
    (1950, CSV_1950, load_census_1950),
    (1960, CSV_1960, load_census_1960),
    (1975, CSV_1975, load_census_1975),
]:
    try:
        df_yr = loader(path)
        frames.append(df_yr)
        print(f"{year}: {df_yr.shape}")
    except Exception as e:
        print(f"Error loading data for year {year} from {path}: {e}. Skipping this year.")

display(frames[0].head() if frames else "No data loaded")


# ----------------------------
# 5. Build panel dataset
#    Merge environmental variables from agg (1950-based)
#    into the long-format panel.
# ----------------------------
env_cols = [
    "municipality",
    "count",
    "basin_main",
    "basin_entropy",
    "geology_main_grp",
    "geology_entropy",
    "igneous_share",
    "sedimentary_share",
    "metamorphic_share",
    "elev_mean",
    "elev_sd",
    "slope_mean",
    "slope_sd",
    "curv_mean",
    "curv_sd",
]

env_cols = [c for c in env_cols if c in agg.columns]

panel = pd.concat(frames, axis=0, ignore_index=True)
panel = panel.merge(agg[env_cols].drop_duplicates("municipality"), on="municipality", how="left")

print("panel shape:", panel.shape)
display(panel.head())


# ----------------------------
# 6. Annual summary statistics
# ----------------------------
panel_summary = panel.groupby("year").agg(
    grass_mean=("grass_area", "mean"),
    grass_median=("grass_area", "median"),
    grass_sum=("grass_area", "sum"),
    n_muni=("municipality", "nunique")
).reset_index()

display(panel_summary)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(panel_summary["year"], panel_summary["grass_mean"], marker="o")
axes[0].set_title("Mean grass area by year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Mean grass area")

axes[1].plot(panel_summary["year"], panel_summary["grass_sum"], marker="o")
axes[1].set_title("Total grass area by year")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Total grass area")

plt.tight_layout()
plt.show()


# ----------------------------
# 7. Basic panel regression — year fixed effects
# ----------------------------
m_panel_1 = smf.ols(
    """
    grass_area ~ C(year)
              + geology_entropy
              + basin_entropy
              + elev_mean
              + slope_mean
              + count
    """,
    data=panel
).fit()

print(m_panel_1.summary())


# ----------------------------
# 8. Year × tatara interaction
#    Tests whether the effect of tatara density on grassland
#    area changes across census years.
# ----------------------------
m_panel_2 = smf.ols(
    """
    grass_area ~ C(year)
              + geology_entropy
              + basin_entropy
              + elev_mean
              + slope_mean
              + count
              + C(year):count
    """,
    data=panel
).fit()

print(m_panel_2.summary())


# ----------------------------
# 9. Municipality-level change scores
#    Grassland change from 1950 to 1975.
# ----------------------------
wide = panel.pivot_table(
    index="municipality",
    columns="year",
    values="grass_area",
    aggfunc="first"
).reset_index()

wide.columns.name = None
wide = wide.rename(columns={
    1950: "grass_1950",
    1960: "grass_1960",
    1975: "grass_1975"
})

wide["delta_50_75"] = wide["grass_1975"] - wide["grass_1950"]
wide["delta_50_60"] = wide["grass_1960"] - wide["grass_1950"]
wide["delta_60_75"] = wide["grass_1975"] - wide["grass_1960"]

wide = wide.merge(agg[env_cols].drop_duplicates("municipality"), on="municipality", how="left")

display(wide.head())

# Model explaining the magnitude of grassland change
m_delta = smf.ols(
    """
    delta_50_75 ~ geology_entropy
                + basin_entropy
                + elev_mean
                + slope_mean
                + count
    """,
    data=wide
).fit()

print(m_delta.summary())


# ----------------------------
# 10. Landscape persistence
#     How well does 1950 grassland area predict 1975 grassland area?
# ----------------------------
m_persist = smf.ols(
    """
    grass_1975 ~ grass_1950
               + geology_entropy
               + basin_entropy
               + elev_mean
               + slope_mean
               + count
    """,
    data=wide
).fit()

print(m_persist.summary())


# ================================================================
# 9. Historical population / density (template)
# ================================================================

POP_PATH = "/content/drive/MyDrive/B22A5163M_BITO/.../historical_population.csv"

if os.path.exists(POP_PATH):
    pop_df = read_csv_multi_encoding(POP_PATH)
    pop_df = pop_df.rename(columns={
        pick_first_existing(pop_df.columns, ["municipality", "市町村名"]): "municipality",
        pick_first_existing(pop_df.columns, ["historical_population", "hist_pop", "人口"]): "historical_population",
        pick_first_existing(pop_df.columns, ["area_km2", "面積"]): "area_km2",
    })

    pop_df["historical_population"]  = pd.to_numeric(pop_df["historical_population"],  errors="coerce")
    pop_df["area_km2"]               = pd.to_numeric(pop_df["area_km2"],               errors="coerce")
    pop_df["historical_pop_density"] = pop_df["historical_population"] / pop_df["area_km2"]

    agg = agg.merge(
        pop_df[["municipality", "historical_population", "historical_pop_density"]],
        on="municipality",
        how="left"
    )

    display(agg[["municipality", "historical_population", "historical_pop_density"]].head())

    m_pop = smf.ols(
        '''
        count ~ historical_pop_density
              + geology_entropy
              + basin_entropy
              + elev_mean
              + slope_mean
        ''',
        data=agg
    ).fit()

    print(m_pop.summary())
else:
    print("historical population file not found; skip this section.")


# ================================================================
# 10. Classification analysis including terrain and population
# ================================================================

cls_df = agg.copy()
thr = cls_df["grass"].quantile(0.75)
cls_df["high_grass"] = (cls_df["grass"] >= thr).astype(int)

features = [
    "basin_main",
    "geology_main_grp",
    "basin_entropy",
    "geology_entropy",
    "igneous_share",
    "sedimentary_share",
    "metamorphic_share",
    "elev_mean",
    "slope_mean",
    "curv_mean",
    "count",
]

if "historical_pop_density" in cls_df.columns:
    features.append("historical_pop_density")

cat_cols = [c for c in features if str(cls_df[c].dtype) in ["object", "category"]]
num_cols = [c for c in features if c not in cat_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ]
)

logit = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
pipe = Pipeline([("prep", pre), ("model", logit)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipe,
    cls_df[features],
    cls_df["high_grass"],
    cv=cv,
    scoring={
        "auc": "roc_auc",
        "acc": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score),
    }
)

print("AUC:", np.mean(scores["test_auc"]))
print("ACC:", np.mean(scores["test_acc"]))
print("F1 :", np.mean(scores["test_f1"]))


# ================================================================
# 11. Final integrated model
#     Simultaneously examines ecological constraints and the
#     condition-dependence of cultural adaptation.
# ================================================================

final_formula = '''
grass ~ geology_entropy
      + basin_entropy
      + elev_mean
      + slope_mean
      + curv_mean
      + count
      + count:geology_entropy
      + count:slope_mean
'''

if "historical_pop_density" in agg.columns:
    final_formula = final_formula.rstrip() + " + historical_pop_density\n"

m_final = smf.ols(final_formula, data=agg).fit()
print(m_final.summary())


# ================================================================
# 12. Interpretation notes
#
# Key signals to look for in the results:
#   - Strong geology / slope effects         → ecological constraint
#   - Significant count:slope_mean or
#     count:geology_entropy interaction      → condition-dependent cultural adaptation
#   - Strong DBSCAN clusters                 → cultural clustering
#   - Significant historical_pop_density     → population density and industrial specialisation
#   - Large year-to-year differences         → landscape persistence vs. transformation
# ================================================================
