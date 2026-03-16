# Grass–Tatara: Environmental Constraints and Historical Industry in Western Japan

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19042339.svg)](https://doi.org/10.5281/zenodo.19042339)

Reproducible spatial analysis code accompanying the paper:

> Onohara, A., Bito, T., & Iwasaki, N. (2026). *Re-examining the Relationship Between Tatara Iron Production and Grassland Distribution in Western Japan: An Open Geospatial Approach to Historical Landscape Analysis.* Presented at FOSS4G 2026, Hiroshima.

## Overview

This repository contains the Python analysis code used to examine the spatial relationship between tatara iron production sites and grassland distribution in Tottori Prefecture, western Japan.

The central question is whether the well-known spatial co-occurrence of tatara sites and historical grasslands reflects a direct causal relationship — iron production creating open landscapes through forest clearance — or whether both patterns are independently structured by shared environmental constraints such as geology, river basin organisation, and mountain topography.

The analysis integrates three classes of data:

- Municipality-level grassland area from the 1950 World Agricultural Census (digitised from printed statistical tables)
- Tatara iron production site locations from the Tottori Prefecture WebGIS cultural heritage database
- Environmental variables derived from open geospatial datasets: geological maps, river basin structures, and a digital elevation model (DEM)

## Repository structure

```
.
├── grass_tatara_evolutionary_extensions.py   # Main analysis script
├── requirements.txt                          # Python dependencies
└── README.md
```

The script is structured to run as a Google Colab notebook. Input data paths point to Google Drive and are not included in this repository (see Data section below).

## Analysis steps

|Step|Description                                                                                          |
|----|-----------------------------------------------------------------------------------------------------|
|1   |Reconstruct base dataset: municipality polygons, geology/basin summaries, tatara site counts         |
|2   |Generate slope and curvature rasters from DEM                                                        |
|3   |Aggregate terrain indicators (elevation, slope, curvature) at municipality level via zonal statistics|
|4   |OLS regression: grassland area ~ environmental variables only                                        |
|5   |Interaction models: tatara × slope, tatara × geology entropy                                         |
|6   |DBSCAN spatial clustering of tatara point data                                                       |
|7   |Environmental summary per production cluster                                                         |
|8   |Time-series panel (1950 / 1960 / 1975) — requires additional CSV inputs                              |
|9   |Historical population density as additional predictor (template)                                     |
|10  |Logistic classification: high vs. low grassland municipalities                                       |
|11  |Final integrated model combining ecological constraints and cultural adaptation terms                |

## Requirements

```
geopandas
rasterio
rasterstats
scipy
scikit-learn
statsmodels
numpy
pandas
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

Or in Google Colab:

```python
!pip -q install geopandas rasterio rasterstats scipy scikit-learn statsmodels
```

## Data

Input data are not distributed in this repository. The following files are required:

|Variable                                     |Source                                                                                                                                                                                  |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Grassland area (1950)                        |1950 World Agricultural Census, municipality-level statistics. Digitised from printed tables using AI-assisted OCR.                                                                     |
|Tatara site locations                        |Tottori Prefecture WebGIS cultural heritage database (とっとりWebマップ). Point data extracted via browser developer tools and processed into CSV.                                             |
|Municipality polygons + basin/geology overlay|Constructed in QGIS from National Land Numerical Information administrative boundaries, hydrological mesh (MLIT 2009), and GSJ seamless geological map (GSJ 2025). Stored as FlatGeobuf.|
|DEM                                          |Tottori Prefecture 10m DEM (EPSG:6673).                                                                                                                                                 |

To use this code with your own data, update the path constants at the top of the script (`EXTRACT_DIR`, `CSV_PATH`, `FGB_PATH`, `TATARA_POINT_CSV`, `DEM_PATH`).

## Key results

- Tatara sites exhibit strong spatial clustering, concentrated in the Hino River basin in western Tottori Prefecture.
- Grassland distribution is most strongly predicted by mean elevation and terrain variables.
- The interaction term `count × slope_mean` is statistically significant and negative: the positive association between tatara density and grassland area weakens in steeper terrain.
- These findings suggest that the observed co-occurrence of tatara sites and grasslands largely reflects shared environmental constraints rather than a simple causal relationship.

## Code archive

The analysis code is permanently archived on Zenodo:

> https://doi.org/10.5281/zenodo.19042339

## Citation

If you use this code, please cite:

```
Onohara, A., Bito, T., & Iwasaki, N. (2026). Re-examining the Relationship Between
Tatara Iron Production and Grassland Distribution in Western Japan: An Open Geospatial
Approach to Historical Landscape Analysis. FOSS4G 2026, Hiroshima.
```

## License

Code: [MIT License](LICENSE)  
Analysis outputs and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Acknowledgements

Grassland census data were digitised from printed volumes held at the National Diet Library of Japan. Tatara site location data were obtained from the Tottori Prefecture Board of Education cultural heritage records via とっとりWebマップ.
