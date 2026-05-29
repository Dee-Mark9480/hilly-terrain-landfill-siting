# Land Suitability Index (LSI) for Urban Development — Shimla District, Himachal Pradesh

This project develops a **Land Suitability Index (LSI)** for urban/infrastructure development in Shimla District using geospatial multi-criteria analysis combined with machine learning. The study area spans approximately 124 sq.km of Shimla Municipal Corporation area. Each 10×10 m pixel across the district is scored on a suitability scale of 1 (Very Low) to 5 (Very High) based on 11 geospatial criteria layers. Random Forest and XGBoost models are trained on this data to predict and map suitability across the entire district.

---

## 📂 Dataset Access

Due to the large size of processed raster layers and intermediate files, the full dataset cannot be hosted on Google Drive. The `.tif` raster files and `.parquet` files are generated through the manual download and QGIS preprocessing steps described in the sections below.

The Drive link below contains only the files that are **not easily available on the internet** and would otherwise require significant effort to obtain or prepare:

- `shimla_boundary_utm.shp` — Shimla Municipal Corporation boundary (reprojected to UTM 43N)
- Land value / property rate spatial data (collected and cleaned from listing sources)

🔗 **[Access supplementary files on Google Drive](https://drive.google.com/drive/folders/1theDKxtPCSQZqcPRpBd0pyjVrZKxUyrF?usp=sharing)**

All remaining layers (elevation, slope, LULC, roads, railways, LST, landslide inventory, lineaments) must be downloaded manually from the sources listed in the Data Collection table and preprocessed in QGIS as described in Section 2. For full methodological details, refer to the research paper:

> **[Land Suitability Index for Urban Development — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2949750725000100)**

---

## 1. Data Collection

Eleven geospatial criteria layers were collected from open public sources. Each layer was clipped to the Shimla district boundary and exported as a GeoTIFF at 10m resolution (aligned to the Sentinel-2 grid, EPSG:32643 UTM Zone 43N).

| Layer | Description | Source |
|---|---|---|
| **LULC** | ESRI Global Land Cover (Sentinel-2 based, 2022) | [ESRI Living Atlas](https://livingatlas.arcgis.com/landcoverexplorer/) |
| **Elevation (DEM)** | SRTM 30m Digital Elevation Model | [NASA EarthData](https://www.earthdata.nasa.gov/data/instruments/srtm) / [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| **Slope** | Derived from SRTM DEM | Computed in QGIS from DEM |
| **Road Network** | OpenStreetMap roads (India extract) | [Geofabrik India](https://download.geofabrik.de/asia/india.html) |
| **Railway Network** | OpenStreetMap railways | [OpenStreetMap](https://www.openstreetmap.org) |
| **Land Surface Temperature (LST)** | Landsat 8/9 Collection-2 Level-2 surface temperature | [USGS EarthExplorer](https://earthexplorer.usgs.gov) |
| **Landslide Inventory** | Historical landslide point/polygon data for HP | [GSI Bhukosh Portal](https://bhukosh.gsi.gov.in/) / [BhuSanket Portal](https://bhusanket.gsi.gov.in/) |
| **Lineaments** | Geological lineament layer (fault/fracture zones) | [GSI Bhukosh Portal](https://bhukosh.gsi.gov.in/) |
| **Study Area Boundary** | Shimla Municipal Corporation / district boundary | [HOT Export Tool](https://export.hotosm.org/) / [Survey of India](https://onlinemaps.surveyofindia.gov.in/) |
| **Land Value** | Property/circle rate spatial data | [MagicBricks](https://www.magicbricks.com) / [99acres](https://www.99acres.com) / [HP Revenue Dept.](https://himachal.nic.in/en-IN/department-of-revenue.html) |

**Derived outputs from raw data:**  
`lulc_suitability.tif`, `distance_from_builtup.tif`, `distance_from_water.tif`, `distance_from_road.tif`, `distance_from_rail.tif`, `distance_from_landslide.tif`, `distance_from_lineament.tif`, `elevation.tif`, `slope.tif`, `lst_celcius.tif`, `land_value.tif`

---

## 2. Preprocessing in QGIS

All raw downloaded files were processed in **QGIS 3.x** before being used in Python. The steps were:

**Coordinate Reference System (CRS):** All layers were reprojected to **EPSG:32643 (WGS 84 / UTM Zone 43N)** to ensure metric distance calculations.

**Boundary clipping:** Every raster and vector layer was clipped to the Shimla district boundary shapefile (`shimla_boundary_utm.shp`) using *Raster → Extraction → Clip Raster by Mask Layer*.

**Raster alignment:** All layers were resampled to a common 10m pixel grid using *Raster → Projections → Warp (Reproject)*, ensuring pixel-perfect alignment across all feature layers.

**Distance rasters:** Vector layers (roads, railways, landslides, lineaments, built-up areas from LULC, water bodies from LULC) were converted to distance rasters using *Raster → Analysis → Proximity (Raster Distance)*.

**LST processing:** Landsat ST_B10 band values were converted from scaled integers to degrees Celsius using the band scale/offset metadata.

**LULC reclassification:** The ESRI LULC classes were reclassified into suitability scores using *Raster → Raster Calculator* or the Reclassify tool.

**LSI computation:** The final continuous LSI raster (`LSI_continuous.tif`) was computed as a weighted overlay of all 11 criteria in QGIS Raster Calculator. It was then reclassified into 5 discrete classes (`LSI_classified.tif`): Very Low (1), Low (2), Moderate (3), High (4), Very High (5).

All final `.tif` files were saved in the `Raw data/` folder before being used in Python.

---

## 3. Code — Notebook Descriptions

The workflow is split across 7 Jupyter notebooks, meant to be run in this order:

---

### `extraction.ipynb`
**Purpose:** Reads all 11 GeoTIFF rasters + the two LSI target rasters, masks them to the Shimla boundary, and extracts pixel-level values into a flat tabular structure.

For each pixel inside the boundary, it records the values of all 11 feature layers (`landslide_dist`, `slope`, `lulc`, `elevation`, `road_dist`, `builtup_dist`, `land_value`, `water_dist`, `lineament_dist`, `lst`, `railway_dist`) along with the LSI continuous score and classified label. Pixel row/column indices are also stored for spatial reconstruction later. The output is saved as `shimla_dataset.parquet` — a flat table with one row per valid pixel (~millions of rows).

---

### `smcArea.ipynb`
**Purpose:** Validates and reconciles the actual study area extent.

The notebook computes the bounding box of the Shimla Municipal Corporation area from approximate lat/lon extents derived from reference maps, converts these to UTM pixel coordinates, and checks whether the pixel count matches the expected ~124 sq.km study area. If there is a mismatch, it computes a scale correction factor and adjusts the bounding box proportionally. This ensures downstream analysis uses the correct spatial extent.

---

### `data_preprocessing.ipynb`
**Purpose:** Performs spatial block-based train/validation/test splitting of the pixel dataset.

The full pixel dataset from `shimla_dataset.parquet` is divided into 1 km × 1 km spatial blocks (100 × 100 pixel blocks). A stratified split (70% train / 15% val / 15% test) is done at the block level — not pixel level — to prevent spatial autocorrelation leakage between splits. Special care is taken to ensure blocks containing the rare Class 5 (Very High suitability) pixels are proportionally represented in all splits. The split labels are saved back into the parquet file.

---

### `model.ipynb`
**Purpose:** Trains and evaluates a **Random Forest** model for LSI classification and regression.

Using the subsampled spatially-split dataset, it trains a `RandomForestClassifier` (300 trees, balanced class weights) to predict the 5-class LSI label, and a `RandomForestRegressor` for the continuous LSI score. Evaluation is done on validation and test sets with accuracy, weighted F1-score, Cohen's Kappa, and a full confusion matrix. Feature importance scores are also extracted.

---

### `xgboost.ipynb`
**Purpose:** Trains **XGBoost** classifier and regressor models and generates full district-wide prediction maps.

An `XGBClassifier` and `XGBRegressor` are trained on the same spatial train split. After training, both models are applied to every valid pixel in the full `shimla_dataset.parquet` (in memory-safe chunks of 5 million rows at a time). The predicted class and continuous LSI values are reconstructed back into raster grids and saved as new GeoTIFF files, producing spatially complete prediction maps of the entire Shimla district.

---

### `PHASE1.ipynb`
**Purpose:** Post-prediction spatial analysis — extracts and visualizes **High suitability zones** (Class 4 and 5) across the district.

Loads the full dataset with predictions, filters pixels classified as High or Very High suitability, computes their area and feature statistics, and generates side-by-side spatial scatter plots overlaid on the Shimla boundary — one showing all classes, another showing only the high-suitability zones. Outputs are saved to the `Shimla Result/` folder.

---

### `PHASEATEMP.ipynb`
**Purpose:** A paper-aligned version of the Phase 1 analysis.

Functionally very similar to `PHASE1.ipynb`, but uses output paths and figure formatting tuned for paper/report figures (`Shimla Result Paper based/` folder). Useful if you want to regenerate the exact maps used in the written report without overwriting the exploratory outputs from `PHASE1.ipynb`.

---

## 4. How to Load the Raster Files

If you want to work directly with the `.tif` files from the Google Drive, here is how to load and inspect them in Python:

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# --- Open a raster file ---
raster_path = "path/to/slope.tif"  # replace with your local path

with rasterio.open(raster_path) as src:
    print("CRS:", src.crs)
    print("Resolution:", src.res)         # (pixel_width, pixel_height) in meters
    print("Dimensions:", src.width, "x", src.height)
    print("Bounds:", src.bounds)
    print("NoData:", src.nodata)
    
    data = src.read(1)                    # read band 1 as numpy array
    transform = src.transform             # affine transform (pixel → coordinates)

# --- Quick plot ---
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='terrain')
plt.colorbar(label='Value')
plt.title('Raster Layer')
plt.tight_layout()
plt.show()
```

**To load and clip to the study boundary:**

```python
import geopandas as gpd
from rasterio.mask import mask

boundary = gpd.read_file("shimla_boundary_utm.shp")
shapes = boundary.geometry.values

with rasterio.open("slope.tif") as src:
    clipped, transform = mask(src, shapes, crop=True, filled=False)
    band = clipped[0].astype(np.float32)
    band[clipped[0].mask] = np.nan     # set outside-boundary pixels to NaN
```

**To convert pixel indices back to geographic coordinates:**

```python
# Given pixel_row and pixel_col (as stored in the parquet files)
with rasterio.open("landslide_distance.tif") as src:
    t = src.transform

easting  = t.c + pixel_col * t.a     # X (easting in UTM)
northing = t.f + pixel_row * t.e     # Y (northing in UTM)
```

**Required Python packages:**

```bash
pip install rasterio geopandas numpy pandas matplotlib pyproj scikit-learn xgboost
```

---

## 5. Project Structure

```
project/
│
├── Raw data/                   # All 11 clipped & aligned GeoTIFFs
│   ├── slope.tif
│   ├── dem_clip.tif
│   ├── lulc_shimla.tif
│   ├── landslide_distance.tif
│   ├── roads_distance.tif
│   ├── railway_distance.tif
│   ├── distance_from_builtup.tif
│   ├── distance_from_water.tif
│   ├── lineament_aligned.tif
│   ├── lst_celcius.tif
│   └── maskedlandcords.tif     # land value raster
│
├── LSI/
│   ├── LSI_continuous.tif      # Weighted overlay LSI (QGIS output)
│   ├── LSI_classified.tif      # 5-class discrete LSI
│   ├── shimla_dataset.parquet  # Full pixel-level feature table
│   └── shimla_subsampled_spatial.parquet  # Train/val/test split dataset
│
├── DATA/shimla shp/
│   └── shimla_boundary_utm.shp # Study area boundary (EPSG:32643)
│
├── extraction.ipynb
├── smcArea.ipynb
├── data_preprocessing.ipynb
├── model.ipynb
├── xgboost.ipynb
├── PHASE1.ipynb
└── PHASEATEMP.ipynb
```

---

## 6. Notes

- All rasters use **EPSG:32643 (WGS 84 / UTM Zone 43N)** and a 10m pixel resolution.
- The parquet files store pixel row/col indices (not coordinates) — use the reference raster's affine transform to convert back to spatial coordinates.
- Class labels: `1 = Very Low`, `2 = Low`, `3 = Moderate`, `4 = High`, `5 = Very High` suitability.
- The spatial block-split in `data_preprocessing.ipynb` is essential — do not shuffle by pixel, as this causes data leakage due to spatial autocorrelation.
