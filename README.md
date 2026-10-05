# Precision Irrigation & Sugarcane Water Productivity

A Streamlit app operationalising the MSc research proposal *"Integrating Satellite
Remote Sensing, UAV Geomatics and Machine Learning for Precision Irrigation and
Sugarcane Water Productivity Assessment."* It ingests field boundaries, yield
records and irrigation/weather logs, computes Sentinel-2 vegetation indices,
compares them with UAV orthomosaics, trains yield-prediction and anomaly-detection
models, estimates spatial water productivity, and combines everything into a
map-based decision-support dashboard.

## Project structure

```
sugarcane-precision-irrigation/
├── app.py                      # Home page: data upload / demo mode, previews
├── config.py                   # Shared constants (CRS, field-ID column, SCL codes, ...)
├── requirements.txt
├── .streamlit/
│   ├── config.toml             # Theme
│   └── secrets.toml.example    # Copy to secrets.toml for live CDSE fetch (optional)
├── modules/
│   ├── data_ingestion.py       # Field boundary / yield / irrigation loaders + validation
│   ├── sentinel_hub_client.py  # CDSE Sentinel Hub OAuth2 + Process API client
│   ├── remote_sensing.py       # NDVI/EVI/SAVI/NDRE, SCL cloud masking, zonal stats
│   ├── ml_models.py            # RF/XGBoost yield model, Isolation Forest + t-stat anomaly detection
│   ├── water_productivity.py   # Water productivity calculator + tiering
│   ├── visualization.py        # folium maps + plotly charts
│   └── sample_data.py          # Synthetic DEMO dataset generator
├── pages/
│   ├── 1_📡_Satellite_Monitoring.py
│   ├── 2_🚁_UAV_Analysis.py
│   ├── 3_🤖_ML_Yield_and_Anomaly.py
│   ├── 4_💧_Water_Productivity.py
│   └── 5_🗺️_Decision_Dashboard.py
└── data/                       # aoi/, uploads/, models/ (gitignored except .gitkeep)
```

## Data formats expected

| Dataset | Format | Required columns |
|---|---|---|
| Field boundaries | GeoJSON, or zipped Shapefile (.zip with .shp/.shx/.dbf/.prj) | An ID column (`ID`, `field_id`, etc. - auto-detected) |
| Yield records | CSV | field ID column, `cane_yield_t_ha` (season optional) |
| Irrigation/weather log | CSV | field ID column, `date`, `irrigation_mm` (`rainfall_mm`/`eto_mm` optional) |
| Satellite/UAV imagery | Multi-band GeoTIFF | Bands you specify in-app, e.g. `blue,red,rededge,nir` |

If you don't have these yet, use **"Try with demo data"** on the home page to
generate a synthetic estate and explore every page immediately - it is always
clearly labelled as demo data in the UI.

## Local setup

```bash
git clone <your-repo-url>
cd sugarcane-precision-irrigation
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional: live Sentinel-2 fetching via Copernicus Data Space Ecosystem (CDSE)

1. Register for CDSE OAuth2 credentials at https://dataspace.copernicus.eu
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in
   `COPERNICUS_CLIENT_ID` / `COPERNICUS_CLIENT_SECRET`.
3. `.streamlit/secrets.toml` is already gitignored - never commit real credentials.

Without this, the app is fully usable via **"Upload Sentinel-2 GeoTIFF bands"**
(e.g. scenes you've already downloaded/clipped in QGIS/SNAP) or demo mode.

> **Known gap to finish before relying on live CDSE fetch:** `modules/sentinel_hub_client.py`
> handles the OAuth2 token exchange and the Process API request, but the
> multipart-TIFF response parsing is left as a marked integration point -
> it's far more reliable to finish that against a live response (exact
> multipart boundary format) than to guess it here. The upload and demo paths
> already exercise the full downstream pipeline (indices → zonal stats →
> models → dashboard) end to end.

## Deploying to Streamlit Community Cloud

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: precision irrigation app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sugarcane-precision-irrigation.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io, sign in with GitHub, click **"New app"**.
3. Select the repo/branch and set **Main file path** to `app.py`.
4. If using live CDSE fetch, open **Advanced settings → Secrets** in the deploy
   dialog and paste the contents of your local `secrets.toml` (the file itself
   is never pushed to GitHub).
5. Click **Deploy**. Community Cloud installs `requirements.txt` automatically.

## Notes on the analytical choices

- **Cloud masking** uses the Sentinel-2 SCL layer (`config.SCL_MASK_CODES`);
  request it as raw DN, not reflectance, or the CDSE API returns HTTP 400.
- **Temporal anomaly detection** fits an OLS baseline trend per field and flags
  the current observation using a Student's-t critical value with
  degrees-of-freedom correction, rather than a fixed z-score threshold - this
  is materially more robust with the short time series typical of a single
  growing season.
- **Water productivity** is reported as t/ha of cane per 100mm of total water
  received (irrigation + rainfall). Treat it as a direct measurement only where
  irrigation is reliably metered; otherwise it is a remote-sensing-informed
  proxy, and the app labels it as such.
- All CRS handling defaults to `EPSG:32736` (UTM 36S); change `config.DEFAULT_CRS`
  if your study area sits in a different UTM zone.

## Roadmap / extension points

- Finish the CDSE multipart-TIFF parsing in `sentinel_hub_client.py` for fully
  live Sentinel-2 fetching (or swap in Google Earth Engine as an alternative
  data source).
- Add a pytest suite (`tests/`) mirroring the module boundaries above.
- Add GitHub Actions for a scheduled weekly Sentinel-2 pull once live fetching
  is wired up.
- Extend `water_productivity.py` with ET-based or economic water-productivity
  variants once reliable evapotranspiration/price data is available.
