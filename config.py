"""
Central configuration and constants for the Precision Irrigation & Sugarcane
Water Productivity app.
"""

# Default working CRS for all zonal/statistical operations (projected, metres).
# Change this in the sidebar at runtime if your AOIs use a different UTM zone.
DEFAULT_CRS = "EPSG:32736"  # UTM 36S - southern Africa

# Column in the uploaded field-boundary file that holds the unique field ID.
# The loader also accepts common alternatives and renames them to this.
FIELD_ID_COL = "ID"
FIELD_ID_ALIASES = ["ID", "id", "field_id", "Field_ID", "FIELD_ID", "name", "Name"]

# Vegetation indices computed by the remote sensing module
VEGETATION_INDICES = ["NDVI", "EVI", "SAVI", "NDRE"]

# Sentinel-2 SCL (Scene Classification Layer) codes treated as invalid/cloud
# and masked out of every index calculation.
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud med/high prob,
# 10 thin cirrus, 11 snow/ice
SCL_MASK_CODES = [0, 1, 3, 8, 9, 10, 11]

# Anomaly detection defaults (a t-statistic approach with degrees-of-freedom
# correction gives a materially lower false-positive rate at field scale than
# a naive z-score threshold)
ANOMALY_BASELINE_WEEKS = 5
ANOMALY_CONFIDENCE = 0.95

MODEL_DIR = "data/models"
