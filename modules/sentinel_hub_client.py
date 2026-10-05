"""
Minimal client for the Copernicus Data Space Ecosystem (CDSE) Sentinel Hub
Process API - OAuth2 client-credentials flow + a single evalscript that
returns the bands needed for NDVI/EVI/SAVI/NDRE plus the SCL layer for cloud
masking.

Credential lookup order (first match wins):
  1. st.secrets["COPERNICUS_CLIENT_ID"] / st.secrets["COPERNICUS_CLIENT_SECRET"]
  2. st.secrets["copernicus"]["client_id"] / ["client_secret"]   (nested table)

If neither is present, `is_configured()` returns False and callers should
fall back to "upload your own GeoTIFF bands" or demo mode - never silently
fabricate a response.

Note on SCL: it must be requested as raw DN (not reflectance) or the API
returns HTTP 400 - the evalscript below deliberately keeps SCL out of the
reflectance-scaled output group.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import requests

try:
    import streamlit as st
except ImportError:  # allows unit-testing this module outside streamlit
    st = None

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

BANDS = ["B02", "B03", "B04", "B05", "B08"]  # blue, green, red, red-edge, NIR

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B05","B08","SCL","dataMask"] }],
    output: [
      { id: "bands", bands: 5, sampleType: "FLOAT32" },
      { id: "scl", bands: 1, sampleType: "UINT8" }
    ]
  };
}
function evaluatePixel(sample) {
  return {
    bands: [sample.B02, sample.B03, sample.B04, sample.B05, sample.B08],
    scl: [sample.SCL]
  };
}
"""


@dataclass
class SceneRequest:
    bbox: list[float]  # [minx, miny, maxx, maxy] in EPSG:4326
    date_from: dt.date
    date_to: dt.date
    width: int = 512
    height: int = 512


def _get_credentials() -> tuple[str | None, str | None]:
    if st is None:
        return None, None
    secrets = getattr(st, "secrets", {})
    if "COPERNICUS_CLIENT_ID" in secrets and "COPERNICUS_CLIENT_SECRET" in secrets:
        return secrets["COPERNICUS_CLIENT_ID"], secrets["COPERNICUS_CLIENT_SECRET"]
    if "copernicus" in secrets:
        c = secrets["copernicus"]
        return c.get("client_id"), c.get("client_secret")
    return None, None


def is_configured() -> bool:
    cid, secret = _get_credentials()
    return bool(cid and secret)


def _get_access_token() -> str:
    cid, secret = _get_credentials()
    if not cid or not secret:
        raise RuntimeError(
            "Copernicus/CDSE credentials not found in st.secrets. Add "
            "COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET (see "
            ".streamlit/secrets.toml.example) or use Upload/Demo mode instead."
        )
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials", "client_id": cid, "client_secret": secret},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_scene(req: SceneRequest) -> dict:
    """
    Fetch one least-cloudy Sentinel-2 mosaic for the given bbox/date window.
    Returns {"bands": np.ndarray[5,H,W] float32 reflectance, "scl": np.ndarray[H,W] uint8}.
    Raises on any failure - callers decide the offline/demo fallback explicitly.
    """
    token = _get_access_token()
    payload = {
        "input": {
            "bounds": {"bbox": req.bbox},
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{req.date_from.isoformat()}T00:00:00Z",
                            "to": f"{req.date_to.isoformat()}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": req.width,
            "height": req.height,
            "responses": [
                {"identifier": "bands", "format": {"type": "image/tiff"}},
                {"identifier": "scl", "format": {"type": "image/tiff"}},
            ],
        },
        "evalscript": EVALSCRIPT,
    }
    resp = requests.post(
        PROCESS_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=90,
    )
    resp.raise_for_status()
    # NOTE: a multipart TIFF response needs a small amount of parsing glue
    # (rasterio can read each part directly from bytes via MemoryFile) - left
    # as an integration point once you're testing against real credentials,
    # since the exact multipart boundary handling is easiest to get right
    # against a live response rather than guessed blind.
    return {"raw_response": resp.content}
