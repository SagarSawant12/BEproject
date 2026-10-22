import os
import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling

class SentinelSnowClassifier:
    def __init__(self, model_path="models/rf_snow_multispectral.joblib"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"[ML] Loaded trained Random Forest snow model from {model_path}")
        else:
            print("[ML] Notice: Model weights not found. Using baseline fallback.")

    def predict_snow_mask(self, b03_path: str, b11_path: str, b02_path=None, b04_path=None, b08_path=None) -> np.ndarray:
        """
        Generates a 2D binary snow mask (1 = Snow, 0 = Non-snow).
        Uses the Random Forest model if trained; falls back to the NDSI threshold if weights are pending.
        """
        # Read reference SWIR band (20m grid)
        with rasterio.open(b11_path) as src:
            b11 = src.read(1).astype(np.float32)
            ref_shape = b11.shape

        # Read Green band and resample to 20m
        with rasterio.open(b03_path) as src:
            b03 = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear).astype(np.float32)

        eps = 1e-5
        ndsi = (b03 - b11) / (b03 + b11 + eps)

        # Baseline threshold fallback if RF model is not loaded yet
        if self.model is None or not all([b02_path, b04_path, b08_path]):
            # Frozen baseline threshold: 0.495
            snow_mask = np.where((ndsi >= 0.495) & np.isfinite(ndsi), 1, 0).astype(np.uint8)
            return snow_mask

        # Multispectral Inference Pipeline
        with rasterio.open(b02_path) as src:
            b02 = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear).astype(np.float32)
        with rasterio.open(b04_path) as src:
            b04 = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear).astype(np.float32)
        with rasterio.open(b08_path) as src:
            b08 = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear).astype(np.float32)

        ndvi = (b08 - b04) / (b08 + b04 + eps)
        ndwi = (b03 - b08) / (b03 + b08 + eps)
        r_nir_swir = b08 / (b11 + eps)
        r_green_swir = b03 / (b11 + eps)

        features = np.dstack([b02, b03, b04, b08, b11, ndsi, ndvi, ndwi, r_nir_swir, r_green_swir])
        h, w, c = features.shape
        flat_x = features.reshape(-1, c)

        preds = np.zeros(flat_x.shape[0], dtype=np.uint8)
        valid_mask = np.all(np.isfinite(flat_x), axis=1)

        if np.any(valid_mask):
            preds[valid_mask] = self.model.predict(flat_x[valid_mask])

        return preds.reshape(h, w)