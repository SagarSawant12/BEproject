"""
GeoGuard CDSE (Copernicus Data Space Ecosystem) Sentinel-2 API Client
Handles OAuth2 Keycloak token authentication and OData satellite imagery search.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

class CDSEClient:
    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    def __init__(self):
        self.username = os.getenv("CDSE_USERNAME", "")
        self.password = os.getenv("CDSE_PASSWORD", "")
        self.client_id = os.getenv("CDSE_CLIENT_ID", "cdse-public")
        self.client_secret = os.getenv("CDSE_CLIENT_SECRET", "")
        self.access_token = None

    def get_token(self) -> str:
        """Retrieves or refreshes Keycloak Bearer access token."""
        payload = {
            "client_id": self.client_id if self.client_id else "cdse-public",
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }

        # If using dedicated API client secret
        if self.client_secret:
            payload["client_secret"] = self.client_secret
            payload["grant_type"] = "client_credentials"
            del payload["username"]
            del payload["password"]

        response = requests.post(self.AUTH_URL, data=payload, timeout=15)
        if response.status_code == 200:
            self.access_token = response.json().get("access_token")
            return self.access_token
        else:
            raise RuntimeError(f"CDSE Authentication Failed ({response.status_code}): {response.text}")

    def search_sentinel2_tiles(self, lat: float, lon: float, max_cloud_cover: int = 20):
        """Queries recent Sentinel-2 multispectral tiles intersecting coordinates."""
        if not self.access_token:
            self.get_token()

        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Standard EPSG:4326 POINT representation
        point_geom = f"geography'SRID=4326;POINT({lon:.5f} {lat:.5f})'"

        filters = [
            "Collection/Name eq 'SENTINEL-2'",
            f"OData.CSC.Intersects(area={point_geom})"
        ]
        
        filter_str = " and ".join(filters)

        params = {
            "$filter": filter_str,
            "$orderby": "ContentDate/Start desc",
            "$top": 5
        }

        response = requests.get(self.ODATA_URL, headers=headers, params=params, timeout=20)
        if response.status_code == 200:
            return response.json().get("value", [])
        else:
            print(f"[CDSE ODATA ERROR] {response.status_code}: {response.text}")
            return []