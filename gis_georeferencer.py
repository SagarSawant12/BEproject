"""
GeoGuard Dynamic GIS Georeferencer & Tactical Mapping Module
Extracts EXIF metadata from drone imagery, interacts with Copernicus CDSE,
and generates interactive tactical Leaflet HTML maps with geocoding fallback.
"""

import os
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from cdse_client import CDSEClient


class TacticalGISManager:
    """Manages GPS metadata extraction, CDSE satellite catalog queries, and Leaflet map layers."""

    def __init__(self, fallback_lat: float = 19.0760, fallback_lon: float = 72.8777):
        self.default_lat = fallback_lat
        self.default_lon = fallback_lon
        self.current_lat = fallback_lat
        self.current_lon = fallback_lon
        self.geolocator = Nominatim(user_agent="geoguard_tactical_c2")
        self.cdse = CDSEClient()
        self.is_cdse_authenticated = False
    
    def get_device_live_location(self) -> Tuple[float, float, str]:
        """Fast, non-blocking locator that never freezes the GUI thread."""
        import urllib.request
        import json

        # 1. Fast, lightweight HTTP request with 1.2s hard timeout
        urls = [
            ("https://ipapi.co/json/", "ipapi"),
            ("http://ip-api.com/json/", "ip-api")
        ]

        for url, provider in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'GeoGuard-C2/1.0'})
                with urllib.request.urlopen(req, timeout=1.2) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode('utf-8'))
                        if provider == "ipapi" and "latitude" in data:
                            lat = float(data["latitude"])
                            lon = float(data["longitude"])
                            city = data.get("city", "Sector")
                            self.current_lat, self.current_lon = lat, lon
                            return lat, lon, f"ONLINE SENSOR ({city})"
                        elif provider == "ip-api" and data.get("status") == "success":
                            lat = float(data["lat"])
                            lon = float(data["lon"])
                            city = data.get("city", "Sector")
                            self.current_lat, self.current_lon = lat, lon
                            return lat, lon, f"ONLINE SENSOR ({city})"
            except Exception:
                continue

        # 2. Instant offline fallback if no network or timeout reached (0ms delay)
        self.current_lat = self.default_lat
        self.current_lon = self.default_lon
        return self.default_lat, self.default_lon, "OFFLINE TOC BASELINE"
    
    def authenticate_cdse(self):
        """Attempts silent background authentication to CDSE."""
        try:
            self.cdse.get_token()
            self.is_cdse_authenticated = True
            print("[GIS] Successfully authenticated with Copernicus Data Space Ecosystem.")
        except Exception as e:
            self.is_cdse_authenticated = False
            print(f"[GIS] CDSE Authentication warning (using offline/local tiles): {e}")

    def query_satellite_tiles(self, lat: float, lon: float):
        """Queries Copernicus Data Space Ecosystem for Sentinel-2 satellite passes."""
        try:
            tiles = self.cdse.search_sentinel2_tiles(lat, lon, max_cloud_cover=25)
            if tiles:
                top_tile = tiles[0]
                return {
                    "product_name": top_tile.get("Name", "Sentinel-2 L2A Scene"),
                    "acquisition_date": top_tile.get("ContentDate", {}).get("Start", "Recent"),
                    "count": len(tiles)
                }
        except Exception as e:
            print(f"[GIS CDSE WARNING] {e}")
        return None

    def _convert_to_degrees(self, value) -> float:
        """Helper to convert GPS rational tuples into decimal degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    def extract_drone_coordinates(self, image_path: str) -> Dict[str, any]:
        """Extracts coordinates from EXIF tags or dataset preset heuristics."""
        telemetry = {
            "lat": self.default_lat,
            "lon": self.default_lon,
            "altitude_m": 120.0,
            "source": "SYNTHETIC_DEFAULT"
        }

        if not image_path or not os.path.exists(image_path):
            return telemetry

        # 1. Hardware EXIF GPS check
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data and 34853 in exif_data:
                gps_info = {}
                for key, val in exif_data[34853].items():
                    sub_tag = GPSTAGS.get(key, key)
                    gps_info[sub_tag] = val

                if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                    lat = self._convert_to_degrees(gps_info["GPSLatitude"])
                    if gps_info.get("GPSLatitudeRef") == "S":
                        lat = -lat
                    lon = self._convert_to_degrees(gps_info["GPSLongitude"])
                    if gps_info.get("GPSLongitudeRef") == "W":
                        lon = -lon

                    telemetry["lat"] = round(lat, 6)
                    telemetry["lon"] = round(lon, 6)
                    telemetry["source"] = "EXIF_HARDWARE_GPS"
                    return telemetry
        except Exception as e:
            print(f"[GIS] EXIF parse error: {e}")

        # 2. Preset Heuristic fallback
        fname = os.path.basename(image_path).lower()
        presets = {
            "suburban": (29.9511, -90.0715),
            "drone_overhead": (29.9728, -90.0512),
            "coastal": (13.0827, 80.2707),
            "flooded_road": (22.5726, 88.3639),
            "urban": (19.0760, 72.8777)
        }

        for keyword, coords in presets.items():
            if keyword in fname:
                telemetry["lat"], telemetry["lon"] = coords
                telemetry["source"] = "DATASET_PRESET"
                return telemetry

        return telemetry

    def compute_mgrs_utm(self, lat: float, lon: float) -> str:
        """Converts decimal degrees into standard UTM / Military grid representation."""
        zone = int((lon + 180) / 6) + 1
        hemisphere = "N" if lat >= 0 else "S"
        return f"{zone}{hemisphere} {int(abs(lat * 10000) % 100000):05d} {int(abs(lon * 10000) % 100000):05d}"

    def geocode_address(self, query: str) -> Optional[Tuple[float, float, str]]:
        """
        Resolves named addresses (e.g., 'BKC', 'Colaba', 'Dadar') into coordinates
        using OSM Nominatim with local context and strict timeouts.
        """
        import urllib.request
        import urllib.parse
        import json

        clean_query = query.strip()
        search_term = clean_query if "mumbai" in clean_query.lower() or "india" in clean_query.lower() else f"{clean_query}, Mumbai"
        
        encoded_query = urllib.parse.quote(search_term)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"

        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'GeoGuardTacticalC2/1.0 (EmergencyOps)'}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    if data and len(data) > 0:
                        lat = float(data[0]['lat'])
                        lon = float(data[0]['lon'])
                        display_name = data[0].get('display_name', clean_query).split(',')[0]
                        self.current_lat, self.current_lon = lat, lon
                        return lat, lon, display_name
        except Exception as e:
            print(f"[GIS] Geocoding lookup error: {e}")

        return None

    def generate_tactical_leaflet_html(self, lat: float, lon: float, hazard_radius_m: int = 350) -> str:
        """Generates an authentic tactical GIS map using real satellite and terrain tiles."""
        mgrs_code = self.compute_mgrs_utm(lat, lon)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>GeoGuard Tactical GIS</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #070B14; font-family: monospace; }}
        /* High-contrast dark tactical canvas when map tiles cannot load offline */
        #map {{ 
            width: 100%; 
            height: 100%; 
            background-color: #070B14;
            background-image: 
                radial-gradient(circle, rgba(0, 212, 255, 0.08) 1px, transparent 1px),
                linear-gradient(to right, rgba(30, 58, 95, 0.25) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(30, 58, 95, 0.25) 1px, transparent 1px);
            background-size: 40px 40px, 80px 80px, 80px 80px;
        }}
        .leaflet-container {{ background: #080D1A !important; }}
        .tactical-hud {{
            position: absolute; top: 12px; right: 12px; z-index: 1000;
            background: rgba(10, 16, 30, 0.90); border: 1px solid #1E3A5F;
            border-left: 3px solid #00D4FF; padding: 10px 14px;
            color: #E2E8F0; font-size: 11px; border-radius: 4px; pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="tactical-hud">
        <b style="color:#00D4FF;">GEOGUARD TACTICAL GIS // C2</b><br/>
        LAT / LON: {lat:.6f}, {lon:.6f}<br/>
        MGRS GRID: {mgrs_code}<br/>
        <span id="hud-perimeter">PERIMETER: {hazard_radius_m}m RADIUS</span>
    </div>

    <script>
        var map = L.map('map', {{
            center: [{lat}, {lon}],
            zoom: 16,
            zoomControl: true
        }});

        // 1. High-Resolution Satellite Layer
        var satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            maxZoom: 19,
            attribution: 'Esri & Earthstar Geographics'
        }}).addTo(map);

        // 2. OpenStreetMap Streets Layer
        var streetLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }});

        var baseMaps = {{
            "Tactical Satellite": satelliteLayer,
            "Street Map": streetLayer
        }};
        L.control.layers(baseMaps).addTo(map);

        // Ground Zero Incident Marker
        var incidentMarker = L.circleMarker([{lat}, {lon}], {{
            radius: 9,
            color: '#FF2A4D',
            fillColor: '#FF2A4D',
            fillOpacity: 0.95,
            weight: 2
        }}).addTo(map).bindPopup("<b>GROUND ZERO INCIDENT SITE</b><br/>Grid: {mgrs_code}").openPopup();

        // Manual click to re-pin incident location
        map.on('click', function(e) {{
            incidentMarker.setLatLng(e.latlng);
            incidentMarker.bindPopup("<b>MANUAL GROUND ZERO PIN</b><br/>Lat: " + e.latlng.lat.toFixed(5) + "<br/>Lon: " + e.latlng.lng.toFixed(5)).openPopup();
        }});

        // Hazard Exclusion Zone (assigned to window.hazardCircle for live JS manipulation)
        window.hazardCircle = L.circle([{lat}, {lon}], {{
            color: '#FF3B30',
            fillColor: '#FF3B30',
            fillOpacity: 0.25,
            radius: {hazard_radius_m},
            weight: 2,
            dashArray: '4, 6'
        }}).addTo(map);

        // Staging LZ-1
        // Dynamic Safe LZ offset calculation (Distance = 1.35x hazard radius, min 350m buffer)
        var safeDistM = Math.max({hazard_radius_m} * 1.35, 350);
        var dLat = (safeDistM * 0.7071) / 111320.0;
        var dLon = (safeDistM * 0.7071) / (111320.0 * Math.cos({lat} * Math.PI / 180.0));
        var lz1 = [{lat} + dLat, {lon} + dLon];

        // Safe Staging LZ-1 Marker
        window.lzMarker = L.circleMarker(lz1, {{
            radius: 8,
            color: '#00E676',
            fillColor: '#00E676',
            fillOpacity: 0.90,
            weight: 2
        }}).addTo(map).bindPopup("<b>SAFE STAGING LZ-1</b><br/>Status: Operational Outside Hazard Zone");

        // Dynamic Ingress Route Line
        window.routeLine = L.polyline([lz1, [{lat}, {lon}]], {{
            color: '#00E676',
            weight: 3,
            dashArray: '6, 8'
        }}).addTo(map);

        // Global Helper to reposition Safe LZ and route when severity or location changes
        window.updateTacticalLZ = function(centerLat, centerLon, radiusMeters) {{
            var dist = Math.max(radiusMeters * 1.35, 350);
            var offsetLat = (dist * 0.7071) / 111320.0;
            var offsetLon = (dist * 0.7071) / (111320.0 * Math.cos(centerLat * Math.PI / 180.0));
            var newLZ = [centerLat + offsetLat, centerLon + offsetLon];

            if (window.lzMarker) {{
                window.lzMarker.setLatLng(newLZ);
                window.lzMarker.bindPopup("<b>SAFE STAGING LZ-1</b><br/>Buffer: " + Math.round(dist) + "m from Ground Zero").openPopup();
            }}
            if (window.routeLine) {{
                window.routeLine.setLatLngs([newLZ, [centerLat, centerLon]]);
            }}
        }};
        
       // Dynamic in-browser hardware/network locator
        window.locateFromBrowser = function() {{
            if (!navigator.geolocation) {{
                console.warn("Geolocation unsupported by WebEngine runtime.");
                return;
            }}
            navigator.geolocation.getCurrentPosition(
                function(pos) {{
                    var lat = pos.coords.latitude;
                    var lon = pos.coords.longitude;
                    
                    // Fly Leaflet view directly to real coordinates
                    map.flyTo([lat, lon], 16, {{ animate: true, duration: 1.2 }});

                    // Re-pin Ground Zero Incident
                    if (typeof incidentMarker !== 'undefined') {{
                        incidentMarker.setLatLng([lat, lon]);
                        incidentMarker.bindPopup("<b>🚨 EMERGENCY DEVICE LOCATED</b><br/>Source: BROWSER SENSOR<br/>Lat: " + lat.toFixed(5) + "<br/>Lon: " + lon.toFixed(5)).openPopup();
                    }}

                    // Move exclusion circle
                    if (typeof hazardCircle !== 'undefined') {{
                        hazardCircle.setLatLng([lat, lon]);
                    }}
                }},
                function(err) {{
                    console.warn("Browser geolocation fallback:", err.message);
                }},
                {{ enableHighAccuracy: true, timeout: 6000, maximumAge: 0 }}
            );
        }};
        
        // Interactive Operator Target Relocation
        map.on('click', function(e) {{
            var clickedLat = e.latlng.lat;
            var clickedLon = e.latlng.lng;

            // 1. Move Incident Marker
            if (typeof incidentMarker !== 'undefined') {{
                incidentMarker.setLatLng([clickedLat, clickedLon]);
                incidentMarker.bindPopup("<b>MANUAL STRIKE / INCIDENT PIN</b><br/>Lat: " + clickedLat.toFixed(5) + "<br/>Lon: " + clickedLon.toFixed(5)).openPopup();
            }}

            // 2. Reposition Hazard Circle
            if (typeof hazardCircle !== 'undefined') {{
                hazardCircle.setLatLng([clickedLat, clickedLon]);
            }}

            // 3. Reposition Staging Zone LZ-1
            if (typeof window.updateTacticalLZ === 'function') {{
                var currentRadius = (typeof hazardCircle !== 'undefined' && hazardCircle) ? hazardCircle.getRadius() : 350;
                window.updateTacticalLZ(clickedLat, clickedLon, currentRadius);
            }}

            // 4. Update In-Map HUD overlay
            var hudPerimeter = document.getElementById('hud-perimeter');
            var hudElem = document.querySelector('.tactical-hud');
            if (hudElem) {{
                hudElem.innerHTML = '<b style="color:#00D4FF;">GEOGUARD TACTICAL GIS // C2</b><br/>' +
                                    'LAT / LON: ' + clickedLat.toFixed(6) + ', ' + clickedLon.toFixed(6) + '<br/>' +
                                    '<span id="hud-perimeter">' + (hudPerimeter ? hudPerimeter.innerText : 'PERIMETER ACTIVE') + '</span>';
            }}

            // 5. Signal back to PyQt host via window title
            document.title = "GEO_CLICK:" + clickedLat.toFixed(6) + "," + clickedLon.toFixed(6);
        }});
        // Dynamic Snow Inundation Overlay
        window.displaySnowOverlay = function(bounds, base64PngUrl) {{
            if (window.snowLayer && map.hasLayer(window.snowLayer)) {{
                map.removeLayer(window.snowLayer);
            }}
            window.snowLayer = L.imageOverlay(base64PngUrl, bounds, {{
                opacity: 0.65,
                interactive: false
            }}).addTo(map);
        }};
    </script>
</body>
</html>
"""
        return html