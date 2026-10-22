"""
GeoGuard Autonomous Asset Allocation & Deployment Optimizer
Solves optimal resource dispatch using the Hungarian Algorithm (Linear Sum Assignment).
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

@dataclass
class RescueAsset:
    asset_id: str
    name: str
    asset_type: str       # "AMPHIBIOUS", "AERIAL_UAV", "PARAMEDIC", "BOAT"
    base_speed_kmh: float
    max_payload_kg: float
    status: str           # "STANDBY", "DEPLOYED", "MAINTENANCE"
    location_xy: Tuple[float, float]  # Normalized coordinate (0.0 to 1.0)


@dataclass
class TriageTarget:
    target_id: str
    label: str
    severity_level: float  # 0.0 to 1.0
    required_asset_type: str
    location_xy: Tuple[float, float]  # Normalized coordinate (0.0 to 1.0)
    urgency_tier: str      # "CRITICAL", "URGENT", "MONITORING"


class AssetAllocationOptimizer:
    """Computes cost-matrix transit times and optimizes multi-agent dispatch."""

    def __init__(self):
        # Default Station Inventory
        self.inventory: List[RescueAsset] = [
            RescueAsset("ASSET-01", "Zodiac Heavy Inflatable", "BOAT", 28.0, 600.0, "STANDBY", (0.05, 0.90)),
            RescueAsset("ASSET-02", "Zodiac Rapid Raft", "BOAT", 35.0, 350.0, "STANDBY", (0.08, 0.95)),
            RescueAsset("ASSET-03", "SkyLifter Medevac UAV", "AERIAL_UAV", 65.0, 45.0, "STANDBY", (0.10, 0.10)),
            RescueAsset("ASSET-04", "All-Terrain Amphibious Track", "AMPHIBIOUS", 22.0, 1200.0, "STANDBY", (0.85, 0.90)),
            RescueAsset("ASSET-05", "Tactical Triage Paramedics", "PARAMEDIC", 45.0, 150.0, "STANDBY", (0.90, 0.15)),
        ]

    def extract_triage_zones_from_mask(self, mask: np.ndarray, severity: float) -> List[TriageTarget]:
        """Auto-generates triage priority targets from synthesized inundation/damage masks."""
        targets = []
        if mask is None or np.sum(mask) == 0:
            # Fallback standard targets
            return [
                TriageTarget("ZONE-ALPHA", "Flooded Foundation / Entry", severity, "BOAT", (0.35, 0.70), "CRITICAL"),
                TriageTarget("ZONE-BRAVO", "Roof Refuge Civilian LZ", severity * 0.85, "AERIAL_UAV", (0.50, 0.40), "URGENT"),
                TriageTarget("ZONE-CHARLIE", "Submerged Ingress Roadway", severity * 0.65, "AMPHIBIOUS", (0.20, 0.85), "MONITORING"),
            ]

        # Find connected centroid components of active hazard zones
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
        h, w = mask.shape

        zone_names = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO"]
        # Take the largest connected zones
        sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:3]

        for i, idx in enumerate(sorted_indices):
            cx = float(centroids[idx + 1][0]) / float(w)
            cy = float(centroids[idx + 1][1]) / float(h)
            area_ratio = float(stats[idx + 1][cv2.CC_STAT_AREA]) / float(h * w)

            urgency = "CRITICAL" if area_ratio > 0.15 else ("URGENT" if area_ratio > 0.05 else "MONITORING")
            asset_req = "BOAT" if cy > 0.55 else "AERIAL_UAV"

            targets.append(
                TriageTarget(
                    target_id=f"ZONE-{zone_names[i % len(zone_names)]}",
                    label=f"Inundation Sector {zone_names[i % len(zone_names)]}",
                    severity_level=float(severity * (1.0 - 0.15 * i)),
                    required_asset_type=asset_req,
                    location_xy=(cx, cy),
                    urgency_tier=urgency
                )
            )

        return targets

    def optimize_dispatch(self, targets: List[TriageTarget]) -> List[Dict]:
        """
        Builds an M x N cost matrix and executes the Hungarian Algorithm
        to minimize total transit time weighted by target severity.
        """
        available_assets = [a for a in self.inventory if a.status == "STANDBY"]
        if not available_assets or not targets:
            return []

        num_assets = len(available_assets)
        num_targets = len(targets)

        # Cost matrix: Rows = Assets, Columns = Targets
        cost_matrix = np.zeros((num_assets, num_targets), dtype=np.float32)

        for a_idx, asset in enumerate(available_assets):
            for t_idx, target in enumerate(targets):
                # Euclidean distance in tactical grid space (km proxy)
                dx = (asset.location_xy[0] - target.location_xy[0]) * 5.0  # 5km tactical radius
                dy = (asset.location_xy[1] - target.location_xy[1]) * 5.0
                dist_km = np.sqrt(dx * dx + dy * dy)

                # Transit time in minutes
                eta_minutes = (dist_km / max(1.0, asset.base_speed_kmh)) * 60.0

                # Capability compatibility bonus / penalty
                type_mismatch_penalty = 1.0
                if asset.asset_type != target.required_asset_type:
                    type_mismatch_penalty = 2.8

                # Priority weighting: high severity zones penalize long ETAs
                urgency_weight = 1.5 if target.urgency_tier == "CRITICAL" else 1.0
                cost_matrix[a_idx, t_idx] = eta_minutes * type_mismatch_penalty * (1.0 / (target.severity_level + 0.1))

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        for r, c in zip(row_ind, col_ind):
            assigned_asset = available_assets[r]
            assigned_target = targets[c]

            dx = (assigned_asset.location_xy[0] - assigned_target.location_xy[0]) * 5.0
            dy = (assigned_asset.location_xy[1] - assigned_target.location_xy[1]) * 5.0
            dist_km = float(np.sqrt(dx * dx + dy * dy))
            eta_min = round((dist_km / max(1.0, assigned_asset.base_speed_kmh)) * 60.0, 1)

            assignments.append({
                "target_id": assigned_target.target_id,
                "target_label": assigned_target.label,
                "urgency": assigned_target.urgency_tier,
                "asset_id": assigned_asset.asset_id,
                "asset_name": assigned_asset.name,
                "asset_type": assigned_asset.asset_type,
                "distance_km": round(dist_km, 2),
                "eta_minutes": eta_min,
                "cost_score": round(float(cost_matrix[r, c]), 2)
            })

        return assignments