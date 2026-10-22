import heapq
import cv2
import numpy as np

class EvacuationPathfinder:
    """A* Pathfinding engine that calculates lowest-cost egress vectors around dynamic hazards."""

    @staticmethod
    def compute_evacuation_overlay(image_np, hazard_mask, depth_map, start_pt=None, end_pt=None):
        h, w, _ = image_np.shape
        overlay = image_np.copy()

        # Build Cost Matrix (Penalizes hazardous terrain and vertical walls)
        cost_grid = np.ones((h, w), dtype=np.float32) * 1.0
        cost_grid[hazard_mask > 0] += 60.0
        cost_grid[depth_map < 0.15] += 100.0

        # Defaults if points are unspecified (Villa entrance -> Bottom-left perimeter)
        if start_pt is None:
            start_pt = (int(h * 0.58), int(w * 0.44))
        if end_pt is None:
            end_pt = (int(h * 0.88), int(w * 0.12))

        # Clamp points within boundary bounds
        start_pt = (max(0, min(h - 1, int(start_pt[0]))), max(0, min(w - 1, int(start_pt[1]))))
        end_pt = (max(0, min(h - 1, int(end_pt[0]))), max(0, min(w - 1, int(end_pt[1]))))

        # Downsample grid for real-time calculation (<15ms response)
        scale = 4
        gh, gw = h // scale, w // scale
        small_cost = cv2.resize(cost_grid, (gw, gh), interpolation=cv2.INTER_AREA)

        s_node = (start_pt[0] // scale, start_pt[1] // scale)
        e_node = (end_pt[0] // scale, end_pt[1] // scale)

        dist = {s_node: 0.0}
        pq = [(0.0, s_node)]
        parent = {}

        while pq:
            d, curr = heapq.heappop(pq)
            if curr == e_node:
                break
            if d > dist.get(curr, float('inf')):
                continue

            cy, cx = curr
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < gh and 0 <= nx < gw:
                    step_cost = small_cost[ny, nx] * (1.414 if dy != 0 and dx != 0 else 1.0)
                    new_dist = d + step_cost
                    if new_dist < dist.get((ny, nx), float('inf')):
                        dist[(ny, nx)] = new_dist
                        parent[(ny, nx)] = curr
                        heapq.heappush(pq, (new_dist, (ny, nx)))

        # Reconstruct vector path
        curr = e_node
        path = []
        while curr in parent:
            path.append((curr[1] * scale, curr[0] * scale))
            curr = parent[curr]
        path.append((s_node[1] * scale, s_node[0] * scale))
        path.reverse()

        # Render Neon Tactical HUD Layer
        hud_layer = np.zeros_like(image_np, dtype=np.uint8)

        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i + 1]
            cv2.line(hud_layer, pt1, pt2, (0, 255, 180), 4, lineType=cv2.LINE_AA)
            if i % 5 == 0:
                cv2.circle(hud_layer, pt1, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # Dynamic Start Beacon (Cyan) & Extraction Beacon (Green)
        cv2.circle(hud_layer, path[0], 8, (0, 220, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(hud_layer, "START [EVAC]", (path[0][0] - 25, max(14, path[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.circle(hud_layer, path[-1], 10, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(hud_layer, "EXTRACT [LZ-1]", (path[-1][0] - 35, min(h - 8, path[-1][1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        final_overlay = cv2.addWeighted(overlay, 1.0, hud_layer, 0.90, 0)
        return final_overlay, len(path) * scale

class ThermalReconEngine:
    """Generates military-grade FLIR/Thermal Reconnaissance HUD from visual and depth telemetry."""

    @staticmethod
    def generate_flir_overlay(image_np, depth_norm, hazard_mask, severity):
        h, w, _ = image_np.shape

        # 1. Compute apparent surface temperature map
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Hazard zones (water/ice/fissure) are thermal sinks (colder)
        temp_map = (gray * 0.45) + ((1.0 - depth_norm) * 255.0 * 0.40)
        temp_map[hazard_mask > 0] *= (0.35 - severity * 0.15)
        temp_map = np.clip(temp_map, 0, 255).astype(np.uint8)

        # 2. Apply FLIR Inferno thermal spectrum
        thermal_rgb = cv2.applyColorMap(temp_map, cv2.COLORMAP_INFERNO)
        thermal_rgb = cv2.cvtColor(thermal_rgb, cv2.COLOR_BGR2RGB)

        # 3. Render Tactical FLIR HUD Crosshair & Calibration Legends
        hud = thermal_rgb.copy()
        
        # Crosshair in center
        cx, cy = w // 2, h // 2
        cv2.line(hud, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(hud, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(hud, (cx, cy), 8, (0, 255, 255), 1, cv2.LINE_AA)

        # Tactical HUD Overlays
        cv2.putText(hud, "SENSOR: FLIR LWIR 14um", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"CAL: -12C [COLD-SINK] / +42C [CORE]", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, "MODE: THERMAL HAZARD DETECT", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

        # Temperature scale bar on right edge
        bar_x = w - 24
        for y_idx in range(60, h - 60):
            ratio = 1.0 - ((y_idx - 60) / (h - 120))
            color_val = int(ratio * 255)
            c_sample = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0][0]
            cv2.line(hud, (bar_x, y_idx), (bar_x + 12, y_idx), (int(c_sample[2]), int(c_sample[1]), int(c_sample[0])), 1)

        cv2.putText(hud, "+42C", (bar_x - 34, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(hud, "-12C", (bar_x - 34, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        return hud
    
# --- Compatibility Aliases for desktop_app.py ---
class DynamicPathfinder:
    @staticmethod
    def compute_egress_corridor(clean_rgb_np, hazard_mask, depth_norm, start_override=None, end_override=None):
        tactical_np, path_dist = EvacuationPathfinder.compute_evacuation_overlay(
            clean_rgb_np,
            hazard_mask,
            depth_norm,
            start_pt=start_override,
            end_pt=end_override
        )
        # Returns (tactical_np, distance_meters, clean_rgb_np)
        return tactical_np, float(path_dist) * 0.35, clean_rgb_np