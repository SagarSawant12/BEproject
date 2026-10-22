import os
import sys
import heapq
import subprocess
import datetime
import cv2
import numpy as np
import torch

# MUST be at the very top before PyQt6 / WebEngine is imported anywhere:
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
    "--disable-gpu "
    "--disable-gpu-compositing "
    "--disable-software-rasterizer "
    "--disable-gpu-rasterization "
    "--disable-accelerated-2d-canvas "
    "--no-sandbox"
)

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtCore import Qt, QTimer
from tactical_audio_generator import TacticalAudioDispatcher
from PIL import Image, ImageEnhance
from asset_allocator import AssetAllocationOptimizer
from terrain_3d_viewer import Tactical3DTerrainDialog
from PyQt6.QtWebEngineWidgets import QWebEngineView
from gis_georeferencer import TacticalGISManager
from snow_classifier import SentinelSnowClassifier
from asset_allocator import AssetAllocationOptimizer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QSlider, QProgressBar,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog,
    QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QSizePolicy
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QSlider, QProgressBar,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFileDialog,
    QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QTextDocument, QPageSize
from PyQt6.QtPrintSupport import QPrinter
from pathfinder import DynamicPathfinder, ThermalReconEngine
from voice_dispatch import VoiceDispatchThread
from transformers import pipeline, CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DPMSolverMultistepScheduler

try:
    import torch_directml
    dml_device = torch_directml.device()
    DEVICE_NAME = f"AMD DirectML [{torch_directml.device_name(0)}]"
except Exception:
    dml_device = torch.device("cpu")
    DEVICE_NAME = "CPU Fallback"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ---------------------------------------------------------------------------
# Global Persistent Model Registry (DirectML Acceleration)
# ---------------------------------------------------------------------------
class ModelRegistry:
    depth_estimator = None
    clip_processor = None
    clip_model = None
    controlnet = None
    pipe = None
    device = dml_device

    @classmethod
    def initialize(cls, log_callback=print):
        if cls.pipe is not None:
            return

        log_callback(f"Loading persistent models into {DEVICE_NAME}...")

        cls.depth_estimator = pipeline(
            "depth-estimation", 
            model="Intel/dpt-large", 
            device=-1
        )

        cls.clip_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        cls.clip_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        log_callback("Configuring ControlNet & Diffusion Engine for AMD DirectML...")
        cls.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float32
        ).to(cls.device)

        cls.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=cls.controlnet,
            torch_dtype=torch.float32,
            use_safetensors=False
        ).to(cls.device)

        cls.pipe.scheduler = DPMSolverMultistepScheduler.from_config(cls.pipe.scheduler.config)
        log_callback("Pipeline components initialized in memory!")

# ---------------------------------------------------------------------------
# Tactical Engines: Evacuation Pathfinder, SITREP & Voice Dispatch
# ---------------------------------------------------------------------------
class EvacuationPathfinder:
    @staticmethod
    def compute_evacuation_overlay(image_np, hazard_mask, depth_map, start_pt=None, end_pt=None):
        h, w, _ = image_np.shape
        overlay = image_np.copy()

        cost_grid = np.ones((h, w), dtype=np.float32) * 1.0
        cost_grid[hazard_mask > 0] += 60.0
        cost_grid[depth_map < 0.15] += 100.0

        if start_pt is None:
            start_pt = (int(h * 0.58), int(w * 0.44))
        if end_pt is None:
            end_pt = (int(h * 0.88), int(w * 0.12))

        # Clamp points within boundary bounds
        start_pt = (max(0, min(h - 1, int(start_pt[0]))), max(0, min(w - 1, int(start_pt[1]))))
        end_pt = (max(0, min(h - 1, int(end_pt[0]))), max(0, min(w - 1, int(end_pt[1]))))

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

        curr = e_node
        path = []
        while curr in parent:
            path.append((curr[1] * scale, curr[0] * scale))
            curr = parent[curr]
        path.append((s_node[1] * scale, s_node[0] * scale))
        path.reverse()

        hud_layer = np.zeros_like(image_np, dtype=np.uint8)

        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i + 1]
            cv2.line(hud_layer, pt1, pt2, (0, 255, 180), 4, lineType=cv2.LINE_AA)
            if i % 5 == 0:
                cv2.circle(hud_layer, pt1, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        cv2.circle(hud_layer, path[0], 7, (0, 200, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(hud_layer, path[-1], 9, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(hud_layer, "EXTRACT POINT [LZ-1]", (path[-1][0] - 30, path[-1][1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        final_overlay = cv2.addWeighted(overlay, 1.0, hud_layer, 0.90, 0)
        return final_overlay, len(path) * scale

class TacticalReportGenerator:
    """Multilingual Tactical SITREP & Dispatch Generator."""

    LOCALES = {
        "English": {
            "title": "GEOGUARD DISASTER RESPONSE REPORT",
            "subtitle": "AUTOMATED MISSION INTELLIGENCE & EVACUATION DOSSIER",
            "threat_levels": {"crit": "CRITICAL / LEVEL 3", "watch": "WATCH / LEVEL 2", "adv": "ADVISORY / LEVEL 1"},
            "sec1": "1. Incident & Geometric Telemetry",
            "sec2": "2. Visual Reconnaissance & Tactical Evacuation HUD",
            "sec3": "3. Operational Directives & Defense Actions",
            "scenario": "DISASTER SCENARIO",
            "severity": "SEVERITY COEFFICIENT (ALPHA)",
            "hazard_cov": "SURFACE HAZARD EXTENT",
            "corridor": "EVACUATION PASSAGEWAY",
            "corridor_val": "USER-DESIGNATED VECTOR [TARGET -> LZ-1]",
            "distance": "TOTAL TRANSIT DISTANCE",
            "transit_time": "ESTIMATED EVACUATION TIME",
            "caption": "Topological Hazard Prediction & Computed A* Egress Vector [LZ-1]",
            "directives": (
                "• Evacuation Route: Civilians and field personnel must traverse strictly along the marked Alpha-1 neon corridor.<br>"
                "• Structural Integrity: Primary building load-bearing envelope verified intact. Exterior ground planes compromised.<br>"
                "• Acoustic Dispatch: Automated offline voice announcement dispatched across emergency response frequencies."
            ),
            "footer": "CONFIDENTIAL — GENERATED BY GEOGUARD TACTICAL SYNTHESIS PLATFORM (IEEE MISSION COMMAND)",
            "voice": "Tactical SITREP broadcast. Threat level {threat}. Hazard accumulation is at {cov} percent. Evacuation distance is {dist} meters. Follow marked perimeter corridors immediately."
        },
        "Spanish": {
            "title": "INFORME DE RESPUESTA A DESASTRES GEOGUARD",
            "subtitle": "DOSSIER AUTOMATIZADO DE INTELIGENCIA Y EVACUACIÓN",
            "threat_levels": {"crit": "CRÍTICO / NIVEL 3", "watch": "ALERTA / NIVEL 2", "adv": "ADVERTENCIA / NIVEL 1"},
            "sec1": "1. Telemetría Geométrica y del Incidente",
            "sec2": "2. Reconocimiento Visual y HUD de Evacuación Táctica",
            "sec3": "3. Directivas Operativas y Acciones de Defensa",
            "scenario": "ESCENARIO DE DESASTRE",
            "severity": "COEFICIENTE DE SEVERIDAD (ALFA)",
            "hazard_cov": "EXTENSIÓN DEL PELIGRO EN SUPERFICIE",
            "corridor": "CORREDOR DE EVACUACIÓN",
            "corridor_val": "VECTOR DESIGNADO POR EL USUARIO [OBJETIVO -> LZ-1]",
            "distance": "DISTANCIA TOTAL DE TRÁNSITO",
            "transit_time": "TIEMPO ESTIMADO DE EVACUACIÓN",
            "caption": "Predicción de Peligro Topológico y Vector de Salida A* [LZ-1]",
            "directives": (
                "• Ruta de evacuación: El personal debe circular estrictamente por el corredor de neón Alpha-1.<br>"
                "• Integridad estructural: Estructura portante principal intacta. Terrenos exteriores comprometidos.<br>"
                "• Transmisión acústica: Anuncio de voz automatizado emitido por frecuencias de emergencia."
            ),
            "footer": "CONFIDENCIAL — GENERADO POR LA PLATAFORMA TÁCTICA GEOGUARD",
            "voice": "Transmisión táctica SITREP. Nivel de amenaza {threat}. Acumulación de peligro al {cov} por ciento. Siga las rutas marcadas inmediatamente."
        },
        "Hindi": {
            "title": "जियोगार्ड आपदा प्रतिक्रिया रिपोर्ट",
            "subtitle": "स्वचालित मिशन इंटेलिजेंस एवं निकासी दस्तावेज़",
            "threat_levels": {"crit": "गंभीर / स्तर 3", "watch": "चेतावनी / स्तर 2", "adv": "सलाहकार / स्तर 1"},
            "sec1": "1. घटना एवं ज्यामितीय टेलीमेट्री",
            "sec2": "2. दृश्य सर्वेक्षण एवं सामरिक निकासी HUD",
            "sec3": "3. परिचालन निर्देश एवं बचाव कार्य",
            "scenario": "आपदा परिदृश्य",
            "severity": "तीव्रता गुणांक (अल्फा)",
            "hazard_cov": "सतह पर खतरे का विस्तार",
            "corridor": "निकासी गलियारा",
            "corridor_val": "उपयोगकर्ता-निर्धारित वेक्टर [प्रारंभ -> LZ-1]",
            "distance": "कुल आवागमन दूरी",
            "transit_time": "अनुमानित निकासी समय",
            "caption": "टोपोलॉजिकल खतरे का विश्लेषण एवं गणना किया गया A* निकास पथ [LZ-1]",
            "directives": (
                "• निकासी मार्ग: सभी नागरिक केवल चिह्नित अल्फा-1 नियॉन गलियारे का उपयोग करें।<br>"
                "• संरचनात्मक स्थिति: मुख्य भवन सुरक्षित है। बाहरी फर्श और मार्ग जलमग्न हैं।<br>"
                "• ऑडियो प्रसारण: आपातकालीन आवृत्तियों पर स्वचालित उद्घोषणा प्रसारित की गई है।"
            ),
            "footer": "गोपनीय — जियोगार्ड सामरिक विश्लेषण प्रणाली द्वारा निर्मित",
            "voice": "Aapaatkaaleen soochanaa. Khatre kaa star {threat}. Satah par jokhim {cov} pratishat hai. Nikaasi doori lagbhag {dist} meter hai. Chinhit surakshit marg par turant aage badhein."
        },
        "French": {
            "title": "RAPPORT DE RÉPONSE AUX CATASTROPHES GEOGUARD",
            "subtitle": "DOSSIER AUTOMATISÉ DE RENSEIGNEMENT ET D'ÉVACUATION",
            "threat_levels": {"crit": "CRITIQUE / NIVEAU 3", "watch": "VIGILANCE / NIVEAU 2", "adv": "AVIS / NIVEAU 1"},
            "sec1": "1. Télémétrie Géométrique et d'Incident",
            "sec2": "2. Reconnaissance Visuelle et HUD d'Évacuation",
            "sec3": "3. Directives Opérationnelles et Actions de Défense",
            "scenario": "SCÉNARIO DE CATASTROPHE",
            "severity": "COEFFICIENT DE SÉVÉRITÉ (ALPHA)",
            "hazard_cov": "ÉTENDUE DU DANGER EN SURFACE",
            "corridor": "COULOIR D'ÉVACUATION",
            "corridor_val": "VECTEUR DÉFINI PAR L'UTILISATEUR [CIBLE -> LZ-1]",
            "distance": "DISTANCE TOTALE DE TRANSIT",
            "transit_time": "TEMPS ESTIMÉ D'ÉVACUATION",
            "caption": "Prédiction Topologique du Danger et Vecteur de Sortie A* [LZ-1]",
            "directives": (
                "• Itinéraire d'évacuation: Le personnel doit emprunter strictement le couloir néon Alpha-1.<br>"
                "• Intégrité structurelle: Structure principale confirmée intacte. Surfaces extérieures compromises.<br>"
                "• Diffusion acoustique: Annonce vocale automatisée diffusée sur les fréquences d'urgence."
            ),
            "footer": "CONFIDENTIEL — GÉNÉRÉ PAR LA PLATEFORME TACTIQUE GEOGUARD",
            "voice": "Diffusion SITREP tactique. Niveau de menace {threat}. Accumulation de danger à {cov} pour cent. Suivez immédiatement le couloir d'évacuation."
        }
    }

    @classmethod
    def generate_report(cls, scenario, severity, impact_coverage_pct, path_dist, lang="English"):
        loc = cls.LOCALES.get(lang, cls.LOCALES["English"])

        tier_key = "crit" if severity >= 0.70 else "watch" if severity >= 0.40 else "adv"
        risk_tier = loc["threat_levels"][tier_key]
        est_transit_time = max(0.5, (path_dist * 0.08) * (1.0 + severity * 0.8))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Convert <br> tags outside the f-string to prevent backslash syntax errors
        clean_directives = loc['directives'].replace('<br>', '\n')
        sec1_header = loc['sec1'].upper()
        sec3_header = loc['sec3'].upper()

        report_text = f"""================================================================================
                    {loc['title']}
================================================================================
{loc['scenario']:<20} : {scenario.upper()}
TIMESTAMP            : {timestamp}
DEFCON THREAT LEVEL  : {risk_tier}
{loc['severity']:<20} : {severity:.2f}
{loc['hazard_cov']:<20} : {impact_coverage_pct:.1f}%
{loc['corridor']:<20} : {loc['corridor_val']}
{loc['distance']:<20} : ~{path_dist:.1f} METERS (EST. TRANSIT: {est_transit_time:.1f} MIN)

[{sec1_header}]
- Status: Telemetry metrics confirmed by neural elevation grid.
- Surface Inundation: {impact_coverage_pct:.1f}% area coverage.

[{sec3_header}]
{clean_directives}
================================================================================"""

        voice_script = loc["voice"].format(
            threat=risk_tier.split('/')[0].strip(),
            cov=int(impact_coverage_pct),
            dist=int(path_dist)
        )
        return report_text, voice_script, timestamp, risk_tier, est_transit_time, loc
# ---------------------------------------------------------------------------
# Interactive Tactical Label (Captures Left-Click Start & Right-Click LZ)
# ---------------------------------------------------------------------------
class InteractiveTacticalLabel(QLabel):
    point_clicked = pyqtSignal(int, int, str)
    hover_moved = pyqtSignal(int, int)
    hover_left = pyqtSignal()
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        # CRITICAL: Prevent QLabel from resizing itself when new pixmaps are set
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.cursor_pos = None

        # Split curtain states
        self.split_active = False
        self.split_ratio = 0.5
        self.is_dragging_curtain = False
        self.pixmap_before = None
        self.pixmap_after = None

        # Tactical Zoom states (1.0 = 100%)
        self.zoom_factor = 1.0
        self.raw_pixmap = None

    def set_image(self, pixmap):
        """Sets the raw pixmap and requests a clean repaint without altering widget geometry."""
        self.raw_pixmap = pixmap
        self.update()

    def set_zoom(self, factor: float):
        self.zoom_factor = max(0.5, min(3.0, factor))
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom = min(3.0, self.zoom_factor + 0.1)
        else:
            new_zoom = max(0.5, self.zoom_factor - 0.1)
        self.set_zoom(new_zoom)
        self.zoom_changed.emit(self.zoom_factor)
        event.accept()

    def set_split_images(self, pixmap_before, pixmap_after):
        self.pixmap_before = pixmap_before
        self.pixmap_after = pixmap_after
        self.split_active = (pixmap_before is not None and pixmap_after is not None)
        self.update()

    def update_split_after(self, pixmap_after):
        self.pixmap_after = pixmap_after
        self.update()

    def disable_split(self):
        self.split_active = False
        self.update()

    def mousePressEvent(self, event):
        pos = event.position()
        px, py = int(pos.x()), int(pos.y())
        w, h = self.width(), self.height()

        # Split-screen curtain drag handling
        if self.split_active and self.pixmap_after:
            img_w = self.pixmap_after.width()
            img_h = self.pixmap_after.height()
            base_scale = min(w / float(max(1, img_w)), h / float(max(1, img_h)))
            target_w = max(1, int(img_w * base_scale * self.zoom_factor))
            ox = (w - target_w) // 2
            divider_x = max(ox, min(ox + target_w, ox + int(target_w * self.split_ratio)))

            if abs(px - divider_x) <= 24 and event.button() == Qt.MouseButton.LeftButton:
                self.is_dragging_curtain = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
                event.accept()
                return
            event.accept()
            return

        # Standard mode click: Map widget coordinates to underlying image pixels
        target_pix = self.raw_pixmap or self.pixmap()
        if target_pix and not target_pix.isNull():
            img_w = target_pix.width()
            img_h = target_pix.height()
            base_scale = min(w / float(max(1, img_w)), h / float(max(1, img_h)))
            target_w = max(1, int(img_w * base_scale * self.zoom_factor))
            target_h = max(1, int(img_h * base_scale * self.zoom_factor))
            ox = (w - target_w) // 2
            oy = (h - target_h) // 2

            # Only accept clicks within the active image boundary
            if ox <= px <= ox + target_w and oy <= py <= oy + target_h:
                norm_x = int((px - ox) * (img_w / float(target_w)))
                norm_y = int((py - oy) * (img_h / float(target_h)))
                norm_x = max(0, min(img_w - 1, norm_x))
                norm_y = max(0, min(img_h - 1, norm_y))

                if event.button() == Qt.MouseButton.LeftButton:
                    self.point_clicked.emit(norm_x, norm_y, "START")
                elif event.button() == Qt.MouseButton.RightButton:
                    self.point_clicked.emit(norm_x, norm_y, "END")
                event.accept()
                return

        event.accept()

    def mouseMoveEvent(self, event):
        pos = event.position()
        px, py = int(pos.x()), int(pos.y())
        w, h = self.width(), self.height()
        self.cursor_pos = (px, py)

        if self.split_active and self.pixmap_after:
            img_w = self.pixmap_after.width()
            img_h = self.pixmap_after.height()
            base_scale = min(w / float(max(1, img_w)), h / float(max(1, img_h)))
            target_w = max(1, int(img_w * base_scale * self.zoom_factor))
            ox = (w - target_w) // 2

            if self.is_dragging_curtain:
                rel_px = px - ox
                self.split_ratio = max(0.05, min(0.95, rel_px / float(max(1, target_w))))
                self.update()
            else:
                divider_x = max(ox, min(ox + target_w, ox + int(target_w * self.split_ratio)))
                if abs(px - divider_x) <= 24:
                    self.setCursor(Qt.CursorShape.SplitHCursor)
                else:
                    self.setCursor(Qt.CursorShape.CrossCursor)

        self.hover_moved.emit(px, py)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_dragging_curtain:
            self.is_dragging_curtain = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.cursor_pos = None
        self.hover_left.emit()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # Tactical dark letterbox background
        painter.fillRect(0, 0, w, h, QColor("#030712"))

        # --- Mode 1: Split Curtain ---
        if self.split_active and self.pixmap_before and self.pixmap_after:
            img_w = self.pixmap_after.width()
            img_h = self.pixmap_after.height()

            base_scale = min(w / float(max(1, img_w)), h / float(max(1, img_h)))
            effective_scale = base_scale * self.zoom_factor
            target_w = max(1, int(img_w * effective_scale))
            target_h = max(1, int(img_h * effective_scale))

            ox = (w - target_w) // 2
            oy = (h - target_h) // 2

            scaled_before = self.pixmap_before.scaled(target_w, target_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            scaled_after = self.pixmap_after.scaled(target_w, target_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)

            divider_x = max(ox, min(ox + target_w, ox + int(target_w * self.split_ratio)))

            # Before partition
            painter.save()
            painter.setClipRect(ox, oy, max(0, divider_x - ox), target_h)
            painter.drawPixmap(ox, oy, scaled_before)
            painter.restore()

            # After partition
            painter.save()
            painter.setClipRect(divider_x, oy, max(0, (ox + target_w) - divider_x), target_h)
            painter.drawPixmap(ox, oy, scaled_after)
            painter.restore()

            # Cyan Divider & Handle
            pen = QPen(QColor("#38BDF8"), 2)
            painter.setPen(pen)
            painter.drawLine(divider_x, oy, divider_x, oy + target_h)

            badge_h = 36
            badge_w = 18
            badge_y = oy + (target_h - badge_h) // 2
            painter.setBrush(QColor("#0B1329"))
            painter.setPen(QPen(QColor("#38BDF8"), 1))
            painter.drawRoundedRect(divider_x - (badge_w // 2), badge_y, badge_w, badge_h, 4, 4)

            painter.setPen(QColor("#38BDF8"))
            painter.drawLine(divider_x - 3, badge_y + 11, divider_x - 3, badge_y + 25)
            painter.drawLine(divider_x + 3, badge_y + 11, divider_x + 3, badge_y + 25)

            font = painter.font()
            font.setPointSize(8)
            font.setBold(True)
            painter.setFont(font)

            if divider_x - ox > 140:
                painter.fillRect(ox + 8, oy + 8, 140, 20, QColor(3, 7, 18, 210))
                painter.setPen(QColor("#38BDF8"))
                painter.drawText(ox + 14, oy + 22, "PRE-INCIDENT OPTICAL")

            if (ox + target_w) - divider_x > 145:
                painter.fillRect(divider_x + 8, oy + 8, 145, 20, QColor(3, 7, 18, 210))
                painter.setPen(QColor("#F87171"))
                painter.drawText(divider_x + 14, oy + 22, "SYNTHESIZED DISASTER")

            painter.end()
            return

        # --- Mode 2: Standard Tactical Single View ---
        target_pix = self.raw_pixmap or self.pixmap()
        if target_pix and not target_pix.isNull():
            img_w = target_pix.width()
            img_h = target_pix.height()

            base_scale = min(w / float(max(1, img_w)), h / float(max(1, img_h)))
            effective_scale = base_scale * self.zoom_factor
            target_w = max(1, int(img_w * effective_scale))
            target_h = max(1, int(img_h * effective_scale))

            ox = (w - target_w) // 2
            oy = (h - target_h) // 2

            scaled = target_pix.scaled(target_w, target_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(ox, oy, scaled)

        painter.end()
        
class VoiceDispatchThread(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            clean_text = self.text.replace("'", "").replace('"', '')
            # PowerShell command that prioritizes Hindi voices (like Kalpana) if present
            ps_script = (
                "$s = New-Object -ComObject SAPI.SpVoice; "
                "$h = $s.GetVoices() | Where-Object { $_.GetDescription() -like '*Hindi*' -or $_.GetDescription() -like '*India*' } | Select-Object -First 1; "
                "if ($h) { $s.Voice = $h }; "
                f"$s.Speak('{clean_text}')"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], shell=True)
        except Exception:
            pass
class ThermalReconEngine:
    @staticmethod
    def generate_flir_overlay(image_np, depth_norm, hazard_mask, severity):
        h, w, _ = image_np.shape
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

        temp_map = (gray * 0.45) + ((1.0 - depth_norm) * 255.0 * 0.40)
        temp_map[hazard_mask > 0] *= max(0.1, (0.35 - severity * 0.15))
        temp_map = np.clip(temp_map, 0, 255).astype(np.uint8)

        thermal_rgb = cv2.applyColorMap(temp_map, cv2.COLORMAP_INFERNO)
        thermal_rgb = cv2.cvtColor(thermal_rgb, cv2.COLOR_BGR2RGB)

        hud = thermal_rgb.copy()
        cx, cy = w // 2, h // 2
        cv2.line(hud, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(hud, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(hud, (cx, cy), 8, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(hud, "SENSOR: FLIR LWIR 14um", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, "CAL: -12C [COLD] / +42C [CORE]", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, "MODE: THERMAL HAZARD DETECT", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1, cv2.LINE_AA)

        bar_x = w - 24
        for y_idx in range(60, h - 60):
            ratio = 1.0 - ((y_idx - 60) / (h - 120))
            color_val = int(ratio * 255)
            c_sample = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0][0]
            cv2.line(hud, (bar_x, y_idx), (bar_x + 12, y_idx), (int(c_sample[2]), int(c_sample[1]), int(c_sample[0])), 1)

        cv2.putText(hud, "+42C", (bar_x - 34, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(hud, "-12C", (bar_x - 34, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        return hud
    
class TemporalHazardEngine:
    """Simulates multi-hour environmental spread and rerouting without diffusion re-inference."""

    @staticmethod
    def forecast_hazard(base_hazard_mask, depth_norm, time_minutes):
        if time_minutes == 0 or base_hazard_mask is None:
            return base_hazard_mask.copy()

        expansion_factor = time_minutes / 30.0
        k_size = int(expansion_factor * 5) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        dilated = cv2.dilate(base_hazard_mask.astype(np.uint8), kernel, iterations=1)
        slope_gate = (depth_norm >= np.percentile(depth_norm, max(10, int(60 - expansion_factor * 8))))
        temporal_mask = ((dilated > 0) | slope_gate).astype(np.uint8) * 255
        temporal_mask[depth_norm < 0.12] = 0
        return temporal_mask
    
class StructuralTriageEngine:
    """Evaluates multi-zone structural compromise, ingress safety, and rooftop extraction readiness."""

    @staticmethod
    def evaluate_zones(hazard_mask, depth_norm, severity):
        h, w = hazard_mask.shape
        
        # Define spatial elevation zones based on depth and vertical geometry
        zones = {
            "Zone A (Perimeter / Approach)": (hazard_mask[int(h * 0.70):, :], depth_norm[int(h * 0.70):, :]),
            "Zone B (Ground Floor / Ingress)": (hazard_mask[int(h * 0.45):int(h * 0.70), :], depth_norm[int(h * 0.45):int(h * 0.70), :]),
            "Zone C (Structural Envelope)": (hazard_mask[int(h * 0.20):int(h * 0.45), :], depth_norm[int(h * 0.20):int(h * 0.45), :]),
            "Zone D (Rooftop Safe-Haven)": (hazard_mask[:int(h * 0.20), :], depth_norm[:int(h * 0.20), :]),
        }

        results = []
        for zone_name, (h_slice, d_slice) in zones.items():
            total_px = h_slice.size
            if total_px == 0:
                continue

            submerged_ratio = float(np.count_nonzero(h_slice > 0)) / float(total_px)
            avg_depth_metric = float(np.mean(d_slice)) * 25.0 * severity  # Scaled to metric equivalent

            # Quantitative Risk Computation
            risk_score = min(100, int((submerged_ratio * 70.0) + (severity * 30.0)))
            
            if risk_score >= 70:
                triage = "CRITICAL / COMPROMISED"
                status_color = "#EF4444"  # Red
                passable = "NO"
            elif risk_score >= 35:
                triage = "WARNING / IMPEDED"
                status_color = "#F59E0B"  # Amber
                passable = "RESTRICTED"
            else:
                triage = "SECURE / VIABLE"
                status_color = "#10B981"  # Emerald
                passable = "YES"

            results.append({
                "zone": zone_name,
                "submerged_pct": f"{submerged_ratio * 100.0:.1f}%",
                "water_depth": f"{avg_depth_metric:.2f} m",
                "risk_score": f"{risk_score} / 100",
                "passable": passable,
                "triage": triage,
                "color": status_color
            })

        return results
    
# ---------------------------------------------------------------------------
# Background Worker Thread (Geometry-Preserving High-Fidelity Synthesis)
# ---------------------------------------------------------------------------
class DisasterWorker(QThread):
    finished = pyqtSignal(object, object, str, str, dict, object, object)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, image_path, disaster_type, target_prompt, user_prompt, severity):
        super().__init__()
        self.image_path = image_path
        self.disaster_type = disaster_type
        self.target_prompt = target_prompt
        self.user_prompt = user_prompt
        self.severity = float(severity)

    def run(self):
        try:
            ModelRegistry.initialize(self.log_signal.emit)
            device = ModelRegistry.device
            self.progress_signal.emit(15)

            # --- Step 1: Metric Depth & Normal Extraction ---
            self.log_signal.emit("Step 1/5: Extracting depth and structural boundaries...")
            init_img = Image.open(self.image_path).convert("RGB").resize((512, 512))
            init_np = np.array(init_img)
            hsv_init = cv2.cvtColor(init_np, cv2.COLOR_RGB2HSV)  # <-- ADD THIS LINE HERE
            depth_raw = np.array(ModelRegistry.depth_estimator(init_img)["depth"])
            depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)
            depth_norm = cv2.resize(depth_norm, (512, 512), interpolation=cv2.INTER_LINEAR)
            depth_control_img = Image.fromarray((depth_norm * 255).astype(np.uint8)).convert("RGB")
            self.progress_signal.emit(30)

            # =========================================================================
            # BRANCH 1: HEAVY RAINSTORM (HALO-FREE SKY, FLOODED GROUND & OPTICAL RAIN)
            # =========================================================================
            if "Rainstorm" in self.disaster_type or self.disaster_type == "Heavy Rainstorm (Dual-Branch)":
                self.log_signal.emit(f"Step 2/5: Calculating hydraulic accumulation for Severity α = {self.severity:.2f}...")
                res_np = init_np.copy().astype(np.float32)
                h, w, _ = res_np.shape

                # 1. Halo-Free Sky Gradation using Continuous Chromatic Weighting
                hsv_init = cv2.cvtColor(init_np, cv2.COLOR_RGB2HSV)
                b_over_r = (res_np[:, :, 2] - res_np[:, :, 0]) / (res_np[:, :, 2] + 1e-5)
                sky_prob = np.clip(b_over_r * 2.5, 0.0, 1.0) * (depth_norm < 0.22).astype(np.float32)

                # Protect green vegetation from sky darkening (prevents cyan border outline)
                is_foliage = (res_np[:, :, 1] > res_np[:, :, 0] * 1.05) & (res_np[:, :, 1] > res_np[:, :, 2])
                sky_prob[is_foliage] = 0.0
                sky_matte = cv2.GaussianBlur(sky_prob, (9, 9), 0)

                # Overcast sky tone shift
                storm_dark = sky_matte * (0.35 + self.severity * 0.40)
                target_r = 70.0 - self.severity * 20.0
                target_g = 80.0 - self.severity * 20.0
                target_b = 92.0 - self.severity * 15.0

                for c, target_c in enumerate([target_r, target_g, target_b]):
                    res_np[:, :, c] = res_np[:, :, c] * (1.0 - storm_dark) + (target_c * storm_dark)

                self.progress_signal.emit(50)

                # 2. Dynamic Ground Water Filling by Severity
                # Ground plane calculation (excluding vertical walls and trees)
                y_coords, _ = np.indices((512, 512))
                grad_y = np.abs(cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=3))
                flat_ground = (grad_y < np.percentile(grad_y, 75)) & (depth_norm >= 0.50) & (y_coords > 240) & (~is_foliage)

                # Water spreads across the patio floor as severity increases
                water_level_threshold = 0.90 - (self.severity * 0.42)
                submerged_ground = flat_ground & (depth_norm >= water_level_threshold)

                # Soft hydraulic blending
                flood_alpha = cv2.GaussianBlur(submerged_ground.astype(np.float32), (7, 7), 0)
                flood_factor = np.clip(flood_alpha * (0.30 + self.severity * 0.65), 0.0, 0.90)

                # Realistic water surface: inverted villa reflection + turquoise refraction
                flipped_refl = cv2.flip(res_np, 0)
                flipped_refl = cv2.GaussianBlur(flipped_refl, (13, 13), 0)
                water_tint = np.array([30.0, 140.0, 160.0], dtype=np.float32)

                for c in range(3):
                    water_pixel = (flipped_refl[:, :, c] * 0.40) + (water_tint[c] * 0.60)
                    res_np[:, :, c] = res_np[:, :, c] * (1.0 - flood_factor) + (water_pixel * flood_factor)

                self.progress_signal.emit(70)

                # 3. Dynamic Raindrop Surface Ripples on All Water Areas
                active_water = (flood_factor > 0.10) | ((depth_norm >= 0.70) & (y_coords > 340))
                ripple_layer = np.zeros((h, w), dtype=np.float32)
                num_ripples = int(30 + self.severity * 60)
                for _ in range(num_ripples):
                    cx = np.random.randint(0, int(w * 0.90))
                    cy = np.random.randint(int(h * 0.52), h)
                    radius = np.random.randint(4, 12)
                    cv2.ellipse(ripple_layer, (cx, cy), (radius, int(radius * 0.35)), 0, 0, 360, float(np.random.uniform(35, 80)), 1, cv2.LINE_AA)

                # 4. Translucent Directional Optical Rain Streaks (Natural, High Visibility)
                rain_layer = np.zeros((h, w), dtype=np.float32)
                num_streaks = int(1200 + self.severity * 1400)
                streak_len = int(10 + self.severity * 10)
                angle_rad = np.radians(78)
                dx = int(streak_len * np.cos(angle_rad))
                dy = int(streak_len * np.sin(angle_rad))

                rx = np.random.randint(0, w - max(1, dx), size=num_streaks)
                ry = np.random.randint(0, h - max(1, dy), size=num_streaks)
                brightness_vals = np.random.uniform(50, 130, size=num_streaks)

                for i in range(num_streaks):
                    cv2.line(
                        rain_layer, 
                        (rx[i], ry[i]), 
                        (rx[i] + dx, ry[i] + dy), 
                        float(brightness_vals[i]), 
                        1, 
                        lineType=cv2.LINE_AA
                    )

                # Motion blur along droplet fall vector
                motion_kernel = np.zeros((streak_len, streak_len), dtype=np.float32)
                for idx in range(streak_len):
                    col_idx = int(idx * (dx / max(1, dy)))
                    if col_idx < streak_len:
                        motion_kernel[idx, col_idx] = 1.0
                motion_kernel /= max(1.0, motion_kernel.sum())
                rain_blurred = cv2.filter2D(rain_layer, -1, motion_kernel)

                # Screen composite preserving image clarity
                rain_alpha = 0.14 + (self.severity * 0.08)
                for c in range(3):
                    res_np[:, :, c] = np.clip(res_np[:, :, c] + (rain_blurred * rain_alpha) + (ripple_layer * active_water * 0.20), 0, 255)

                result_clean = Image.fromarray(res_np.astype(np.uint8))
                hazard_mask = (submerged_ground | (depth_norm >= 0.70)).astype(np.uint8) * 255
                impact_pct = (submerged_ground > 0).mean() * 100.0

            # =========================================================================
            # BRANCH 2: BLIZZARD, FLOOD & COLLAPSE (CONTROLNET DIFFUSION INPAINTING)
            # =========================================================================
            else:
                # --- Step 2: Semantic Segmentation & Facade Protection ---
                self.log_signal.emit(f"Step 2/5: Detecting semantic elements for [{self.disaster_type}]...")
                targets = [t.strip() for t in self.target_prompt.split(",") if t.strip()]
                if not targets:
                    targets = ["ground", "sky", "roof"]

                combined_mask_pred = np.zeros((512, 512), dtype=np.float32)
                for t_query in targets:
                    inputs = ModelRegistry.clip_processor(
                        text=[t_query], 
                        images=[init_img], 
                        padding="max_length", 
                        return_tensors="pt"
                    )
                    with torch.no_grad():
                        outputs = ModelRegistry.clip_model(**inputs)
                    m_pred = torch.sigmoid(outputs.logits.unsqueeze(1))[0][0].cpu().numpy()
                    m_pred = cv2.resize(m_pred, (512, 512), interpolation=cv2.INTER_LINEAR)
                    combined_mask_pred = np.maximum(combined_mask_pred, m_pred)

                # Strict building facade isolation to prevent structural morphing
                inputs_bldg = ModelRegistry.clip_processor(
                    text=["house walls, glass windows, building facade, door frame"],
                    images=[init_img],
                    padding="max_length",
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs_bldg = ModelRegistry.clip_model(**inputs_bldg)
                bldg_facade_mask = torch.sigmoid(outputs_bldg.logits.unsqueeze(1))[0][0].cpu().numpy()
                bldg_facade_mask = cv2.resize(bldg_facade_mask, (512, 512), interpolation=cv2.INTER_LINEAR) > 0.22

                self.progress_signal.emit(45)

                # --- Step 3: Elevation & Surface Normal Aware Inpaint Masking ---
                self.log_signal.emit(f"Step 3/5: Computing mask geometry (Severity α = {self.severity:.2f})...")
                y_coords, _ = np.indices((512, 512))

                if "Blizzard" in self.disaster_type:
                  # 1. Ground, patio, and pool plane (everything at ground elevation)
                  ground_level = np.percentile(depth_norm, int(86 - self.severity * 30))
                  ground_zone = (depth_norm >= ground_level) | (y_coords > 300)

                  # 2. Roof surface: Target upper architectural horizontal surfaces
                  grad_y = np.abs(cv2.Sobel(depth_norm, cv2.CV_64F, 0, 1, ksize=3))
                  flat_horizontal = grad_y < np.percentile(grad_y, 70)
                  upper_structure = (y_coords < 330) & (depth_norm > 0.08) & (depth_norm < 0.65)
                  roof_zone = upper_structure & flat_horizontal

                  # 3. Foliage tops: Allow snow naturally on garden plants
                  foliage_zone = (combined_mask_pred > 0.22) & (y_coords > 180)

                  # Include the swimming pool so it freezes over with snow/ice instead of remaining turquoise
                  pool_zone = (hsv_init[:, :, 0] >= 80) & (hsv_init[:, :, 0] <= 125) & (y_coords > 320)

                  snow_base = ground_zone | roof_zone | foliage_zone | pool_zone
                  dilate_k = int(self.severity * 8) + 2
                  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
                  regulated_mask = cv2.dilate(snow_base.astype(np.uint8) * 255, kernel, iterations=1)
                
                  # Protect open sky and vertical facade glass
                  regulated_mask[depth_norm < 0.04] = 0
                  regulated_mask[bldg_facade_mask & (y_coords > 240)] = 0

                elif "Collapse" in self.disaster_type:
                    ground_mask = (depth_norm >= np.percentile(depth_norm, int(70 - self.severity * 30)))
                    collapse_base = ground_mask | (combined_mask_pred > 0.25)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    regulated_mask = cv2.dilate(collapse_base.astype(np.uint8) * 255, kernel, iterations=2)
                    regulated_mask[bldg_facade_mask] = 0

                else:  # Flash Flood
                    flood_base = (combined_mask_pred > 0.20).astype(np.uint8) * 255
                    if flood_base.sum() < (512 * 512 * 0.02 * 255):
                        flood_base = (depth_norm >= np.percentile(depth_norm, 80)).astype(np.uint8) * 255

                    iterations = int(self.severity * 24)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    dilated = cv2.dilate(flood_base, kernel, iterations=max(1, iterations))
                    active_depths = depth_norm[flood_base > 0]
                    base_elev = np.percentile(active_depths, 50) if len(active_depths) > 0 else 0.5
                    elev_ceiling = base_elev + (self.severity * 0.45)
                    growth_zone = (dilated > 0) & (depth_norm <= elev_ceiling)

                    regulated_mask = np.zeros_like(flood_base)
                    regulated_mask[growth_zone] = 255
                    regulated_mask[flood_base > 0] = 255
                    regulated_mask[bldg_facade_mask] = 0

                regulated_mask_img = Image.fromarray(regulated_mask)
                coverage = (regulated_mask > 0).mean() * 100.0
                regulated_mask_img.save("debug_current_mask.png")
                self.log_signal.emit(f"Active inpaint area: {coverage:.1f}% of frame (Severity: {self.severity:.2f})")
                self.progress_signal.emit(65)

                # --- Step 4: Geometry-Locked Diffusion Inference ---
                self.log_signal.emit(f"Step 4/5: Running diffusion on {DEVICE_NAME}...")

                if "Blizzard" in self.disaster_type:
                    if self.severity < 0.40:
                        effective_prompt = (
                            "photograph of modern villa with light fresh powdery snow dusting, "
                            "thin white snow layer on rooftop, light frost on patio and garden leaves, "
                            "cold winter day, sharp details, photorealistic, 8k"
                        )
                        cfg = 7.0
                        ctrl = 0.75
                    elif self.severity < 0.70:
                        effective_prompt = (
                            "photograph of modern villa during winter blizzard, "
                            "deep powdery white snowpack covering the house roof and patio, "
                            "snow resting naturally on garden plant leaves, winter snowdrifts, photorealistic, 8k"
                        )
                        cfg = 7.6
                        ctrl = 0.68
                    else:
                        effective_prompt = (
                            "award-winning photograph of modern villa buried under heavy winter blizzard, "
                            "thick heavy white snowdrifts completely covering the house roof and patio ground, "
                            "snow piled high on rooftop edges, garden bush dusted in snow, authentic winter disaster, 8k, sharp focus"
                        )
                        cfg = 8.2
                        ctrl = 0.62

                    negative_prompt = (
                      "blue water, turquoise water, swimming pool, water puddles, lake, "
                      "cartoon, illustration, 3d render, painting, blurry, smudged, chalk, "
                      "pine tree, fir tree, altered architecture, different house, extra buildings, sunny"
                  )

                elif "Collapse" in self.disaster_type:
                    effective_prompt = (
                        "documentary photograph of massive ground collapse, shattered broken concrete, "
                        "deep earth fissure, landslide mud rubble debris field, realistic disaster aftermath, 8k"
                    )
                    negative_prompt = "cartoon, illustration, painting, blurry, smooth ground, sunny"
                    cfg = 7.8
                    ctrl = 0.55

                else:  # Flash Flood
                    effective_prompt = (
                        "documentary photograph of turbulent muddy brown flash flood overflowing grounds, "
                        "churning raging floodwater, submerged patio, debris, overcast stormy lighting, hyperrealistic, 8k"
                    )
                    negative_prompt = "clean blue water, sunny, calm pool, cartoon, painting"
                    cfg = 7.8
                    ctrl = 0.52

                self.log_signal.emit(f"Running ControlNet Pipeline (CFG: {cfg}, CtrlScale: {ctrl})...")

                generator = torch.Generator(device="cpu").manual_seed(105)

                result_clean = ModelRegistry.pipe(
                    prompt=effective_prompt,
                    negative_prompt=negative_prompt,
                    image=init_img,
                    mask_image=regulated_mask_img,
                    control_image=depth_control_img,
                    num_inference_steps=24,
                    guidance_scale=cfg,
                    controlnet_conditioning_scale=ctrl,
                    generator=generator
                ).images[0]
                self.progress_signal.emit(85)

                # --- Step 5: Realistic Optical Atmospheric Simulation ---
                self.log_signal.emit("Step 5/5: Applying atmospheric post-processing...")
                res_np = np.array(result_clean).astype(np.float32)

                if "Blizzard" in self.disaster_type:
                    res_np[:, :, 0] = np.clip(res_np[:, :, 0] * 0.94 + (self.severity * 14), 0, 255)
                    res_np[:, :, 1] = np.clip(res_np[:, :, 1] * 0.96 + (self.severity * 16), 0, 255)
                    res_np[:, :, 2] = np.clip(res_np[:, :, 2] * 1.03 + (self.severity * 24), 0, 255)
                    result_clean = Image.fromarray(res_np.astype(np.uint8))

                hazard_mask = regulated_mask
                impact_pct = coverage

            # --- Step 6: Tactical Pathfinding & Intelligence Generation ---
            self.log_signal.emit("Step 6/6: Generating Evacuation Corridor & SITREP Report...")
            tactical_overlay_np, path_dist = EvacuationPathfinder.compute_evacuation_overlay(
                np.array(result_clean), hazard_mask, depth_norm
            )
            result_tactical = Image.fromarray(tactical_overlay_np)

            sitrep_text, voice_script, timestamp, risk_tier, est_transit, loc = TacticalReportGenerator.generate_report(
            self.disaster_type, self.severity, impact_pct, path_dist
            )

            meta = {
            "scenario": self.disaster_type,
            "severity": self.severity,
            "coverage_pct": impact_pct,
            "path_dist": path_dist,
            "timestamp": timestamp,
            "risk_tier": risk_tier,
            "transit_time": est_transit,
            "loc": loc
            }

            self.progress_signal.emit(100)
            self.log_signal.emit("Synthesis and tactical dispatch generation complete!")
            self.finished.emit(result_clean, result_tactical, sitrep_text, voice_script, meta, hazard_mask, depth_norm)

        except Exception as e:
            self.error_signal.emit(str(e))

class LocationWorker(QThread):
    finished = pyqtSignal(float, float, str)
    failed = pyqtSignal(str)

    def __init__(self, gis_manager):
        super().__init__()
        self.gis_manager = gis_manager

    def run(self):
        try:
            lat, lon, src = self.gis_manager.get_device_live_location()
            self.finished.emit(lat, lon, src)
        except Exception as e:
            self.failed.emit(str(e))
            
class ManualLocationWorker(QThread):
    finished = pyqtSignal(float, float, str)
    failed = pyqtSignal(str)

    def __init__(self, gis_manager, query_text):
        super().__init__()
        self.gis_manager = gis_manager
        self.query_text = query_text

    def run(self):
        import re
        query = self.query_text.strip()
        if not query:
            self.failed.emit("Query is empty")
            return

        coord_pattern = r"[-+]?\d*\.\d+|\d+"
        numbers = re.findall(coord_pattern, query)
        if len(numbers) >= 2:
            try:
                lat = float(numbers[0])
                lon = float(numbers[1])
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    self.gis_manager.current_lat = lat
                    self.gis_manager.current_lon = lon
                    self.finished.emit(lat, lon, f"COORDINATES [{lat:.4f}, {lon:.4f}]")
                    return
            except ValueError:
                pass

        res = self.gis_manager.geocode_address(query)
        if res:
            lat, lon, name = res
            self.finished.emit(lat, lon, f"LOCATION: {name.upper()}")
        else:
            self.failed.emit(f"Could not resolve location: '{query}'")
                      
# ---------------------------------------------------------------------------
# Native Desktop GUI Window (PyQt6)
# ---------------------------------------------------------------------------
class GeoGuardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_language = "English"
        self.selected_image_path = None
        self.result_clean = None
        self.result_tactical = None
        self.result_flir = None
        self.cached_hazard_mask = None
        self.cached_depth_norm = None
        self.report_meta = None
        self.voice_script = ""
        self.hud_active = False
        self.flir_active = False
        self.custom_start_pt = None
        self.custom_end_pt = None

        self.setWindowTitle("GeoGuard - Dynamic Disaster Severity Synthesis (IEEE)")
        self.resize(1280, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #111827; }
            QLabel { color: #E5E7EB; }
            QGroupBox { color: #9CA3AF; font-weight: bold; border: 1px solid #374151; border-radius: 8px; margin-top: 10px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QComboBox { background-color: #1F2937; color: #FFFFFF; border: 1px solid #4B5563; border-radius: 4px; padding: 6px; }
            QComboBox QAbstractItemView { 
                background-color: #1F2937; 
                color: #FFFFFF; 
                selection-background-color: #2563EB; 
                selection-color: #FFFFFF; 
            }
            QMessageBox { background-color: #1F2937; }
            QMessageBox QLabel { color: #F9FAFB; font-size: 13px; }
            QMessageBox QPushButton { background-color: #2563EB; color: white; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton { background-color: #2563EB; color: white; border-radius: 6px; padding: 10px; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #1D4ED8; }
            QPushButton:disabled { background-color: #4B5563; }
            QTextEdit { background-color: #030712; color: #10B981; font-family: Consolas, monospace; border: 1px solid #374151; border-radius: 4px; }
            QProgressBar { border: 1px solid #374151; border-radius: 4px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #2563EB; }
            QTabWidget::pane { border: 1px solid #374151; background: #111827; }
            QTabBar::tab { background: #1F2937; color: #9CA3AF; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; }
            QTabBar::tab:selected { background: #2563EB; color: white; }
        """)

        self.init_ui()
    
    def _on_map_viewport_clicked(self, title: str):
        """Catches Leaflet click events to synchronize C2 telemetry and log updates."""
        if not title.startswith("GEO_CLICK:"):
            return

        try:
            raw_coords = title.replace("GEO_CLICK:", "").strip()
            lat_str, lon_str = raw_coords.split(",")
            lat, lon = float(lat_str), float(lon_str)

            # Update GIS Manager state
            self.gis_manager.current_lat = lat
            self.gis_manager.current_lon = lon

            # Compute military grid reference
            mgrs = self.gis_manager.compute_mgrs_utm(lat, lon)
            hazard_radius = self.calculate_hazard_radius()

            # Update Header & Console
            status_text = f"TACTICAL C2 GIS // [OPERATOR PIN]: {lat:.5f}, {lon:.5f} | GRID: {mgrs}"
            if hasattr(self, 'lbl_gis_status'):
                self.lbl_gis_status.setText(status_text)
            self.append_log(f"> Tactical Target Re-pinned: ({lat:.5f}, {lon:.5f}) | Grid: {mgrs} | Perimeter: {hazard_radius}m")

        except Exception as e:
            print(f"[GIS Bridge] Viewport click parse error: {e}")
            
    def launch_3d_terrain(self):
        """Launches the 3D Digital Elevation Flyover Window."""
        if not hasattr(self, "cached_depth_norm") or self.cached_depth_norm is None:
            QMessageBox.warning(self, "Notice", "No depth model data available. Synthesize a scene first.")
            return

        active_img = self.result_tactical if getattr(self, "hud_active", True) else self.result_clean
        rgb_np = np.array(active_img.convert("RGB"))
        depth_np = self.cached_depth_norm

        dialog = Tactical3DTerrainDialog(rgb_np, depth_np, self)
        dialog.exec()
        
    def run_asset_optimization(self):
        """Computes regional triage metrics and populates the asset deployment table using Hungarian optimization."""
        from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox
        from asset_allocator import AssetAllocationOptimizer

        if not hasattr(self, "cached_hazard_mask") or self.cached_hazard_mask is None:
            QMessageBox.warning(self, "Notice", "Synthesize a disaster scenario first to generate triage zones.")
            return

        optimizer = AssetAllocationOptimizer()
        severity = getattr(self, "current_severity", 0.75)

        # 1. Extract triage targets from the computed hazard mask
        targets = optimizer.extract_triage_zones_from_mask(self.cached_hazard_mask, severity)

        # 2. Run Hungarian linear sum assignment
        dispatch_plan = optimizer.optimize_dispatch(targets)

        # 3. Populate self.tbl_assets directly
        if hasattr(self, 'tbl_assets') and self.tbl_assets is not None:
            self.tbl_assets.setRowCount(len(dispatch_plan))
            for row, item in enumerate(dispatch_plan):
                # Col 0: Asset Unit ID
                self.tbl_assets.setItem(row, 0, QTableWidgetItem(str(item["asset_id"])))
                # Col 1: Asset Type / Platform
                self.tbl_assets.setItem(row, 1, QTableWidgetItem(str(item["asset_type"])))
                # Col 2: Target Sector
                self.tbl_assets.setItem(row, 2, QTableWidgetItem(str(item["target_id"])))
                # Col 3: Mission Priority
                self.tbl_assets.setItem(row, 3, QTableWidgetItem(str(item["urgency"])))
                # Col 4: Assigned Operational Mission
                mission_desc = f"{item['asset_name']} -> {item['target_label']} (ETA: {item['eta_minutes']}m | Dist: {item['distance_km']}km)"
                self.tbl_assets.setItem(row, 4, QTableWidgetItem(mission_desc))
                # Col 5: Status
                self.tbl_assets.setItem(row, 5, QTableWidgetItem("DISPATCHED"))
    
    def on_zoom_slider_changed(self, value):
        factor = value / 100.0
        self.lbl_zoom_val.setText(f"{value}%")
        self.lbl_gen_viewport.set_zoom(factor)
            
    def on_toggle_split_curtain(self):
        """Swaps the synthesized viewport between standard tactical view and interactive split curtain."""
        if not hasattr(self, "result_clean") or self.result_clean is None:
            QMessageBox.warning(self, "Notice", "Synthesize a disaster scene first to enable the comparison curtain.")
            return

        if self.lbl_gen_viewport.split_active:
            self.lbl_gen_viewport.disable_split()
            self.lbl_orig_viewport.setVisible(True)  # Restore left viewport
            active_hud = getattr(self, "hud_active", True)
            active_img = self.result_tactical if (active_hud and hasattr(self, "result_tactical")) else self.result_clean
            self.display_pil_image(active_img)
            self.btn_split_curtain.setText("Enable Split-Screen Curtain")
            self.btn_split_curtain.setStyleSheet("background-color: #1E293B; color: #38BDF8; border: 1px solid #38BDF8; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
            self.append_log("Split-screen comparison curtain disabled.")
        else:
            from PIL import Image
            
            # 1. Grab target synthesized disaster image respecting route hide/show toggle
            active_hud = getattr(self, "hud_active", True)
            after_pil = self.result_tactical if (active_hud and hasattr(self, "result_tactical")) else self.result_clean

            # 2. Grab baseline image and MATCH its resolution & aspect ratio exactly to after_pil
            if hasattr(self, "selected_image_path") and os.path.exists(self.selected_image_path):
                orig_pil = Image.open(self.selected_image_path).convert("RGB")
                # Force exact pixel-to-pixel alignment with the synthesized canvas
                orig_pil = orig_pil.resize(after_pil.size, Image.Resampling.LANCZOS)
            else:
                orig_pil = self.result_clean

            pm_before = self.pil2pixmap(orig_pil)
            pm_after = self.pil2pixmap(after_pil)

            # Optional: Hide left original box so split curtain expands across the full display
            self.lbl_orig_viewport.setVisible(False)

            self.lbl_gen_viewport.set_split_images(pm_before, pm_after)
            self.btn_split_curtain.setText("Exit Split-Screen Mode")
            self.btn_split_curtain.setStyleSheet("background-color: #0284C7; color: #FFFFFF; border: 1px solid #38BDF8; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
            self.append_log("Split-screen comparison curtain activated. Drag the center cyan divider to swipe between states.")
    
    def on_viewport_hover(self, vx, vy):
        if not hasattr(self, "cached_depth_norm") or self.cached_depth_norm is None:
            return

        lbl_w = self.lbl_gen_viewport.width()
        lbl_h = self.lbl_gen_viewport.height()
        
        dh, dw = self.cached_depth_norm.shape
        if lbl_w <= 0 or lbl_h <= 0 or dh <= 0 or dw <= 0:
            return

        scale_x = dw / float(lbl_w)
        scale_y = dh / float(lbl_h)
        orig_x = int(vx * scale_x)
        orig_y = int(vy * scale_y)

        orig_x = max(0, min(dw - 1, orig_x))
        orig_y = max(0, min(dh - 1, orig_y))

        norm_depth = float(self.cached_depth_norm[orig_y, orig_x])
        elevation_m = (1.0 - norm_depth) * 15.0

        hazard_mask = getattr(self, "cached_hazard_mask", None)
        is_submerged = False
        hazard_label = "TERRAIN CLEAR"
        hazard_color = "#10B981"

        if hazard_mask is not None:
            local_window = hazard_mask[max(0, orig_y - 3):min(dh, orig_y + 4), max(0, orig_x - 3):min(dw, orig_x + 4)]
            if np.any(local_window > 0):
                is_submerged = True
                hazard_label = "INUNDATED / BLOCKED"
                hazard_color = "#EF4444"

        status_text = (
            f"<b style='color:#38BDF8;'>COORD:</b> [{orig_x:04d}, {orig_y:04d}] &nbsp;|&nbsp; "
            f"<b style='color:#38BDF8;'>REL ELEVATION:</b> {elevation_m:.2f}m &nbsp;|&nbsp; "
            f"<b style='color:#38BDF8;'>SURFACE:</b> <span style='color:{hazard_color}; font-weight:bold;'>{hazard_label}</span> &nbsp;|&nbsp; "
            f"<b style='color:#38BDF8;'>PASSABILITY:</b> "
            f"<span style='color:{hazard_color}; font-weight:bold;'>{'RESTRICTED / NO-GO' if is_submerged else 'CLEAR / TRANSIT OK'}</span>"
        )
        self.lbl_telemetry_hud.setText(status_text)

    def on_viewport_leave(self):
        """Resets HUD strip when cursor leaves the viewport."""
        self.lbl_telemetry_hud.setText("HUD TELEMETRY: STANDBY [AWAITING CURSOR FOCUS]")
        
    def update_triage_dashboard(self, hazard_mask):
        if getattr(self, "cached_depth_norm", None) is None or hazard_mask is None:
            return

        severity = self.report_meta.get("severity", 0.75) if getattr(self, "report_meta", None) else 0.75
        triage_data = StructuralTriageEngine.evaluate_zones(hazard_mask, self.cached_depth_norm, severity)

        self.table_triage.setRowCount(len(triage_data))
        for row, item in enumerate(triage_data):
            self.table_triage.setRowHeight(row, 38)

            # Col 0: Structural Zone
            item_zone = QTableWidgetItem(item["zone"])
            item_zone.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            item_zone.setForeground(QColor("#E5E7EB"))
            self.table_triage.setItem(row, 0, item_zone)

            # Col 1: Inundation %
            item_sub = QTableWidgetItem(item["submerged_pct"])
            item_sub.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            item_sub.setForeground(QColor("#9CA3AF"))
            self.table_triage.setItem(row, 1, item_sub)

            # Col 2: Water / Drift Depth
            item_depth = QTableWidgetItem(item["water_depth"])
            item_depth.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            item_depth.setForeground(QColor("#9CA3AF"))
            self.table_triage.setItem(row, 2, item_depth)

            # Col 3: Integrity Risk
            item_risk = QTableWidgetItem(item["risk_score"])
            item_risk.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            item_risk.setForeground(QColor("#F3F4F6"))
            self.table_triage.setItem(row, 3, item_risk)

            # Col 4: Passability Status (Color-Coded)
            pass_item = QTableWidgetItem(item["passable"])
            pass_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            pass_item.setForeground(QColor(item["color"]))
            self.table_triage.setItem(row, 4, pass_item)

            # Col 5: Triage Action Recommendation (Color-Coded)
            action_item = QTableWidgetItem(item["triage"])
            action_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            action_item.setForeground(QColor(item["color"]))
            self.table_triage.setItem(row, 5, action_item)

        # Ensure no manual resizeColumnsToContents overrides the QHeaderView section stretch modes
            
    def on_temporal_forecast_changed(self, minutes):
        if hasattr(self, "lbl_forecast"):
            self.lbl_forecast.setText(f"Temporal Forecast: T+{minutes:02d} Min (Physics Projection)")
            
        if not hasattr(self, "cached_hazard_mask") or self.cached_hazard_mask is None or getattr(self, "result_clean", None) is None:
            return

        # 1. Compute expanded environmental hazard mask
        forecasted_mask = TemporalHazardEngine.forecast_hazard(
            self.cached_hazard_mask,
            self.cached_depth_norm,
            minutes
        )

        clean_np = np.array(self.result_clean)
        visual_forecast = clean_np.copy()

        # 2. Crash-proof OpenCV water overlay blend
        if minutes > 0:
            # Water flood tint: BGR/RGB Blue-Aqua tint
            tint_layer = np.full_like(clean_np, (20, 90, 180), dtype=np.uint8)
            alpha = min(0.60, 0.20 + (minutes / 120.0) * 0.40)
            blended = cv2.addWeighted(clean_np, 1.0 - alpha, tint_layer, alpha, 0)
            
            mask_bool = (forecasted_mask > 0)
            visual_forecast[mask_bool] = blended[mask_bool]

        # 3. Recalculate dynamic egress route under encroaching conditions
        start_pt = getattr(self, "custom_start_pt", None)
        end_pt = getattr(self, "custom_end_pt", None)
        tactical_np, path_dist, _ = DynamicPathfinder.compute_egress_corridor(
            visual_forecast,
            forecasted_mask,
            self.cached_depth_norm,
            start_override=start_pt,
            end_override=end_pt
        )

        self.result_tactical = Image.fromarray(tactical_np)

        # 4. Update the active view immediately
        if getattr(self, "hud_active", False):
            self.display_pil_image(self.result_tactical)
        elif getattr(self, "flir_active", False):
            severity = self.report_meta.get("severity", 0.75) if getattr(self, "report_meta", None) else 0.75
            flir_np = ThermalReconEngine.generate_flir_overlay(
                clean_np,
                self.cached_depth_norm,
                forecasted_mask,
                severity
            )
            self.result_flir = Image.fromarray(flir_np)
            self.display_pil_image(self.result_flir)
        else:
            self.display_pil_image(Image.fromarray(visual_forecast))

        # Place it right here (aligned with the 8-space indentation):
        self.update_triage_dashboard(forecasted_mask)

    def on_language_changed(self, lang):
        self.current_language = lang
        self.append_log(f"Language updated to: {lang}")

        # If a simulation has completed, re-render the SITREP report in the selected language
        if getattr(self, "report_meta", None) and self.result_tactical:
            current_dist = self.report_meta.get("path_dist", 0.0)
            sitrep_text, voice_script, timestamp, risk_tier, est_transit, loc = TacticalReportGenerator.generate_report(
                self.report_meta["scenario"],
                self.report_meta["severity"],
                self.report_meta["coverage_pct"],
                current_dist,
                lang=self.current_language
            )
            self.voice_script = voice_script
            self.report_meta["path_dist"] = current_dist
            self.report_meta["transit_time"] = est_transit
            self.report_meta["risk_tier"] = risk_tier
            self.report_meta["loc"] = loc
            self.txt_sitrep.setPlainText(sitrep_text)
            
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Left Panel
        left_panel = QVBoxLayout()

        src_group = QGroupBox("1. Input Scene Image")
        # Language Selection Dropdown
        lang_box = QHBoxLayout()
        lang_box.addWidget(QLabel("Interface Language:"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["English", "Spanish", "Hindi", "French"])
        self.combo_lang.currentTextChanged.connect(self.on_language_changed)
        lang_box.addWidget(self.combo_lang)
        left_panel.addLayout(lang_box)
        src_layout = QVBoxLayout()
        self.btn_browse = QPushButton("Select Image from PC")
        self.btn_browse.clicked.connect(self.select_image)
        self.lbl_file_name = QLabel("No image selected.")
        self.lbl_file_name.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        src_layout.addWidget(self.btn_browse)
        src_layout.addWidget(self.lbl_file_name)
        src_group.setLayout(src_layout)
        left_panel.addWidget(src_group)
        
        # In __init__ or right after self.slider_severity is configured:
        self.slider_timer = QTimer(self)
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(40)  # 40ms debounce
        self.slider_timer.timeout.connect(self._sync_hazard_radius_to_map)
        
        cfg_group = QGroupBox("2. Disaster & Semantic Prompts")
        cfg_layout = QVBoxLayout()
        cfg_layout.addWidget(QLabel("Disaster Scenario:"))
        self.combo_disaster = QComboBox()
        self.combo_disaster.addItems([
            "Heavy Rainstorm (Dual-Branch)",
            "Blizzard / Deep Snow (Atmospheric)",
            "Flash Flood / Inundation",
            "Terrain / Road Collapse (Landslide)"
        ])
        self.combo_disaster.currentTextChanged.connect(self.update_preset_prompts)
        cfg_layout.addWidget(self.combo_disaster)

        cfg_layout.addWidget(QLabel("Segmentation Target:"))
        self.txt_target = QLineEdit("sky, ground, pavement, water")
        cfg_layout.addWidget(self.txt_target)

        cfg_layout.addWidget(QLabel("Synthesis Generation Prompt:"))
        self.txt_prompt = QLineEdit("dark rainclouds, wet reflective stone patio, dripping rainwater")
        cfg_layout.addWidget(self.txt_prompt)

        severity_box = QHBoxLayout()
        self.lbl_severity = QLabel("Alert Level: Warning (0.75)")
        severity_box.addWidget(self.lbl_severity)
        cfg_layout.addLayout(severity_box)

        self.slider_severity = QSlider(Qt.Orientation.Horizontal)
        self.slider_severity.setMinimum(15)
        self.slider_severity.setMaximum(95)
        self.slider_severity.setValue(75)
        self.slider_severity.valueChanged.connect(self.on_slider_changed)
        self.slider_severity.valueChanged.connect(self.on_severity_slider_changed)
        cfg_layout.addWidget(self.slider_severity)

        cfg_group.setLayout(cfg_layout)
        left_panel.addWidget(cfg_group)

        self.btn_run = QPushButton("Synthesize Disaster Scene")
        self.btn_run.clicked.connect(self.run_disaster_synthesis)
        left_panel.addWidget(self.btn_run)

        self.btn_dispatch = QPushButton("Broadcast Voice Dispatch")
        self.btn_dispatch.setStyleSheet("background-color: #DC2626;")
        self.btn_dispatch.setEnabled(False)
        self.btn_dispatch.clicked.connect(self.trigger_voice_dispatch)
        left_panel.addWidget(self.btn_dispatch)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_panel.addWidget(self.progress_bar)

        log_group = QGroupBox("Console & Runtime Status")
        log_layout = QVBoxLayout()
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        log_layout.addWidget(self.txt_logs)
        log_group.setLayout(log_layout)
        left_panel.addWidget(log_group)

        main_layout.addLayout(left_panel, stretch=2)

        # Right Panel
        right_panel = QVBoxLayout()
        self.tabs = QTabWidget()

        # Tab 1: Visual Assessment View
        view_tab = QWidget()
        view_layout = QGridLayout(view_tab)

        self.lbl_orig_viewport = QLabel("Source Scene Recon")
        self.lbl_orig_viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig_viewport.setStyleSheet("background-color: #030712; border: 1px dashed #374151; min-height: 380px; min-width: 380px;")

        # --- Live HUD Telemetry Strip ---
        self.lbl_telemetry_hud = QLabel("HUD TELEMETRY: STANDBY [HOVER OVER SYNTHESIZED SCENE FOR RECON]")
        self.lbl_telemetry_hud.setStyleSheet("""
            QLabel {
                background-color: #030712;
                border: 1px solid #1F2937;
                border-left: 3px solid #38BDF8;
                color: #94A3B8;
                font-family: Consolas, "Courier New", monospace;
                font-size: 11px;
                padding: 6px 12px;
                border-radius: 4px;
                margin-top: 4px;
            }
        """)
        self.lbl_gen_viewport = InteractiveTacticalLabel()
        self.lbl_gen_viewport.setText("Synthesized Disaster Outcome\n\n[After Synthesis: Left-Click to set Start | Right-Click to set Exit]")
        self.lbl_gen_viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gen_viewport.setStyleSheet("background-color: #030712; border: 1px dashed #374151; min-height: 380px; min-width: 380px;")
        # Viewport Interaction Connections
        self.lbl_gen_viewport.point_clicked.connect(self.on_viewport_point_clicked)
        self.lbl_gen_viewport.hover_moved.connect(self.on_viewport_hover)
        self.lbl_gen_viewport.hover_left.connect(self.on_viewport_leave)

        # Row 0: Labels
        view_layout.addWidget(QLabel("Input Geometry:"), 0, 0)
        view_layout.addWidget(QLabel("Synthesized Scene:"), 0, 1)

        # Row 1: Image Viewports (Give images the stretch priority: weight = 1)
        view_layout.addWidget(self.lbl_orig_viewport, 1, 0)
        view_layout.addWidget(self.lbl_gen_viewport, 1, 1)
        view_layout.setRowStretch(1, 1)

        # Row 2: Live HUD Telemetry Strip (Fixed height, no stretch: weight = 0)
        view_layout.addWidget(self.lbl_telemetry_hud, 2, 0, 1, 2)
        view_layout.setRowStretch(2, 0)

        # Row 3: Temporal Forecast Slider (Fixed height, no stretch: weight = 0)
        forecast_box = QHBoxLayout()
        self.lbl_forecast = QLabel("Temporal Forecast: T+00 Min (Real-Time Current)")
        self.lbl_forecast.setStyleSheet("color: #38BDF8; font-weight: bold; font-family: Consolas, monospace;")
        self.slider_forecast = QSlider(Qt.Orientation.Horizontal)
        self.slider_forecast.setRange(0, 120)
        self.slider_forecast.setValue(0)
        self.slider_forecast.setEnabled(False)
        self.slider_forecast.valueChanged.connect(self.on_temporal_forecast_changed)
        forecast_box.addWidget(self.lbl_forecast)
        forecast_box.addWidget(self.slider_forecast)
        view_layout.addLayout(forecast_box, 3, 0, 1, 2)
        view_layout.setRowStretch(3, 0)
        
        # Tactical Viewport Zoom Slider Line
        zoom_box = QHBoxLayout()
        lbl_zoom = QLabel("VIEWPORT ZOOM:")
        lbl_zoom.setStyleSheet("color: #94A3B8; font-weight: bold; font-family: Consolas, monospace; font-size: 11px;")
        
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(50, 300)  # 50% to 300%
        self.slider_zoom.setValue(100)      # 100% standard
        self.slider_zoom.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #1E293B; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #38BDF8; border-radius: 2px; }
            QSlider::handle:horizontal { background: #38BDF8; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; }
        """)
        
        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setStyleSheet("color: #38BDF8; font-weight: bold; font-family: Consolas, monospace; font-size: 11px; min-width: 40px;")
        
        self.btn_reset_zoom = QPushButton("1:1 Reset")
        self.btn_reset_zoom.setStyleSheet("background-color: #1E293B; color: #94A3B8; border: 1px solid #334155; font-size: 10px; padding: 2px 8px; border-radius: 3px;")
        self.btn_reset_zoom.clicked.connect(lambda: self.slider_zoom.setValue(100))

        self.slider_zoom.valueChanged.connect(self.on_zoom_slider_changed)
        self.lbl_gen_viewport.zoom_changed.connect(lambda factor: self.slider_zoom.setValue(int(factor * 100)))

        zoom_box.addWidget(lbl_zoom)
        zoom_box.addWidget(self.slider_zoom)
        zoom_box.addWidget(self.lbl_zoom_val)
        zoom_box.addWidget(self.btn_reset_zoom)

        # Add zoom_box to view_layout (Row 4) and push action buttons to Row 5
        view_layout.addLayout(zoom_box, 4, 0, 1, 2)
        view_layout.setRowStretch(4, 0)
        
        # Clean 3-Button Control Bar
        toggle_box = QHBoxLayout()

        # 1. Evacuation Vector Toggle
        self.btn_toggle_hud = QPushButton("Show Evacuation Route")
        self.btn_toggle_hud.setStyleSheet("background-color: #059669; color: white;")
        self.btn_toggle_hud.setEnabled(False)
        self.btn_toggle_hud.clicked.connect(self.toggle_hud_display)
        toggle_box.addWidget(self.btn_toggle_hud)
        
        # Split-Screen Curtain Slider Toggle
        self.btn_split_curtain = QPushButton("Enable Split-Screen Curtain")
        self.btn_split_curtain.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #38BDF8;
                border: 1px solid #38BDF8;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0284C7;
                color: #FFFFFF;
            }
        """)
        self.btn_split_curtain.clicked.connect(self.on_toggle_split_curtain)
        toggle_box.addWidget(self.btn_split_curtain)

        # 2. FLIR Thermal Toggle (Single button that toggles between FLIR and RGB)
        self.btn_toggle_flir = QPushButton("FLIR Thermal Recon")
        self.btn_toggle_flir.setStyleSheet("background-color: #D97706; color: white;")
        self.btn_toggle_flir.setEnabled(False)
        self.btn_toggle_flir.clicked.connect(self.toggle_flir_display)
        toggle_box.addWidget(self.btn_toggle_flir)

        # 3. Single Save Button
        self.btn_save = QPushButton("Save Synthesized Image to File")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_image)
        toggle_box.addWidget(self.btn_save)
        
        # 4. 3D LiDAR Digital Twin Flyover
        self.btn_3d_flyover = QPushButton("3D LiDAR Flyover")
        self.btn_3d_flyover.setEnabled(False)
        self.btn_3d_flyover.setStyleSheet("""
            QPushButton {
                background-color: #4338CA;
                color: #FFFFFF;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4F46E5; }
            QPushButton:disabled { background-color: #1E293B; color: #475569; }
        """)
        self.btn_3d_flyover.clicked.connect(self.launch_3d_terrain)
        toggle_box.addWidget(self.btn_3d_flyover)
        
        # Row 5: Attach assembled button bar to view_layout
        view_layout.addLayout(toggle_box, 5, 0, 1, 2)
        view_layout.setRowStretch(5, 0)
        
        self.tabs.addTab(view_tab, "Tactical Imagery & Evacuation HUD")

        # Tab 2: SITREP Report View & PDF Export
        sitrep_tab = QWidget()
        sitrep_layout = QVBoxLayout(sitrep_tab)
        self.txt_sitrep = QTextEdit()
        self.txt_sitrep.setReadOnly(True)
        self.txt_sitrep.setStyleSheet("color: #38BDF8; font-size: 13px; line-height: 1.4;")
        sitrep_layout.addWidget(self.txt_sitrep)

        self.btn_export_pdf = QPushButton("Export Full SITREP Dossier to PDF")
        self.btn_export_pdf.setStyleSheet("background-color: #4F46E5; color: white; font-size: 14px; padding: 12px;")
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_pdf.clicked.connect(self.export_pdf_report)
        sitrep_layout.addWidget(self.btn_export_pdf)

        self.tabs.addTab(sitrep_tab, "Tactical SITREP Report")

        right_panel.addWidget(self.tabs)
        main_layout.addLayout(right_panel, stretch=4)
        self.setCentralWidget(main_widget)
        self.append_log("System ready. Select an image to begin.")
        
       # --- TAB 3: Structural Triage Dashboard ---
        self.tab_triage = QWidget()
        triage_layout = QVBoxLayout()
        triage_layout.setContentsMargins(16, 16, 16, 16)
        triage_layout.setSpacing(12)

        self.lbl_triage_header = QLabel("AUTOMATED STRUCTURAL INTEGRITY & DISPATCH TRIAGE MATRIX")
        self.lbl_triage_header.setStyleSheet("""
            QLabel {
                font-family: Consolas, "Courier New", monospace;
                font-size: 13px;
                font-weight: 700;
                color: #38BDF8;
                letter-spacing: 0.5px;
            }
        """)
        triage_layout.addWidget(self.lbl_triage_header)

        self.table_triage = QTableWidget()
        self.table_triage.setColumnCount(6)
        self.table_triage.setHorizontalHeaderLabels([
            "Structural Zone",
            "Inundation %",
            "Water / Drift Depth",
            "Integrity Risk",
            "Passability",
            "Triage Action"
        ])

        # Configure Header Sizing: Explicit Content Proportions
        from PyQt6.QtWidgets import QHeaderView
        header = self.table_triage.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Full zone name without clipping
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Inundation %
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Water Depth
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Integrity Risk
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Passability
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)           # Triage Recommendation fills remaining width

        # UI Ergonomics: Hide row index counter, disable selection highlights, fix row height
        self.table_triage.verticalHeader().setVisible(False)
        self.table_triage.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table_triage.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table_triage.setShowGrid(True)

        # Enforce Complete Dark Tactical Styling on Viewport and Cells
        self.table_triage.setStyleSheet("""
            QTableWidget {
                background-color: #030712;
                border: 1px solid #1F2937;
                border-radius: 6px;
                gridline-color: #1F2937;
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
                color: #F3F4F6;
            }
            QTableWidget::viewport {
                background-color: #030712;
            }
            QTableWidget::item {
                padding: 10px 12px;
                border-bottom: 1px solid #111827;
                background-color: #030712;
            }
            QHeaderView::section {
                background-color: #111827;
                color: #9CA3AF;
                font-family: Consolas, monospace;
                font-weight: 700;
                font-size: 11px;
                padding: 10px 8px;
                border: none;
                border-right: 1px solid #1F2937;
                border-bottom: 2px solid #374151;
                text-transform: uppercase;
            }
        """)

        triage_layout.addWidget(self.table_triage)
        self.tab_triage.setLayout(triage_layout)
        self.tabs.addTab(self.tab_triage, "Structural Damage Triage")
        
        # --- TAB 4: Autonomous Multi-Agent Asset Allocation ---
        self.tab_assets = QWidget()
        self.init_asset_allocation_tab()
        self.tabs.addTab(self.tab_assets, "Autonomous Asset Allocation")
        
        # --- TAB 5: Dynamic GIS Georeferenced Tactical Map ---
        self.gis_manager = TacticalGISManager()
        # PHYS-DIFF Snow Analysis Engine
        self.snow_classifier = SentinelSnowClassifier()
        self.tab_gis = QWidget()
        self.init_gis_map_tab()
        self.tabs.addTab(self.tab_gis, "GIS Tactical Map")
    
    def init_asset_allocation_tab(self):
        """Builds the Multi-Agent Rescue Asset Dispatch Dashboard."""
        layout = QVBoxLayout(self.tab_assets)

        header_box = QHBoxLayout()
        lbl_title = QLabel("TACTICAL RESCUE ASSET DEPLOYMENT MATRIX")
        lbl_title.setStyleSheet("color: #38BDF8; font-weight: bold; font-size: 14px; font-family: Consolas, monospace;")
        
        self.btn_auto_dispatch = QPushButton("Auto-Optimize Asset Dispatch")
        self.btn_auto_dispatch.setStyleSheet("""
            QPushButton {
                background-color: #0284C7;
                color: white;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0369A1; }
        """)
        self.btn_auto_dispatch.clicked.connect(self.run_asset_optimization)

        header_box.addWidget(lbl_title)
        header_box.addStretch()
        header_box.addWidget(self.btn_auto_dispatch)
        layout.addLayout(header_box)

        # Asset Allocation Table
        self.tbl_assets = QTableWidget(0, 6)
        self.tbl_assets.setHorizontalHeaderLabels([
            "Asset Unit ID", "Asset Type / Platform", "Target Sector",
            "Mission Priority", "Assigned Operational Mission", "Status"
        ])
        self.tbl_assets.setStyleSheet("""
            QTableWidget {
                background-color: #0B1329;
                color: #E2E8F0;
                gridline-color: #1E293B;
                border: 1px solid #334155;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1E293B;
                color: #38BDF8;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #334155;
            }
        """)
        from PyQt6.QtWidgets import QHeaderView
        self.tbl_assets.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tbl_assets)
        self.tbl_assets.horizontalHeader().setStretchLastSection(False)
        self.tbl_assets.resizeColumnsToContents()
        
    def init_gis_map_tab(self):
        """Constructs the embedded Leaflet GIS tactical viewport with manual coordinate/address override."""
        gis_layout = QVBoxLayout(self.tab_gis)
        gis_layout.setContentsMargins(6, 6, 6, 6)

        # 1. Top Telemetry Status Line
        self.lbl_gis_status = QLabel("TACTICAL C2 GIS // STANDBY")
        self.lbl_gis_status.setStyleSheet("color: #00D4FF; font-weight: bold; font-size: 12px; font-family: monospace;")
        gis_layout.addWidget(self.lbl_gis_status)

        # 2. Operator Location Override Control Bar
        ctrl_bar = QHBoxLayout()
        
        lbl_search = QLabel("Target Location / Lat,Lon:")
        lbl_search.setStyleSheet("color: #94A3B8; font-weight: bold; font-size: 11px;")
        ctrl_bar.addWidget(lbl_search)

        self.txt_manual_location = QLineEdit()
        self.txt_manual_location.setPlaceholderText("Enter address (e.g. 'BKC Mumbai') or '19.0760, 72.8777'")
        self.txt_manual_location.setStyleSheet("""
            QLineEdit { background-color: #0F172A; color: #F8FAFC; border: 1px solid #334155; 
                        padding: 5px 8px; border-radius: 3px; font-family: monospace; }
        """)
        self.txt_manual_location.returnPressed.connect(self.apply_manual_location)
        ctrl_bar.addWidget(self.txt_manual_location, stretch=2)

        btn_apply = QPushButton("Deploy Grid")
        btn_apply.setStyleSheet("""
            QPushButton { background-color: #2563EB; color: #FFFFFF; font-weight: bold; padding: 5px 12px; border-radius: 3px; }
            QPushButton:hover { background-color: #1D4ED8; }
        """)
        btn_apply.clicked.connect(self.apply_manual_location)
        ctrl_bar.addWidget(btn_apply)

        btn_recenter = QPushButton("Center Ground Zero")
        btn_recenter.setStyleSheet("""
            QPushButton { background-color: #1E293B; color: #E2E8F0; padding: 5px 10px; border-radius: 3px; border: 1px solid #475569; }
            QPushButton:hover { background-color: #334155; }
        """)
        btn_recenter.clicked.connect(self.update_tactical_map)
        ctrl_bar.addWidget(btn_recenter)

        gis_layout.addLayout(ctrl_bar)

        # 3. WebEngine Leaflet Map Viewport
        self.web_map_view = QWebEngineView()
        self.web_map_view.page().featurePermissionRequested.connect(self._handle_geolocation_permission)
        self.web_map_view.setStyleSheet("background: #070B14; border: 1px solid #1E293B;")
        self.web_map_view.titleChanged.connect(self._on_map_viewport_clicked)
        gis_layout.addWidget(self.web_map_view)
        
        # --- Inside init_gis_map_tab(), right after btn_apply ---
        self.btn_device_locate = QPushButton("🚨 Emergency: Locate Device")
        self.btn_device_locate.setStyleSheet("""
            QPushButton { 
                background-color: #DC2626; 
                color: #FFFFFF; 
                font-weight: bold; 
                padding: 5px 12px; 
                border-radius: 3px; 
                border: 1px solid #EF4444; 
            }
            QPushButton:hover { background-color: #B91C1C; }
        """)
        self.btn_device_locate.clicked.connect(self.trigger_emergency_device_locate)
        ctrl_bar.addWidget(self.btn_device_locate)
        
        # Initial render
        self.update_tactical_map()
    
    def _handle_geolocation_permission(self, security_origin, feature):
        from PyQt6.QtWebEngineCore import QWebEnginePage
        from PyQt6.QtWidgets import QMessageBox

        if feature == QWebEnginePage.Feature.Geolocation:
            reply = QMessageBox.question(
                self,
                "Sensor Authorization Request",
                f"Tactical C2 Node requests authorization to access host device geolocation sensors.\nOrigin: {security_origin.toString()}\n\nGrant access?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.web_map_view.page().setFeaturePermission(
                    security_origin, feature, QWebEnginePage.PermissionPolicy.PermissionGrantedByUser
                )
                self.append_log("> Host device location sensor authorized by operator.")
            else:
                self.web_map_view.page().setFeaturePermission(
                    security_origin, feature, QWebEnginePage.PermissionPolicy.PermissionDeniedByUser
                )
                self.append_log("[WARN] Geolocation permission denied by operator.")
                
    def calculate_hazard_radius(self) -> int:
        """Computes dynamic tactical exclusion zone in meters based on active alert level & scenario."""
        # Read from self.slider_severity
        alpha = self.slider_severity.value() / 100.0 if hasattr(self, 'slider_severity') else 0.75
        scenario = self.combo_disaster.currentText() if hasattr(self, 'combo_disaster') else "Heavy Rainstorm"

        scenario_profiles = {
            "Heavy Rainstorm (Dual-Branch)": (150, 600),
            "Blizzard / Deep Snow (Atmospheric)": (200, 800),
            "Flash Flood / Inundation": (200, 750),
            "Terrain / Road Collapse (Landslide)": (80, 450),
            "Wildfire Flare": (250, 1200),
            "Industrial Hazmat / Toxic Spill": (300, 1500),
            "Earthquake / Structural Debris": (80, 400)
        }

        base_min, base_max = scenario_profiles.get(scenario, (150, 500))
        calculated_radius = int(base_min + (alpha * (base_max - base_min)))
        return calculated_radius
    
    def on_severity_slider_changed(self, value):
        """Updates UI label instantly and debounces the Leaflet map update."""
        alpha = value / 100.0
        label_text = "Advisory" if alpha < 0.4 else "Warning" if alpha < 0.8 else "Critical"
        if hasattr(self, 'lbl_alert_val'):
            self.lbl_alert_val.setText(f"Alert Level: {label_text} ({alpha:.2f})")
        if hasattr(self, 'lbl_severity'):
            self.lbl_severity.setText(f"Alert Level: {label_text} ({alpha:.2f})")

        # Restart debounce timer (prevents flooding the WebEngine process)
        if hasattr(self, 'slider_timer'):
            self.slider_timer.start()

    def _sync_hazard_radius_to_map(self):
        """Safely updates the Leaflet canvas, red hazard circle, and green staging LZ in real time."""
        if not hasattr(self, 'web_map_view') or not self.web_map_view:
            return

        hazard_radius = self.calculate_hazard_radius()

        # Read current center coordinates from GIS manager
        lat = getattr(self.gis_manager, 'current_lat', 19.0760)
        lon = getattr(self.gis_manager, 'current_lon', 72.8777)

        js_update = f"""
            if (typeof hazardCircle !== 'undefined' && hazardCircle) {{
                hazardCircle.setRadius({hazard_radius});
            }}
            var hudElem = document.getElementById('hud-perimeter');
            if (hudElem) {{
                hudElem.innerText = 'PERIMETER: {hazard_radius}m RADIUS';
            }}
            if (typeof window.updateTacticalLZ === 'function') {{
                window.updateTacticalLZ({lat}, {lon}, {hazard_radius});
            }}
        """
        try:
            self.web_map_view.page().runJavaScript(js_update)
        except Exception as e:
            print(f"[GIS] Realtime sync error: {e}")
            
    def update_tactical_map(self):
        """Extracts coordinates and updates GIS without tearing down the browser engine."""
        if not hasattr(self, 'gis_manager') or not hasattr(self, 'web_map_view'):
            return

        image_path = getattr(self, 'selected_image_path', "")
        telemetry = self.gis_manager.extract_drone_coordinates(image_path)

        lat = telemetry["lat"]
        lon = telemetry["lon"]
        mgrs = self.gis_manager.compute_mgrs_utm(lat, lon)
        hazard_radius = self.calculate_hazard_radius()

        source = telemetry.get("source", "SYNTHETIC_DEFAULT")
        prefix = "DRONE EXIF GPS" if source == "EXIF_HARDWARE_GPS" else "SYNTHETIC BASELINE"
        self.lbl_gis_status.setText(f"TACTICAL C2 GIS // [{prefix}]: {lat:.5f}, {lon:.5f} | GRID: {mgrs}")

        # Check if the map has already been loaded once
        if not getattr(self, '_map_initialized', False):
            # First load only: initialize full HTML
            html_content = self.gis_manager.generate_tactical_leaflet_html(lat, lon, hazard_radius_m=hazard_radius)
            self.web_map_view.setHtml(html_content)
            self._map_initialized = True
        else:
            # Subsequent updates: instant JavaScript repositioning (zero freeze)
            js_reposition = f"""
                if (typeof map !== 'undefined') {{
                    map.panTo([{lat}, {lon}]);
                    if (typeof incidentMarker !== 'undefined') {{
                        incidentMarker.setLatLng([{lat}, {lon}]);
                        incidentMarker.bindPopup("<b>GROUND ZERO INCIDENT SITE</b><br/>Grid: {mgrs}").openPopup();
                    }}
                    if (typeof hazardCircle !== 'undefined') {{
                        hazardCircle.setLatLng([{lat}, {lon}]);
                        hazardCircle.setRadius({hazard_radius});
                    }}
                }}
            """
            self.web_map_view.page().runJavaScript(js_reposition)

    def apply_manual_location(self):
        """Dispatches manual address/coordinate resolution without blocking the UI thread."""
        query = self.txt_manual_location.text().strip()
        if not query:
            return

        self.append_log(f"> [MANUAL GRID] Resolving target: '{query}'...")
        if hasattr(self, 'lbl_gis_status'):
            self.lbl_gis_status.setText(f"TACTICAL C2 GIS // [RESOLVING: {query.upper()}...]")

        # Launch resolution in background thread
        self.manual_loc_worker = ManualLocationWorker(self.gis_manager, query)
        self.manual_loc_worker.finished.connect(self._on_manual_location_resolved)
        self.manual_loc_worker.failed.connect(lambda err: self.append_log(f"[WARN] {err}"))
        self.manual_loc_worker.start()

    def _on_manual_location_resolved(self, lat: float, lon: float, tag: str):
        """Updates the Leaflet map and tactical elements smoothly once resolved."""
        mgrs = self.gis_manager.compute_mgrs_utm(lat, lon)
        hazard_radius = self.calculate_hazard_radius()

        status_msg = f"TACTICAL C2 GIS // [{tag}]: {lat:.5f}, {lon:.5f} | GRID: {mgrs}"
        if hasattr(self, 'lbl_gis_status'):
            self.lbl_gis_status.setText(status_msg)
        self.append_log(f"> Target Grid Deployed: {tag} ({lat:.5f}, {lon:.5f}) | Grid: {mgrs}")

        # Update map view smoothly via JavaScript
        js_update = f"""
            if (typeof map !== 'undefined') {{
                map.flyTo([{lat}, {lon}], 16, {{ animate: true, duration: 1.0 }});
                if (typeof incidentMarker !== 'undefined') {{
                    incidentMarker.setLatLng([{lat}, {lon}]);
                    incidentMarker.bindPopup("<b>MANUAL TARGET GROUND ZERO</b><br/>{tag}<br/>Lat: {lat:.5f}, Lon: {lon:.5f}<br/>Grid: {mgrs}").openPopup();
                }}
                if (typeof hazardCircle !== 'undefined') {{
                    hazardCircle.setLatLng([{lat}, {lon}]);
                    hazardCircle.setRadius({hazard_radius});
                }}
                if (typeof window.updateTacticalLZ === 'function') {{
                    window.updateTacticalLZ({lat}, {lon}, {hazard_radius});
                }}
            }}
        """
        if hasattr(self, 'web_map_view') and self.web_map_view:
            self.web_map_view.page().runJavaScript(js_update)
    
    def trigger_emergency_device_locate(self):
        """Dispatches location lookup on a non-blocking background thread."""
        self.append_log("> [EMERGENCY PROTOCOL] Polling host device positioning in background...")
        if hasattr(self, 'lbl_gis_status'):
            self.lbl_gis_status.setText("TACTICAL C2 GIS // [SEARCHING SENSORS...]")

        # Start asynchronous worker
        self.loc_worker = LocationWorker(self.gis_manager)
        self.loc_worker.finished.connect(self._on_emergency_location_resolved)
        self.loc_worker.failed.connect(lambda err: self.append_log(f"[WARN] Sensor timeout: {err}"))
        self.loc_worker.start()

    def _on_emergency_location_resolved(self, lat: float, lon: float, source_type: str):
        """Called automatically when coordinates are locked without blocking the UI."""
        mgrs = self.gis_manager.compute_mgrs_utm(lat, lon)
        hazard_radius = self.calculate_hazard_radius()

        status_msg = f"TACTICAL C2 GIS // [EMERGENCY: {source_type}]: {lat:.5f}, {lon:.5f} | GRID: {mgrs}"
        if hasattr(self, 'lbl_gis_status'):
            self.lbl_gis_status.setText(status_msg)
        self.append_log(f"> Beacon Lock: [{source_type}] at ({lat:.5f}, {lon:.5f}) | Grid: {mgrs}")

        # Inside _on_emergency_location_resolved:
        js_fly = f"""
            if (typeof map !== 'undefined') {{
                map.flyTo([{lat}, {lon}], 16, {{ animate: true, duration: 1.0 }});
                if (typeof incidentMarker !== 'undefined') {{
                    incidentMarker.setLatLng([{lat}, {lon}]);
                }}
                if (typeof hazardCircle !== 'undefined') {{
                    hazardCircle.setLatLng([{lat}, {lon}]);
                    hazardCircle.setRadius({hazard_radius});
                }}
                if (typeof window.updateTacticalLZ === 'function') {{
                    window.updateTacticalLZ({lat}, {lon}, {hazard_radius});
                }}
            }}
        """
        if hasattr(self, 'web_map_view') and self.web_map_view:
            self.web_map_view.page().runJavaScript(js_fly)
                     
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.selected_image_path = file_name
            if hasattr(self, 'update_tactical_map'):
                self.update_tactical_map()
            self.lbl_file_name.setText(os.path.basename(file_name))
            pixmap = QPixmap(file_name).scaled(380, 380, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_orig_viewport.setPixmap(pixmap)
            self.append_log(f"Loaded image: {os.path.basename(file_name)}")
            self.update_preset_prompts(self.combo_disaster.currentText())
    
    def on_slider_changed(self, value):
        val_float = value / 100.0
        if val_float < 0.40:
            level = "Level 1: Advisory"
        elif val_float < 0.70:
            level = "Level 2: Watch"
        else:
            level = "Level 3: Critical Warning"
        self.lbl_severity.setText(f"{level} [{val_float:.2f}]")

    def update_preset_prompts(self, scenario):
        if scenario == "Heavy Rainstorm (Dual-Branch)":
            self.txt_target.setText("sky, ground, pavement, water")
            self.txt_prompt.setText("dark overcast rainclouds, wet reflective stone patio, dripping rainwater")
        elif scenario == "Blizzard / Deep Snow (Atmospheric)":
            self.txt_target.setText("roof, ground, tiles, plant, bush")
            self.txt_prompt.setText("crisp powdery white snow on rooftop and patio, natural snow on foliage")
        elif scenario == "Flash Flood / Inundation":
            self.txt_target.setText("water, river, road, ground, valley")
            self.txt_prompt.setText("turbulent brown muddy floodwaters, overflowing banks, floating debris")
        elif scenario == "Terrain / Road Collapse (Landslide)":
            self.txt_target.setText("road, ground, pavement, slope")
            self.txt_prompt.setText("massive ground collapse, shattered cracked earth, landslide fissure")

    def run_disaster_synthesis(self):
        if not self.selected_image_path:
            self.append_log("[ERROR] Please select an image first.")
            return
        
        self.slider_forecast.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_dispatch.setEnabled(False)
        self.btn_toggle_hud.setEnabled(False)
        self.btn_toggle_flir.setEnabled(False)
        self.btn_export_pdf.setEnabled(False)
        self.progress_bar.setValue(0)

        # Reset custom waypoints
        self.custom_start_pt = None
        self.custom_end_pt = None

        severity_val = self.slider_severity.value() / 100.0

        self.worker = DisasterWorker(
            image_path=self.selected_image_path,
            disaster_type=self.combo_disaster.currentText(),
            target_prompt=self.txt_target.text(),
            user_prompt=self.txt_prompt.text(),
            severity=severity_val
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_synthesis_complete)
        self.worker.error_signal.connect(self.on_synthesis_error)
        self.worker.start()

    def on_synthesis_complete(self, clean_pil, tactical_pil, report_text, voice_script, meta, hazard_mask, depth_norm):
        self.result_clean = clean_pil
        self.result_tactical = tactical_pil
        self.voice_script = voice_script
        self.cached_hazard_mask = hazard_mask
        self.cached_depth_norm = depth_norm
        self.txt_sitrep.setPlainText(report_text)

        self.report_meta = meta if meta is not None else {}
        self.report_meta["lang"] = getattr(self, "current_language", "English")

        clean_np = np.array(self.result_clean)
        flir_np = ThermalReconEngine.generate_flir_overlay(
            clean_np,
            self.cached_depth_norm,
            self.cached_hazard_mask,
            self.report_meta.get("severity", 0.75)
        )
        self.result_flir = Image.fromarray(flir_np)
        self.flir_active = False
        
        if hasattr(self, "btn_export_pdf"):
            self.btn_export_pdf.setEnabled(True)
            
        if hasattr(self, "btn_3d_flyover"):
            self.btn_3d_flyover.setEnabled(True)
        
        if hasattr(self, "btn_toggle_flir"):
            self.btn_toggle_flir.setEnabled(True)
            self.btn_toggle_flir.setText("FLIR Thermal Recon")
            self.btn_toggle_flir.setStyleSheet("background-color: #D97706; color: white;")

        self.hud_active = False
        self.display_pil_image(self.result_clean)
        self.btn_toggle_hud.setText("Show Evacuation Route")
        self.btn_toggle_hud.setStyleSheet("background-color: #059669; color: white;")
        
        self.slider_forecast.setEnabled(True)
        self.slider_forecast.setValue(0)
        self.lbl_forecast.setText("Temporal Forecast: T+00 Min (Real-Time Current)")
        self.btn_run.setEnabled(True)
        self.btn_browse.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_3d_flyover.setEnabled(True)
        self.btn_dispatch.setEnabled(True)
        self.btn_toggle_hud.setEnabled(True)
        self.btn_export_pdf.setEnabled(True)
        self.append_log("Ready! Click viewport to adjust Start (Left Click) or Rescue LZ (Right Click).")

        # Place it right here at the very end of on_synthesis_complete:
        self.update_triage_dashboard(self.cached_hazard_mask)
        
    def on_viewport_point_clicked(self, x, y, point_type):
        if getattr(self.lbl_gen_viewport, "split_active", False):
            return
        if self.result_clean is None or self.cached_hazard_mask is None:
            return

        if point_type == "START":
            self.custom_start_pt = (y, x)
            self.append_log(f"Set Evacuee Start Point at ({x}, {y})")
        elif point_type == "END":
            self.custom_end_pt = (y, x)
            self.append_log(f"Set Extraction LZ-1 at ({x}, {y})")

        # Fast live A* path recalculation across cached hazard cost matrix
        clean_np = np.array(self.result_clean)
        tactical_np, path_dist = EvacuationPathfinder.compute_evacuation_overlay(
            clean_np,
            self.cached_hazard_mask,
            self.cached_depth_norm,
            start_pt=self.custom_start_pt,
            end_pt=self.custom_end_pt
        )
        self.result_tactical = Image.fromarray(tactical_np)

        # Update SITREP and dispatch telemetry live
        report_output = TacticalReportGenerator.generate_report(
            self.report_meta["scenario"],
            self.report_meta["severity"],
            self.report_meta["coverage_pct"],
            path_dist,
            lang=self.current_language
        )
        sitrep_text = report_output[0]
        voice_script = report_output[1]
        timestamp = report_output[2]
        risk_tier = report_output[3]
        est_transit = report_output[4]
        self.voice_script = voice_script
        self.report_meta["path_dist"] = path_dist
        self.report_meta["transit_time"] = est_transit
        self.txt_sitrep.setPlainText(sitrep_text)

        # Display route overlay immediately
        self.hud_active = True
        self.display_pil_image(self.result_tactical)
        self.btn_toggle_hud.setText("Hide Evacuation Route (Show Clean View)")
        self.btn_toggle_hud.setStyleSheet("background-color: #475569; color: white;")
        self.append_log(f"Live Egress Updated: ~{path_dist:.1f}m (Transit: ~{est_transit:.1f} min)")
    
    def pil2pixmap(self, pil_img):
        """Converts a PIL Image to a QPixmap via an in-memory buffer."""
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        pix = QPixmap()
        pix.loadFromData(buf.getvalue())
        return pix

    def display_pil_image(self, pil_img):
        """Passes the raw unscaled pixmap directly to the viewport."""
        if pil_img is None:
            return
        pix = self.pil2pixmap(pil_img)
        self.lbl_gen_viewport.set_image(pix)

    def toggle_hud_display(self):
        """Toggles between showing the A* evacuation vector and the clean hazard view."""
        if not hasattr(self, "result_clean") or self.result_clean is None:
            return

        self.hud_active = not getattr(self, "hud_active", True)

        # Select which image to show based on toggle state
        target_pil = self.result_tactical if self.hud_active else self.result_clean

        if self.lbl_gen_viewport.split_active:
            # If split curtain is active, update the right-side disaster pixmap live
            pm_after = self.pil2pixmap(target_pil)
            self.lbl_gen_viewport.update_split_after(pm_after)
        else:
            # Standard single viewport
            self.display_pil_image(target_pil)

        # Update button text & style
        if self.hud_active:
            self.btn_toggle_hud.setText("Hide Evacuation Route (Show Clean View)")
            self.btn_toggle_hud.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 5px 8px; border-radius: 4px;")
            self.append_log("Evacuation corridor vectors displayed.")
        else:
            self.btn_toggle_hud.setText("Show Evacuation Route")
            self.btn_toggle_hud.setStyleSheet("background-color: #1E293B; color: #E2E8F0; font-weight: bold; padding: 5px 8px; border-radius: 4px;")
            self.append_log("Evacuation corridor hidden. Displaying clean synthesized scene.")
            
    def toggle_flir_display(self):
        if not getattr(self, "result_flir", None):
            return

        self.flir_active = not getattr(self, "flir_active", False)
        if self.flir_active:
            self.hud_active = False
            self.btn_toggle_hud.setText("Show Evacuation Route")
            self.btn_toggle_hud.setStyleSheet("background-color: #059669; color: white;")

            self.display_pil_image(self.result_flir)
            self.btn_toggle_flir.setText("Exit FLIR (Standard RGB)")
            self.btn_toggle_flir.setStyleSheet("background-color: #475569; color: white;")
            self.append_log("Multispectral FLIR thermal sensor active.")
        else:
            self.display_pil_image(self.result_clean)
            self.btn_toggle_flir.setText("FLIR Thermal Recon")
            self.btn_toggle_flir.setStyleSheet("background-color: #D97706; color: white;")
            self.append_log("Restored standard optical imagery.")
            
    def trigger_voice_dispatch(self):
        """Emits an edge-synthesized military radio dispatch with squelch bursts and formal telemetry."""
        from tactical_audio_generator import TacticalAudioDispatcher

        if not hasattr(self, 'audio_dispatcher'):
            self.audio_dispatcher = TacticalAudioDispatcher()

        scenario = self.combo_disaster.currentText() if hasattr(self, 'combo_disaster') else "Severe Inundation"
        severity = self.slider_severity.value() / 100.0 if hasattr(self, 'slider_severity') else 0.75
        selected_lang = self.combo_lang.currentText() if hasattr(self, 'combo_lang') else "English"

        # Check if assets have been computed by the optimizer
        active_assets = []
        if hasattr(self, 'asset_optimizer') and hasattr(self, 'cached_hazard_mask') and self.cached_hazard_mask is not None:
            targets = self.asset_optimizer.extract_triage_zones_from_mask(self.cached_hazard_mask, severity)
            active_assets = self.asset_optimizer.optimize_dispatch(targets)

        # Build standardized incident command broadcast
        script = self.audio_dispatcher.format_nato_dispatch(
            scenario=scenario,
            severity=severity,
            sector="BRAVO",
            lang=selected_lang,
            dispatched_assets=active_assets
        )

        self.append_log(f"> Transmitting formal tactical emergency broadcast [{selected_lang}]...")
        self.audio_dispatcher.dispatch_vocal_alert(script, lang=selected_lang)

    def export_pdf_report(self):
        """Compiles mission telemetry, imagery, and structural triage matrix into both an interactive HTML web report and a standalone PDF."""
        if not hasattr(self, "result_clean") or self.result_clean is None:
            QMessageBox.warning(self, "Export Error", "No synthesized disaster scene available to export.")
            return

        base_default_name = f"GeoGuard_SITREP_Dossier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Incident Mission Report",
            f"{base_default_name}.html",
            "Interactive Web Page (*.html);;PDF Document (*.pdf)"
        )
        if not save_path:
            return

        try:
            import base64
            import io
            import webbrowser

            # Convert images to embedded base64 data URIs so the HTML works standalone anywhere
            def pil_to_base64(img):
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

            if hasattr(self, "selected_image_path") and os.path.exists(self.selected_image_path):
                from PIL import Image
                orig_b64 = pil_to_base64(Image.open(self.selected_image_path))
            else:
                orig_b64 = pil_to_base64(self.result_clean)

            active_tactical = getattr(self, "result_tactical", self.result_clean)
            tactical_b64 = pil_to_base64(active_tactical)

            # Build structural triage metrics rows
            severity = self.report_meta.get("severity", 0.75) if getattr(self, "report_meta", None) else 0.75
            triage_rows = StructuralTriageEngine.evaluate_zones(self.cached_hazard_mask, self.cached_depth_norm, severity)

            triage_table_html = ""
            for item in triage_rows:
                triage_table_html += f"""
                <tr>
                    <td style="padding: 10px; font-weight: bold; border-bottom: 1px solid #1e293b;">{item['zone']}</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #1e293b;">{item['submerged_pct']}</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #1e293b;">{item['water_depth']}</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #1e293b;">{item['risk_score']}</td>
                    <td style="padding: 10px; text-align: center; color: {item['color']}; font-weight: bold; border-bottom: 1px solid #1e293b;">{item['passable']}</td>
                    <td style="padding: 10px; color: {item['color']}; font-weight: bold; border-bottom: 1px solid #1e293b;">{item['triage']}</td>
                </tr>
                """

            scenario = self.combo_disaster.currentText()
            sitrep_body = self.txt_sitrep.toPlainText().replace("\n", "<br>")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

            # Complete self-contained Tactical Web/Print HTML Dossier
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoGuard Mission SITREP - {scenario}</title>
    <style>
        body {{
            background-color: #0b0f17;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 32px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        .header {{
            border-bottom: 2px solid #0284c7;
            padding-bottom: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .title {{
            font-size: 24px;
            font-weight: 800;
            letter-spacing: 0.5px;
            color: #38bdf8;
            text-transform: uppercase;
            margin: 0;
        }}
        .subtitle {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
            font-family: monospace;
        }}
        .meta-bar {{
            background-color: #1e293b;
            border-left: 4px solid #38bdf8;
            padding: 12px 16px;
            margin-bottom: 24px;
            font-size: 13px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            font-family: monospace;
        }}
        .meta-item b {{ color: #38bdf8; }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 28px;
        }}
        .card {{
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 6px;
            overflow: hidden;
        }}
        .card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .card-caption {{
            padding: 10px;
            font-size: 11px;
            font-weight: 700;
            color: #94a3b8;
            text-align: center;
            font-family: monospace;
            background: #0b0f17;
            border-top: 1px solid #1e293b;
        }}
        .section-title {{
            font-size: 14px;
            font-weight: 700;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            border-left: 3px solid #38bdf8;
            padding-left: 8px;
            font-family: monospace;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            font-family: monospace;
            margin-bottom: 28px;
            background: #0f172a;
            border-radius: 6px;
            overflow: hidden;
        }}
        th {{
            background-color: #1e293b;
            color: #94a3b8;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            border-bottom: 2px solid #334155;
        }}
        .sitrep-box {{
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid #10b981;
            padding: 18px;
            font-size: 13px;
            font-family: Consolas, monospace;
            line-height: 1.6;
            color: #f1f5f9;
        }}
        @media print {{
            body {{ background: white; color: black; padding: 0; }}
            .container {{ border: none; box-shadow: none; padding: 10px; background: white; color: black; }}
            .title {{ color: #0284c7; }}
            .meta-bar {{ background: #f1f5f9; color: black; }}
            .card {{ border: 1px solid #ccc; }}
            .sitrep-box {{ background: #f8fafc; color: black; border-color: #cbd5e1; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="title">GeoGuard Tactical Incident Dossier</div>
                <div class="subtitle">AI SYNTHESIS • HAZARD DYNAMICS • EMERGENCY DISPATCH TELEMETRY</div>
            </div>
            <div style="font-family: monospace; font-size: 11px; color: #64748b;">IEEE AI RESCUE PROTOCOL</div>
        </div>

        <div class="meta-bar">
            <div class="meta-item"><b>SCENARIO:</b> {scenario.upper()}</div>
            <div class="meta-item"><b>TIMESTAMP:</b> {timestamp}</div>
            <div class="meta-item"><b>SEVERITY:</b> {severity:.2f}</div>
            <div class="meta-item"><b>STATUS:</b> ACTIVE DISPATCH</div>
        </div>

        <div class="section-title">Visual Hazard Comparison & Corridor</div>
        <div class="grid">
            <div class="card">
                <img src="{orig_b64}" alt="Optical Geometry">
                <div class="card-caption">BASELINE PRE-DISASTER OPTICAL SCENE</div>
            </div>
            <div class="card">
                <img src="{tactical_b64}" alt="Tactical HUD">
                <div class="card-caption">HAZARD SYNTHESIS & A* EVACUATION CORRIDOR</div>
            </div>
        </div>

        <div class="section-title">Structural Compromise & Triage Matrix</div>
        <table>
            <thead>
                <tr>
                    <th style="text-align: left;">Structural Zone</th>
                    <th>Inundation %</th>
                    <th>Water / Drift Depth</th>
                    <th>Integrity Risk</th>
                    <th>Passability</th>
                    <th style="text-align: left;">Triage Directive</th>
                </tr>
            </thead>
            <tbody>
                {triage_table_html}
            </tbody>
        </table>

        <div class="section-title">Emergency Responder SITREP & Field Briefing</div>
        <div class="sitrep-box">
            {sitrep_body}
        </div>
    </div>
</body>
</html>
"""

            # Save as HTML
            if save_path.lower().endswith(".html"):
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                self.append_log(f"Mission web dossier saved: {os.path.basename(save_path)}")
                webbrowser.open(f"file://{os.path.abspath(save_path)}")
                QMessageBox.information(self, "Export Successful", f"Web Dossier generated and opened in browser:\n{save_path}")

            # Save as PDF
            else:
                doc = QTextDocument()
                doc.setHtml(html_content)
                printer = QPrinter(QPrinter.PrinterMode.HighResolution)
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(save_path)
                printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                doc.print(printer)

                self.append_log(f"Mission PDF dossier saved: {os.path.basename(save_path)}")
                webbrowser.open(f"file://{os.path.abspath(save_path)}")
                QMessageBox.information(self, "Export Successful", f"PDF Dossier generated and opened in viewer:\n{save_path}")

        except Exception as e:
            self.append_log(f"Export Error: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export dossier:\n{str(e)}")

    def on_synthesis_error(self, err_msg):
        self.append_log(f"[FATAL ERROR]: {err_msg}")
        self.btn_run.setEnabled(True)
        self.btn_browse.setEnabled(True)

    def save_image(self):
        target = self.result_tactical if (self.hud_active and self.result_tactical) else self.result_clean
        if target:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Synthesized Image", "disaster_output.png", "PNG (*.png);;JPEG (*.jpg)")
            if save_path:
                target.save(save_path)
                self.append_log(f"Saved result to: {save_path}")

    def append_log(self, text):
        self.txt_logs.append(f"> {text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GeoGuardApp()
    window.show()
    sys.exit(app.exec())