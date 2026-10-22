"""
GeoGuard 3D Elevation Mesh & Tactical LiDAR Digital Twin
Smooth photogrammetric point-splatting with anti-aliased depth reconstruction.
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox
)
from PyQt6.QtCore import Qt
from vispy import scene
from vispy.scene import visuals


class Tactical3DTerrainDialog(QDialog):
    """3D GPU-accelerated tactical elevation & LiDAR flyover dialog."""

    def __init__(self, rgb_image_np: np.ndarray, depth_map_norm: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GeoGuard Tactical — 3D LiDAR Elevation Flyover")
        self.resize(1100, 750)
        self.setStyleSheet("background-color: #030712; color: #E2E8F0;")

        layout = QVBoxLayout(self)

        # Control Header
        header = QHBoxLayout()
        title = QLabel("TACTICAL 3D RECON (ORBIT: LEFT-DRAG | PAN: RIGHT-DRAG | ZOOM: WHEEL)")
        title.setStyleSheet("color: #38BDF8; font-weight: bold; font-family: Consolas, monospace; font-size: 11px;")
        
        lbl_render = QLabel("RENDER STYLE:")
        lbl_render.setStyleSheet("color: #94A3B8; font-size: 10px; font-weight: bold; font-family: Consolas, monospace;")
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Dense LiDAR Point Cloud", "Continuous Surface Mesh", "Thermal Elevation Heatmap"])
        self.combo_mode.setStyleSheet("""
            QComboBox { background-color: #1E293B; color: #E2E8F0; border: 1px solid #334155; padding: 3px 8px; border-radius: 4px; font-size: 11px; }
            QComboBox QAbstractItemView { background-color: #0B1329; color: #E2E8F0; selection-background-color: #0284C7; }
        """)
        self.combo_mode.currentIndexChanged.connect(self.rebuild_visual)

        lbl_scale = QLabel("DEPTH EXAGGERATION:")
        lbl_scale.setStyleSheet("color: #94A3B8; font-size: 10px; font-weight: bold; font-family: Consolas, monospace;")

        self.slider_elev = QSlider(Qt.Orientation.Horizontal)
        self.slider_elev.setRange(1, 30)
        self.slider_elev.setValue(6)
        self.slider_elev.setFixedWidth(100)
        self.slider_elev.valueChanged.connect(self.rebuild_visual)

        btn_reset = QPushButton("Reset Camera")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #38BDF8;
                border: 1px solid #38BDF8;
                font-weight: bold;
                padding: 4px 12px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0284C7; color: white; }
        """)
        btn_reset.clicked.connect(self.reset_camera)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(lbl_render)
        header.addWidget(self.combo_mode)
        header.addWidget(lbl_scale)
        header.addWidget(self.slider_elev)
        header.addWidget(btn_reset)
        layout.addLayout(header)

        # 3D Vispy Scene Canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='#030712', parent=self)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(elevation=25, azimuth=-45, distance=2.2)
        layout.addWidget(self.canvas.native)

        # Cache inputs
        self.rgb_raw = rgb_image_np
        self.depth_raw = depth_map_norm
        self.active_node = None

        self.rebuild_visual()

    def reset_camera(self):
        self.view.camera.elevation = 25
        self.view.camera.azimuth = -45
        self.view.camera.distance = 2.2

    def rebuild_visual(self):
        """Constructs an anti-aliased 3D point cloud without staircasing artifacts."""
        if self.active_node is not None:
            self.active_node.parent = None
            self.active_node = None

        mode = self.combo_mode.currentIndex()
        h, w = self.depth_raw.shape

        stride = 1
        d_sub = self.depth_raw[::stride, ::stride].astype(np.float32)
        rgb_sub = self.rgb_raw[::stride, ::stride]
        rows, cols = d_sub.shape

        # 1. Multi-pass smoothing to eradicate stepped contour lines
        d_smooth = cv2.GaussianBlur(d_sub, (5, 5), 1.2)
        d_smooth = cv2.bilateralFilter(d_smooth, d=7, sigmaColor=0.08, sigmaSpace=4)

        # 2. Linear metric depth calculation
        d_norm = (d_smooth - np.min(d_smooth)) / (np.max(d_smooth) - np.min(d_smooth) + 1e-6)
        
        # Controlled relief curve (prevents wild perspective separation)
        depth_mult = (self.slider_elev.value() / 100.0) * 0.45
        z_curve = (1.0 - d_norm) * depth_mult

        # 3. Spatial pinhole grid
        aspect = cols / float(rows)
        x = np.linspace(-1.0 * aspect, 1.0 * aspect, cols, dtype=np.float32)
        y = np.linspace(1.0, -1.0, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        zz = z_curve

        # 4. Color normalization
        if mode == 2:
            z_scale = (zz - np.min(zz)) / (np.max(zz) - np.min(zz) + 1e-6)
            heat = cv2.applyColorMap((z_scale.flatten() * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            colors = (cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0).astype(np.float32)
        else:
            colors = (rgb_sub.reshape(-1, 3) / 255.0).astype(np.float32)

        rgba = np.hstack([colors, np.ones((colors.shape[0], 1), dtype=np.float32)])

        if mode == 1:
            # Mode 1: Continuous Surface Mesh
            vertices = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten())).astype(np.float32)
            faces = []
            max_drop = 0.035

            for r in range(0, rows - 1, 2):
                for c in range(0, cols - 1, 2):
                    i0 = r * cols + c
                    i1 = i0 + 1
                    i2 = (r + 1) * cols + c
                    i3 = i2 + 1

                    z0, z1, z2, z3 = zz.flat[i0], zz.flat[i1], zz.flat[i2], zz.flat[i3]

                    if max(abs(z0 - z1), abs(z1 - z2), abs(z0 - z2)) < max_drop:
                        faces.append([i0, i1, i2])
                    if max(abs(z1 - z3), abs(z3 - z2), abs(z1 - z2)) < max_drop:
                        faces.append([i1, i3, i2])

            faces = np.array(faces, dtype=np.uint32)

            self.active_node = visuals.Mesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=rgba,
                shading='smooth'
            )
        else:
            # Mode 0 & 2: Dense Point Splats (Overlapping continuous point surface)
            points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten())).astype(np.float32)
            self.active_node = visuals.Markers()
            self.active_node.set_data(
                pos=points,
                face_color=rgba,
                edge_color=None,
                size=4.0
            )

        self.view.add(self.active_node)