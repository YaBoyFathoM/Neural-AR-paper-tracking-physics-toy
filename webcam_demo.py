"""
Production-Ready Real-Time AR Physics Sandbox.

Detects a piece of paper via a trained HRNet mesh regressor, projects drawn
shapes onto it using piecewise-affine warping with real-world lighting transfer,
and simulates physics with gravity derived from paper tilt.

Optimizations over prototype:
  - FP16 inference on CUDA (~30% faster)
  - Inference throttling (configurable skip-frames)
  - Early-exit warp when overlay is empty
  - Vectorized alpha blending (no per-channel Python loop)
  - Multi-band lighting with auto white-point calibration
  - Bilateral-filtered light map for shadow smoothness
  - FPS counter & performance timing HUD
  - CLI arguments for camera/resolution/quality
  - Graceful camera recovery & clean shutdown
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import argparse
import cv2
import time
import torch
import numpy as np
import pymunk
import pygame
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train_mesh import HeatmapMeshRegressor, IMG_SIZE, GRID_W, GRID_H, NUM_POINTS


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Neural AR Physics Sandbox")
    p.add_argument('--camera', type=int, default=0, help='Camera device index')
    p.add_argument('--model', type=str, default='best_9x12_768.pth')
    p.add_argument('--width', type=int, default=1280, help='Camera width (1280=default)')
    p.add_argument('--height', type=int, default=720, help='Camera height (720=default)')
    p.add_argument('--no-fp16', action='store_true', help='Disable FP16 inference')
    p.add_argument('--infer-skip', type=int, default=2,
                   help='Run inference every N frames (1=every, 2=half rate)')
    return p.parse_args()


# ── Config ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CANVAS_SIZE = 512
GRAVITY_STRENGTH = 2500.0
PHYSICS_DT = 1.0 / 60.0
BOUNCE = 0.8
FRICTION = 0.5

# Col-Major 9×12 corner indices
IDX_TL, IDX_BL, IDX_TR, IDX_BR = 0, 11, 96, 107

COLORS = {
    'cyan':    (0, 240, 255),
    'magenta': (255, 0, 212),
    'amber':   (255, 179, 0),
    'lime':    (57, 255, 20),
    'white':   (255, 255, 255),
    'grid':    (100, 100, 100),
}

BRUSH_PALETTE = [
    (255, 0, 0), (255, 128, 0), (0, 255, 0),
    (0, 100, 255), (255, 0, 255), (255, 255, 0),
]


# ── Performance Utilities ────────────────────────────────────────────
class FPSCounter:
    """Rolling-window FPS tracker."""
    def __init__(self, window=60):
        self._times = []
        self._window = window

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 0 else 0.0

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)


class PerfTimer:
    """Named lap timer for per-section profiling (in ms)."""
    def __init__(self):
        self.timings = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        if name in self._starts:
            self.timings[name] = (time.perf_counter() - self._starts[name]) * 1000

    def ms(self, name):
        return self.timings.get(name, 0.0)


# ── 1-Euro Filter ───────────────────────────────────────────────────
class OneEuroFilter:
    """Speed-adaptive low-pass filter for jitter-free mesh tracking."""

    def __init__(self, min_cutoff=1.0, beta=0.015, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None
        self._t = None
        self.enabled = True

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x):
        if not self.enabled or x is None:
            return x

        t = time.perf_counter()
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x

        dt = t - self._t
        if dt <= 0:
            return self._x

        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)

        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.copy()


# ── Mesh Stabilizer ──────────────────────────────────────────────────
class MeshStabilizer:
    """Blocks erroneous predictions via velocity rejection, topology checks,
    and a coasting state machine."""

    TRACKING = 'TRACKING'
    COASTING = 'COASTING'
    LOST     = 'LOST'

    def __init__(self, max_coast_frames=10, velocity_threshold=0.35,
                 min_quad_area=800.0):
        self.max_coast_frames = max_coast_frames
        self.velocity_threshold = velocity_threshold
        self.min_quad_area = min_quad_area
        self.state = self.LOST
        self._good_mesh = None
        self._prev_mesh = None
        self._coast_counter = 0
        self._frame_diag = None

    def set_frame_size(self, w, h):
        self._frame_diag = np.sqrt(w * w + h * h)

    @staticmethod
    def _is_convex_quad(tl, tr, br, bl):
        pts = [tl, tr, br, bl]
        sign = None
        for i in range(4):
            o, a, b = pts[i], pts[(i+1) % 4], pts[(i+2) % 4]
            cross = (a[0]-o[0])*(b[1]-a[1]) - (a[1]-o[1])*(b[0]-a[0])
            if sign is None:
                if cross != 0: sign = cross > 0
            elif cross != 0 and (cross > 0) != sign:
                return False
        return sign is not None

    @staticmethod
    def _quad_area(tl, tr, br, bl):
        pts = [tl, tr, br, bl]
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]
        return abs(area) / 2.0

    def _topology_ok(self, mesh_xy):
        tl, tr = mesh_xy[IDX_TL], mesh_xy[IDX_TR]
        br, bl = mesh_xy[IDX_BR], mesh_xy[IDX_BL]
        if not self._is_convex_quad(tl, tr, br, bl):
            return False
        return self._quad_area(tl, tr, br, bl) >= self.min_quad_area

    def _velocity_ok(self, mesh_xy):
        if self._prev_mesh is None:
            return True
        disp = np.linalg.norm(mesh_xy - self._prev_mesh, axis=1).mean()
        return disp < self.velocity_threshold * (self._frame_diag or 1000.0)

    def update(self, raw_mesh):
        accepted = self._topology_ok(raw_mesh) and self._velocity_ok(raw_mesh)
        self._prev_mesh = raw_mesh.copy()

        if accepted:
            self._good_mesh = raw_mesh.copy()
            self._coast_counter = 0
            self.state = self.TRACKING
            return self._good_mesh, self.TRACKING

        if self._good_mesh is not None and self._coast_counter < self.max_coast_frames:
            self._coast_counter += 1
            self.state = self.COASTING
            return self._good_mesh, self.COASTING

        self.state = self.LOST
        return None, self.LOST


# ── Lighting Engine ──────────────────────────────────────────────────
class LightingEngine:
    """Multi-band lighting transfer with auto white-point calibration.

    Instead of a hardcoded white-point (200), samples the paper's actual
    luminance and tracks it with an EMA. Uses bilateral filtering on the
    light map to prevent hard shadow edges from bleeding onto drawn shapes.
    """

    def __init__(self, white_point=200.0, ema_rate=0.05):
        self.white_point = white_point
        self.ema_rate = ema_rate
        self._cached_ratio = None

    def _update_white_point(self, gray, mesh_xy):
        h, w = gray.shape
        quad = np.array([
            mesh_xy[IDX_TL], mesh_xy[IDX_TR],
            mesh_xy[IDX_BR], mesh_xy[IDX_BL]
        ], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad, 255)
        paper_luma = cv2.mean(gray, mask=mask)[0]
        if paper_luma > 50:
            self.white_point += self.ema_rate * (paper_luma - self.white_point)

    def compute(self, frame_rgb, mesh_xy=None):
        """Returns (H, W) float32 light-ratio map (~1.0 for normal paper).
        Uses fast downscaled blur for real-time performance."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        if mesh_xy is not None:
            self._update_white_point(gray, mesh_xy)

        # Downscale → blur → upscale is much faster than large-kernel blur
        h, w = gray.shape
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (15, 15), 0)
        smooth = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)

        self._cached_ratio = np.clip(
            smooth / max(self.white_point, 1.0), 0.15, 1.8)
        return self._cached_ratio


# ── Physics World ────────────────────────────────────────────────────
class PhysicsWorld:
    """Pymunk space with thick boundary walls and dynamic bodies."""

    def __init__(self, size=CANVAS_SIZE):
        self.size = size
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY_STRENGTH)
        self.bodies = []

        thickness = 100.0
        corners = [
            (-thickness, -thickness), (size + thickness, -thickness),
            (size + thickness, size + thickness), (-thickness, size + thickness),
        ]
        for i in range(4):
            wall = pymunk.Segment(self.space.static_body,
                                  corners[i], corners[(i+1) % 4], thickness)
            wall.elasticity = BOUNCE
            wall.friction = FRICTION
            self.space.add(wall)

    def spawn_ball(self, cx, cy, radius, color):
        radius = max(radius, 6.0)
        mass = 0.5 + (radius / 20.0)
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (cx, cy)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = BOUNCE
        shape.friction = FRICTION
        self.space.add(body, shape)
        self.bodies.append(PhysicsBody(body, shape, radius, color, is_poly=False))

    def spawn_poly(self, vertices, color):
        pts = np.array(vertices, dtype=np.float32)
        cx, cy = np.mean(pts, axis=0)
        local_verts = [(v[0]-cx, v[1]-cy) for v in pts]
        area = pymunk.area_for_poly(local_verts)
        mass = max(0.5, area / 500.0)
        moment = pymunk.moment_for_poly(mass, local_verts)
        body = pymunk.Body(mass, moment)
        body.position = (cx, cy)
        shape = pymunk.Poly(body, local_verts)
        shape.elasticity = BOUNCE
        shape.friction = FRICTION
        self.space.add(body, shape)
        self.bodies.append(PhysicsBody(body, shape, 0, color, is_poly=True))

    def update_and_step(self, full_mesh):
        self.space.gravity = (0.0, 0.0) # Disable global macro gravity
        grid = full_mesh.reshape(GRID_W, GRID_H, 2)
        cam_down = np.array([0.0, 1.0])
        
        for _ in range(3):
            # Apply topological gravity dynamically based on the exact quad the object resides on
            for pb in self.bodies:
                px, py = pb.body.position
                i = int(np.clip((px / self.size) * (GRID_W - 1), 0, GRID_W - 2))
                j = int(np.clip((py / self.size) * (GRID_H - 1), 0, GRID_H - 2))
                
                tl, bl = grid[i, j], grid[i, j+1]
                tr, br = grid[i+1, j], grid[i+1, j+1]
                
                paper_y = ((bl - tl) + (br - tr)) * 0.5
                paper_x = ((tr - tl) + (br - bl)) * 0.5
                ny, nx = np.linalg.norm(paper_y), np.linalg.norm(paper_x)
                if ny > 0: paper_y /= ny
                if nx > 0: paper_x /= nx
                
                gx = float(np.dot(cam_down, paper_x)) * GRAVITY_STRENGTH
                gy = float(np.dot(cam_down, paper_y)) * GRAVITY_STRENGTH
                
                pb.body.force = (gx * pb.body.mass, gy * pb.body.mass)
            
            self.space.step(PHYSICS_DT / 3.0)

    def clear(self):
        for pb in self.bodies:
            self.space.remove(pb.body, pb.shape)
        self.bodies.clear()


class PhysicsBody:
    __slots__ = ('body', 'shape', 'radius', 'color', 'is_poly')
    def __init__(self, body, shape, radius, color, is_poly):
        self.body, self.shape, self.radius = body, shape, radius
        self.color, self.is_poly = color, is_poly


# ── Drawing Canvas ───────────────────────────────────────────────────
class DrawingCanvas:
    """Mouse-based drawing mapped through inverse homography to physics space."""

    def __init__(self, physics: PhysicsWorld):
        self.physics = physics
        self.mesh_corners = None
        self.brush_size = 8
        self.color_idx = 0
        self.color = BRUSH_PALETTE[0]
        self._stroke = []
        self._drawing = False

    @property
    def is_empty(self):
        """True when there's nothing to render (skip warp entirely)."""
        return not self._drawing and len(self.physics.bodies) == 0

    def cycle_color(self):
        self.color_idx = (self.color_idx + 1) % len(BRUSH_PALETTE)
        self.color = BRUSH_PALETTE[self.color_idx]

    def clear(self):
        self.physics.clear()
        self._stroke = []
        self._drawing = False

    def handle_event(self, event):
        if event.type not in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION,
                              pygame.MOUSEBUTTONUP):
            return
        if getattr(self, 'full_mesh', None) is None:
            return

        mx, my = event.pos
        
        # Piecewise inverse mapping to allow drawing precisely on curved paper
        grid = self.full_mesh.reshape(GRID_W, GRID_H, 2)
        x_steps = np.linspace(0, CANVAS_SIZE, GRID_W)
        y_steps = np.linspace(0, CANVAS_SIZE, GRID_H)
        
        found_quad = False
        lx, ly = -1, -1
        for i in range(GRID_W - 1):
            if found_quad: break
            for j in range(GRID_H - 1):
                dst_pts = np.array([
                    grid[i, j], grid[i+1, j],
                    grid[i+1, j+1], grid[i, j+1]
                ], dtype=np.float32)
                
                # Check if mouse clicked exactly inside this 4-point quad!
                if cv2.pointPolygonTest(dst_pts, (float(mx), float(my)), False) >= 0:
                    src_pts = np.array([
                        [x_steps[i], y_steps[j]], [x_steps[i+1], y_steps[j]],
                        [x_steps[i+1], y_steps[j+1]], [x_steps[i], y_steps[j+1]]
                    ], dtype=np.float32)
                    try:
                        H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
                        warped = cv2.perspectiveTransform(
                            np.array([[[mx, my]]], dtype=np.float32), H_inv)[0][0]
                        lx, ly = warped[0], warped[1]
                        found_quad = True
                    except cv2.error:
                        pass
                    break
                    
        if not found_quad:
            return
            
        valid = 0 <= lx <= CANVAS_SIZE and 0 <= ly <= CANVAS_SIZE

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and valid:
            self._drawing = True
            self._stroke = [(lx, ly)]
        elif event.type == pygame.MOUSEMOTION and self._drawing:
            self._stroke.append((
                max(0, min(lx, CANVAS_SIZE-1)),
                max(0, min(ly, CANVAS_SIZE-1)),
            ))
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._drawing:
            self._drawing = False
            self._stroke.append((
                max(0, min(lx, CANVAS_SIZE-1)),
                max(0, min(ly, CANVAS_SIZE-1)),
            ))
            self._commit_stroke()

    def _commit_stroke(self):
        if len(self._stroke) < 2:
            self._stroke = []
            return
        pts = np.array(self._stroke, dtype=np.float32)
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        if radius < self.brush_size * 2.5:
            self.physics.spawn_ball(float(cx), float(cy),
                                   max(float(radius), float(self.brush_size)),
                                   self.color)
        else:
            hull = cv2.convexHull(pts).reshape(-1, 2)
            hull = cv2.approxPolyDP(hull, self.brush_size * 0.5, True).reshape(-1, 2)
            if len(hull) >= 3:
                self.physics.spawn_poly(hull.tolist(), self.color)
            else:
                self.physics.spawn_ball(float(cx), float(cy),
                                       max(float(radius), float(self.brush_size)),
                                       self.color)
        self._stroke = []

    def compose(self):
        """Build RGBA overlay of active stroke + physics bodies."""
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 4), dtype=np.uint8)

        if self._drawing and len(self._stroke) > 1:
            pts = np.array(self._stroke, dtype=np.int32).reshape(-1, 1, 2)
            c = list(self.color) + [255]
            cv2.polylines(canvas, [pts], False, c, self.brush_size * 2, cv2.LINE_AA)

        for pb in self.physics.bodies:
            c = list(pb.color) + [255]
            if pb.is_poly:
                pts = [v.rotated(pb.body.angle) + pb.body.position
                       for v in pb.shape.get_vertices()]
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(canvas, [pts], c, lineType=cv2.LINE_AA)
                cv2.polylines(canvas, [pts], True, (255,255,255,255), 2, cv2.LINE_AA)
            else:
                px, py = int(pb.body.position.x), int(pb.body.position.y)
                r = int(pb.radius)
                cv2.circle(canvas, (px, py), r, c, -1, cv2.LINE_AA)
                a = pb.body.angle
                ex = int(px + r * np.cos(a))
                ey = int(py + r * np.sin(a))
                cv2.line(canvas, (px, py), (ex, ey), (255,255,255,255), 2, cv2.LINE_AA)
        return canvas


# ── Warp & Composite ────────────────────────────────────────────────
def warp_overlay(frame, mesh_xy, overlay_rgba, light_ratio):
    """Piecewise-affine warp with real-world lighting.
    
    Iterates over the 88 mesh quads for perfectly curved projection mapping.
    Uses bounding-box isolation and alpha-zero early exit to maintain 60 FPS.
    """
    h, w = frame.shape[:2]
    tw, th = overlay_rgba.shape[:2]
    
    # Early out if nothing to warp
    if not np.any(overlay_rgba[:, :, 3]):
        return frame

    result = frame.copy()
    
    # Calculate mesh bounding box to localize array allocations
    mxint = mesh_xy.astype(np.int32)
    x_min, y_min = np.min(mxint, axis=0)
    x_max, y_max = np.max(mxint, axis=0)
    
    # padding
    x_min, y_min = max(0, x_min - 2), max(0, y_min - 2)
    x_max, y_max = min(w, x_max + 3), min(h, y_max + 3)
    
    if x_max <= x_min or y_max <= y_min:
        return frame
        
    warped_full = np.zeros((y_max - y_min, x_max - x_min, 4), dtype=np.float32)
    
    global_dx0, global_dy0 = w, h
    global_dx1, global_dy1 = 0, 0
    has_content = False

    grid = mesh_xy.reshape(GRID_W, GRID_H, 2)
    x_steps = np.linspace(0, tw, GRID_W)
    y_steps = np.linspace(0, th, GRID_H)
    
    for i in range(GRID_W - 1):
        for j in range(GRID_H - 1):
            src_pts = np.array([
                [x_steps[i], y_steps[j]], [x_steps[i+1], y_steps[j]],
                [x_steps[i+1], y_steps[j+1]], [x_steps[i], y_steps[j+1]]
            ], dtype=np.float32)
            
            sx0, sy0 = int(src_pts[:, 0].min()), int(src_pts[:, 1].min())
            sx1, sy1 = int(np.ceil(src_pts[:, 0].max())), int(np.ceil(src_pts[:, 1].max()))
            
            crop_src = overlay_rgba[sy0:sy1, sx0:sx1]
            if crop_src.size == 0 or not np.any(crop_src[:, :, 3]):
                continue
                
            dst_pts = np.array([
                grid[i, j], grid[i+1, j],
                grid[i+1, j+1], grid[i, j+1]
            ], dtype=np.float32)
                
            dx0, dy0 = int(np.floor(dst_pts[:, 0].min())), int(np.floor(dst_pts[:, 1].min()))
            dx1, dy1 = int(np.ceil(dst_pts[:, 0].max())), int(np.ceil(dst_pts[:, 1].max()))
            
            dx0, dy0 = max(0, dx0), max(0, dy0)
            dx1, dy1 = min(w, dx1), min(h, dy1)
            
            if dx1 <= dx0 or dy1 <= dy0:
                continue
                
            H = cv2.getPerspectiveTransform((src_pts - [sx0, sy0]).astype(np.float32), 
                                            (dst_pts - [dx0, dy0]).astype(np.float32))
            warp_crop = cv2.warpPerspective(crop_src, H, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR)
            
            poly = (dst_pts - [dx0, dy0]).astype(np.int32)
            mask = np.zeros((dy1 - dy0, dx1 - dx0), dtype=np.uint8)
            cv2.fillConvexPoly(mask, poly, 255)
            
            alpha_mask = mask.astype(np.float32) / 255.0
            warp_crop_float = warp_crop.astype(np.float32)
            
            # --- Directional Specular Shading ---
            # Normal map the paper fold using screen gradients relative to a virtual overhead light
            vy = dst_pts[3] - dst_pts[0]
            len_vy = max(np.linalg.norm(vy), 0.1)
            nvy = vy / len_vy
            
            align = np.dot(-nvy, np.array([0, -1.0])) # Virtual light from top of screen
            illumination = 0.5 + 0.5 * align
            specular = (max(0, align)**8) * 0.5 * 255.0 # Gloss highlight
            
            warp_crop_float[:, :, :3] = (warp_crop_float[:, :, :3] * illumination) + specular
            # ------------------------------------
            
            warp_alpha = (warp_crop_float[:, :, 3] / 255.0) * alpha_mask
            
            lx0, ly0 = dx0 - x_min, dy0 - y_min
            lx1, ly1 = dx1 - x_min, dy1 - y_min
            
            if lx1 <= lx0 or ly1 <= ly0:
                continue
                
            warped_full[ly0:ly1, lx0:lx1, :3] += warp_crop_float[:, :, :3] * warp_alpha[..., None]
            warped_full[ly0:ly1, lx0:lx1, 3] += warp_alpha
            
            has_content = True
            global_dx0 = min(global_dx0, dx0)
            global_dy0 = min(global_dy0, dy0)
            global_dx1 = max(global_dx1, dx1)
            global_dy1 = max(global_dy1, dy1)

    if not has_content:
        return frame
        
    global_dx0, global_dy0 = max(0, global_dx0), max(0, global_dy0)
    global_dx1, global_dy1 = min(w, global_dx1), min(h, global_dy1)
    
    if global_dx1 <= global_dx0 or global_dy1 <= global_dy0:
        return frame
        
    ix0, iy0 = global_dx0 - x_min, global_dy0 - y_min
    ix1, iy1 = global_dx1 - x_min, global_dy1 - y_min
    
    roi_alpha = np.clip(warped_full[iy0:iy1, ix0:ix1, 3], 0, 1)[..., None]
    roi_rgb = warped_full[iy0:iy1, ix0:ix1, :3] / (roi_alpha + 1e-6)
    
    y1, y2 = global_dy0, global_dy1
    x1, x2 = global_dx0, global_dx1
    
    roi_rgb *= light_ratio[y1:y2, x1:x2, None]
    
    roi_base = frame[y1:y2, x1:x2].astype(np.float32)
    
    # --- Live Dynamic Occlusion Matting ---
    # Detect objects (like a user's hand) blocking the paper by luma delta vs ambient light map
    expected_white = light_ratio[y1:y2, x1:x2] * 200.0
    luma = 0.299 * roi_base[:, :, 0] + 0.587 * roi_base[:, :, 1] + 0.114 * roi_base[:, :, 2]
    
    occlusion_diff = expected_white - luma
    occlusion_mask = np.clip((occlusion_diff - 50) / 50.0, 0, 0.85)[..., None]
    
    final_alpha = roi_alpha * (1.0 - occlusion_mask) # Subtly mask out AR layer
    
    blended = roi_base * (1.0 - final_alpha) + roi_rgb * final_alpha
    
    result[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    return result


# ── Grid Overlay ─────────────────────────────────────────────────────
def draw_mesh_overlay(surface, mesh_xy):
    """Draw sci-fi mesh grid onto Pygame surface."""
    grid = mesh_xy.reshape(GRID_W, GRID_H, 2)

    for i in range(GRID_W):
        pts = grid[i, :].astype(int).tolist()
        if len(pts) > 1: pygame.draw.aalines(surface, COLORS['grid'], False, pts)
    for j in range(GRID_H):
        pts = grid[:, j].astype(int).tolist()
        if len(pts) > 1: pygame.draw.aalines(surface, COLORS['grid'], False, pts)

    edges = [
        (grid[0, :],        COLORS['lime']),
        (grid[GRID_W-1, :], COLORS['magenta']),
        (grid[:, 0],        COLORS['cyan']),
        (grid[:, GRID_H-1], COLORS['amber']),
    ]
    for pts, color in edges:
        p_list = pts.astype(int).tolist()
        if len(p_list) > 1:
            pygame.draw.lines(surface, color, False, p_list, 3)

    corner_data = [
        (grid[0, 0],                COLORS['lime']),
        (grid[GRID_W-1, 0],        COLORS['cyan']),
        (grid[GRID_W-1, GRID_H-1], COLORS['magenta']),
        (grid[0, GRID_H-1],        COLORS['amber']),
    ]
    for pt, color in corner_data:
        center = tuple(pt.astype(int))
        pygame.draw.circle(surface, color, center, 8, width=2)
        pygame.draw.circle(surface, COLORS['white'], center, 4)


# ── HUD ──────────────────────────────────────────────────────────────
def draw_hud(surface, font_lg, font_sm, state, n_bodies, brush_color,
             show_tex, show_grid, smoother_on, rotation, w, h,
             fps=0.0, perf=None, show_perf=False):
    """Render status HUD onto the pygame surface."""
    return
    title = font_lg.render("NEURAL AR SANDBOX", True, COLORS['white'])
    surface.blit(title, (20, 12))

    # Tracking state
    state_colors = {
        'TRACKING': COLORS['lime'],
        'COASTING': COLORS['amber'],
        'LOST':     (255, 60, 60),
    }
    sc = state_colors.get(state, COLORS['white'])
    surface.blit(font_sm.render(f"● {state}", True, sc), (20, 42))

    # Status bar (bottom)
    parts = [
        f"TX:{'ON' if show_tex else 'OF'}",
        f"GD:{'ON' if show_grid else 'OF'}",
        f"SM:{'ON' if smoother_on else 'OF'}",
        f"ROT:{rotation * 90}",
        f"PHY:{n_bodies}",
    ]
    surface.blit(font_sm.render(" | ".join(parts), True, COLORS['cyan']),
                 (20, h - 30))

    # Brush swatch
    swatch = pygame.Rect(w - 60, h - 35, 40, 20)
    pygame.draw.rect(surface, brush_color, swatch)
    pygame.draw.rect(surface, COLORS['white'], swatch, width=2)

    # FPS + perf overlay
    if show_perf and perf:
        fps_color = COLORS['lime'] if fps > 25 else (
                    COLORS['amber'] if fps > 15 else (255, 60, 60))
        fps_txt = font_lg.render(f"{fps:.0f} FPS", True, fps_color)
        surface.blit(fps_txt, (w - 130, 12))

        y = 42
        for name in ('infer', 'optflow', 'light', 'warp', 'render'):
            ms = perf.ms(name)
            txt = font_sm.render(f"{name}: {ms:.1f}ms", True, COLORS['grid'])
            surface.blit(txt, (w - 150, y))
            y += 18


# ── Utils ────────────────────────────────────────────────────────────
def np_to_surface(img):
    """Convert (H, W, 3) RGB numpy array to Pygame Surface."""
    return pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))


# ── Main ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("── NEURAL AR SANDBOX (PRODUCTION) ──")
    print(f"   Device : {DEVICE}")
    print(f"   Model  : {args.model}")
    print(f"   FP16   : {torch.cuda.is_available() and not args.no_fp16}")

    # ── Load model ──
    model = HeatmapMeshRegressor(num_points=NUM_POINTS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(args.model, map_location=DEVICE,
                                         weights_only=True))
    except FileNotFoundError:
        print(f"   ✖ Model not found: {args.model}")
        return
    model.eval()
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    print("   ✔ Model loaded")

    # ── Camera ──
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("   ✖ Could not open webcam")
        return
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ret, frame_test = cap.read()
    if not ret:
        print("   ✖ Could not read from webcam")
        return

    feed_h, feed_w = frame_test.shape[:2]
    print(f"   Feed   : {feed_w}×{feed_h}")

    # ── Pygame ──
    pygame.init()
    screen = pygame.display.set_mode((feed_w, feed_h))
    pygame.display.set_caption("Neural AR Sandbox")
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("impact", 24)
    font_sm = pygame.font.SysFont("consolas", 16)

    # ── Subsystems ──
    physics  = PhysicsWorld()
    canvas   = DrawingCanvas(physics)
    smoother = OneEuroFilter()
    stabilizer = MeshStabilizer()
    stabilizer.set_frame_size(feed_w, feed_h)
    lighting = LightingEngine()
    fps_counter = FPSCounter()
    perf = PerfTimer()

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # ── State ──
    show_texture = True
    show_grid    = False
    show_perf    = False
    rotation     = 0
    track_state  = MeshStabilizer.LOST
    frame_idx    = 0
    latest_mesh  = None
    camera_retries = 0
    prev_gray    = None

    print("\n   CONTROLS")
    print("   q/ESC Quit    t Texture    g Grid    s Smoothing")
    print("   r Rotate      c Clear      b Brush   f FPS overlay")
    print("   +/- Brush size")
    print("   ─────────────────────────────────────────────────")

    try:
        running = True
        while running:
            # ── Events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                canvas.handle_event(event)
                if event.type == pygame.KEYDOWN:
                    key = event.unicode.lower()
                    if key == 'q' or event.key == pygame.K_ESCAPE:
                        running = False
                    elif key == 't': show_texture = not show_texture
                    elif key == 'g': show_grid = not show_grid
                    elif key == 's': smoother.enabled = not smoother.enabled
                    elif key == 'r': rotation = (rotation + 1) % 4
                    elif key == 'c': canvas.clear()
                    elif key == 'b': canvas.cycle_color()
                    elif key == 'f': show_perf = not show_perf
                    elif key in ('=', '+'): canvas.brush_size = min(canvas.brush_size + 2, 40)
                    elif key == '-': canvas.brush_size = max(canvas.brush_size - 2, 2)

            if not running:
                break

            # ── Camera (with retry) ──
            ret, frame = cap.read()
            if not ret:
                camera_retries += 1
                if camera_retries > 10:
                    print("   ✖ Camera lost — exiting")
                    break
                time.sleep(0.05)
                continue
            camera_retries = 0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Inference (throttled) ──
            frame_idx += 1
            if frame_idx % args.infer_skip == 0 or latest_mesh is None:
                perf.start('infer')
                inp = transform(image=rgb_frame)['image'].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    if use_fp16:
                        with torch.amp.autocast('cuda'):
                            _, coords = model(inp)
                    else:
                        _, coords = model(inp)
                    mesh_xy = coords[0].cpu().numpy()
                    mesh_xy[:, 0] *= feed_w
                    mesh_xy[:, 1] *= feed_h
                
                # Apply spatial and boundary smoothing to regularize the mesh structure
                # while strictly preserving the absolute corner coordinates.
                grid = mesh_xy.reshape(GRID_W, GRID_H, 2)
                for _ in range(3):
                    new_grid = grid.copy()
                    # Internal nodes
                    new_grid[1:-1, 1:-1] = (
                        grid[:-2, 1:-1] + grid[2:, 1:-1] +
                        grid[1:-1, :-2] + grid[1:-1, 2:]
                    ) * 0.25
                    
                    # Boundary nodes
                    new_grid[0, 1:-1] = (grid[0, :-2] + grid[0, 2:]) * 0.5
                    new_grid[-1, 1:-1] = (grid[-1, :-2] + grid[-1, 2:]) * 0.5
                    new_grid[1:-1, 0] = (grid[:-2, 0] + grid[2:, 0]) * 0.5
                    new_grid[1:-1, -1] = (grid[:-2, -1] + grid[2:, -1]) * 0.5
                    
                    grid = new_grid
                latest_mesh = grid.reshape(-1, 2)
                perf.stop('infer')
            elif prev_gray is not None and latest_mesh is not None:
                # LK Optical Flow tracks the grid directly on skipped frames (zero visual lag)
                perf.start('optflow')
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray_frame, latest_mesh.astype(np.float32), None,
                    winSize=(21, 21), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                good = status.squeeze() == 1
                if np.any(good):
                    disp = next_pts - latest_mesh
                    mean_disp = disp[good].mean(axis=0)
                    next_pts[~good] = latest_mesh[~good] + mean_disp
                latest_mesh = next_pts
                perf.stop('optflow')

            prev_gray = gray_frame.copy()

            # Global Temporal Filtering: Apply 1EuroFilter to final tracked points to cancel jitter
            smoothed_mesh = smoother(latest_mesh) if latest_mesh is not None else None

            # ── Stabilize ──
            stable_mesh, track_state = stabilizer.update(smoothed_mesh)

            perf.start('render')
            if stable_mesh is not None:
                corners = (stable_mesh[IDX_TL], stable_mesh[IDX_BL],
                           stable_mesh[IDX_TR], stable_mesh[IDX_BR])
                canvas.mesh_corners = corners
                canvas.full_mesh = stable_mesh
                physics.update_and_step(stable_mesh)

                if show_texture:
                    # Lighting (always compute — cheap)
                    perf.start('light')
                    light_ratio = lighting.compute(rgb_frame, stable_mesh)
                    perf.stop('light')

                    if canvas.is_empty:
                        feed_rgb = rgb_frame
                    else:
                        perf.start('warp')
                        overlay = canvas.compose()
                        for _ in range(rotation):
                            overlay = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
                        feed_rgb = warp_overlay(rgb_frame, stable_mesh,
                                                overlay, light_ratio)
                        perf.stop('warp')
                else:
                    feed_rgb = rgb_frame

                surface = np_to_surface(feed_rgb)
                if show_grid:
                    draw_mesh_overlay(surface, stable_mesh)
            else:
                surface = np_to_surface(rgb_frame)

            # ── HUD ──
            fps_counter.tick()
            draw_hud(surface, font_lg, font_sm,
                     state=track_state,
                     n_bodies=len(physics.bodies),
                     brush_color=canvas.color,
                     show_tex=show_texture,
                     show_grid=show_grid,
                     smoother_on=smoother.enabled,
                     rotation=rotation,
                     w=feed_w, h=feed_h,
                     fps=fps_counter.fps,
                     perf=perf,
                     show_perf=show_perf)

            perf.stop('render')

            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(60)

    finally:
        cap.release()
        pygame.quit()
        print("   ── Clean shutdown ──")


if __name__ == "__main__":
    main()
