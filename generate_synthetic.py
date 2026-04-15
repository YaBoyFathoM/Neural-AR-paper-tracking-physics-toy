"""
Synthetic Paper Dataset Generator
==================================
Generates unlimited training images of a white paper sheet with:
  - Random 3D perspective (camera position/angle)
  - Paper deformations (bend, curl, wave)
  - Background variation (solid colors, gradients, textures)
  - Lighting variation (brightness, shadows)

Mesh labels are generated automatically from the 3D vertex positions
projected into 2D image coordinates — zero manual labeling needed.

Usage:
  python generate_synthetic.py --count 500 --out_dir ./SyntheticPaper

Output:
  <out_dir>/images/synth_00001.jpg
  <out_dir>/labels/synth_00001.json  (same format as OpenCamera_Mesh_Labels)
"""
# --- DATA STANDARDS ---
# Corner Order: [TL, TR, BR, BL] (Clockwise)
#   0: Top-Left, 1: Top-Right, 2: Bottom-Right, 3: Bottom-Left
# Colors: 1:Green, 2:Blue, 3:Red, 4:Yellow
# Orientation: Raw pixels (no EXIF transpose).
# ──────────────────────────────────────────────────────────────────────

import os
import json
import argparse
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Grid Settings (Labels)
GRID_W = 5
GRID_H = 5
NUM_POINTS = GRID_W * GRID_H
IMG_W, IMG_H = 768, 768  # Output image size

# High-Resolution Render Mesh (for smooth paper edges)
RENDER_W = 25 
RENDER_H = 34


# ──────────────────────────────────────────────────────────────────────
#  3D PAPER MESH
# ──────────────────────────────────────────────────────────────────────
def create_flat_paper(grid_w=RENDER_W, grid_h=RENDER_H, paper_w=215.9, paper_h=298.45):
    """
    Create a flat 3D paper mesh centered at origin.
    Default dimensions: 215.9mm × 298.45mm.
    Returns (grid_w * grid_h, 3) array of [x, y, z] points.
    """
    # Col-Major indexing (matches mesh_dragger.py standard)
    # Index = col * grid_h + row
    # mesh[0]=TL, mesh[grid_h-1]=BL, mesh[(grid_w-1)*grid_h]=TR, mesh[-1]=BR
    xs = np.linspace(-paper_w / 2, paper_w / 2, grid_w)
    ys = np.linspace(-paper_h / 2, paper_h / 2, grid_h)
    
    mesh = np.zeros((grid_w * grid_h, 3), dtype=np.float64)
    for c in range(grid_w):
        for r in range(grid_h):
            mesh[c * grid_h + r] = [xs[c], ys[r], 0.0]
    return mesh


def apply_bend_x(mesh, grid_size, strength):
    """Bend paper along X axis (like a page curl left-right)."""
    mesh = mesh.copy()
    xs = mesh[:, 0]
    x_min, x_max = xs.min(), xs.max()
    x_range = x_max - x_min
    if x_range < 1e-6:
        return mesh
    t = (xs - x_min) / x_range  # 0 to 1
    mesh[:, 2] += strength * np.sin(t * np.pi)
    return mesh


def apply_bend_y(mesh, grid_size, strength):
    """Bend paper along Y axis (like a page curl top-bottom)."""
    mesh = mesh.copy()
    ys = mesh[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_range = y_max - y_min
    if y_range < 1e-6:
        return mesh
    t = (ys - y_min) / y_range
    mesh[:, 2] += strength * np.sin(t * np.pi)
    return mesh


def apply_wave(mesh, grid_size, amplitude, frequency):
    """Apply a sinusoidal wave deformation."""
    mesh = mesh.copy()
    xs = mesh[:, 0]
    x_min, x_max = xs.min(), xs.max()
    x_range = x_max - x_min
    if x_range < 1e-6:
        return mesh
    t = (xs - x_min) / x_range
    mesh[:, 2] += amplitude * np.sin(t * frequency * np.pi)
    return mesh


def apply_soft_drag(mesh, grid_size, strength, radius=0.5):
    """
    Move a random point and propagate movement with Gaussian falloff.
    strength: movement in Z (mm)
    radius: falloff radius (relative to paper size)
    """
    mesh = mesh.copy()
    # Select random point on mesh
    idx = np.random.randint(len(mesh))
    target_pt = mesh[idx].copy()
    
    # Calculate paper diagonal for relative radius
    diag = np.linalg.norm(mesh.max(axis=0) - mesh.min(axis=0))
    sig = radius * diag
    
    dists = np.linalg.norm(mesh - target_pt, axis=1)
    weights = np.exp(-(dists**2) / (2 * sig**2))
    
    mesh[:, 2] += strength * weights
    return mesh


def apply_random_twists(mesh, num_twists=5):
    """Apply multiple random soft drags to create complex curvature."""
    for _ in range(num_twists):
        # High strength local deformations
        strength = np.random.uniform(-80, 80)
        radius = np.random.uniform(0.1, 0.4) # Smaller radius for sharper local bends
        mesh = apply_soft_drag(mesh, None, strength, radius)
        
        # Large scale global bends
        strength = np.random.uniform(-40, 40)
        radius = np.random.uniform(0.5, 1.2)
        mesh = apply_soft_drag(mesh, None, strength, radius)
    return mesh


def apply_sharp_fold(mesh, strength_deg=None):
    """Pick a random line across the paper and 'fold' the mesh around it."""
    mesh = mesh.copy()
    if strength_deg is None:
        strength_deg = np.random.uniform(-25, 25)
    
    # Random fold line: origin and direction
    idx = np.random.randint(len(mesh))
    origin = mesh[idx].copy()
    
    # Direction vector on XY plane
    angle = np.random.uniform(0, 2 * np.pi)
    dir_v = np.array([np.cos(angle), np.sin(angle), 0.0])
    
    # Normal to the fold line on XY plane
    normal = np.array([-np.sin(angle), np.cos(angle), 0.0])
    
    # Rotation matrix around dir_v
    theta = np.radians(strength_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = dir_v
    # Rodrigues' rotation formula: R = I + sin(theta)K + (1-cos(theta))K^2
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])
    R = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)
    
    # All points on one side of the line get rotated relative to origin
    relative_pts = mesh - origin
    sides = (relative_pts @ normal) > 0
    
    mesh[sides] = (relative_pts[sides] @ R.T) + origin
    return mesh


def apply_crumple(mesh, intensity=10):
    """Apply many high-frequency, low-amplitude micro-deformations."""
    for _ in range(intensity):
        strength = np.random.uniform(-15, 15)
        radius = np.random.uniform(0.02, 0.08)
        mesh = apply_soft_drag(mesh, None, strength, radius)
    return mesh


def apply_random_rotation(mesh):
    """Rotate mesh randomly around X, Y, Z axes."""
    # Push X/Y limits to very grazing angles
    rx = np.radians(np.random.uniform(-82, 82))
    ry = np.radians(np.random.uniform(-82, 82))
    rz = np.radians(np.random.uniform(-180, 180))
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    return mesh @ R.T


def project_to_2d(mesh, fov=60, cam_z=500):
    """Project 3D points to 2D image coordinates."""
    # Move paper away from camera along Z
    pts = mesh.copy()
    pts[:, 2] += cam_z
    
    # Simple perspective projection
    f = (IMG_W / 2) / np.tan(np.radians(fov / 2))
    
    pts_2d = np.zeros((pts.shape[0], 2))
    pts_2d[:, 0] = (pts[:, 0] * f / pts[:, 2]) + (IMG_W / 2)
    pts_2d[:, 1] = (pts[:, 1] * f / pts[:, 2]) + (IMG_H / 2)
    return pts_2d


# ──────────────────────────────────────────────────────────────────────
#  AUGMENTATIONS & RENDERING
# ──────────────────────────────────────────────────────────────────────
def augment_texture(img):
    """Apply brightness, contrast, saturation, sharpness, and blur variations."""
    out_img = img.copy()
    
    # 1. Brightness & Contrast
    alpha = np.random.uniform(0.5, 1.5)  # Contrast
    beta = np.random.randint(-60, 60)    # Brightness
    out_img = cv2.convertScaleAbs(out_img, alpha=alpha, beta=beta)
    
    # 2. Saturation
    if np.random.random() < 0.5:
        hsv = cv2.cvtColor(out_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = np.random.uniform(0.1, 4.0) # Boosted upper limit significantly
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        out_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    # 3. Sharpness / Blur
    blur_type = np.random.random()
    if blur_type < 0.3:
        # Motion blur
        kernel_size = np.random.randint(5, 25)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[(kernel_size-1)//2, :] = np.ones(kernel_size)
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), np.random.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel /= kernel.sum()
        out_img = cv2.filter2D(out_img, -1, kernel)
    elif blur_type < 0.5:
        # Gaussian blur (out of focus)
        k = np.random.choice([3, 5, 7, 9, 11])
        out_img = cv2.GaussianBlur(out_img, (k, k), 0)
    elif blur_type < 0.8:
        # Sharpen (Regular or Extreme Unsharp Mask)
        if np.random.random() < 0.5:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            out_img = cv2.filter2D(out_img, -1, kernel)
        else:
            # Extreme Unsharp mask to make subtle lines pop heavily
            blurred = cv2.GaussianBlur(out_img, (0, 0), 3.0)
            out_img = cv2.addWeighted(out_img, 2.5, blurred, -1.5, 0)
        
    # 4. Gauss Noise
    if np.random.random() < 0.3:
        noise = np.random.normal(0, np.random.uniform(5, 15), out_img.shape).astype(np.float32)
        out_img = np.clip(out_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
    return out_img

def apply_occlusions(img, pts_2d):
    """Draw random polygons/ellipses over the image to simulate occlusions/hands."""
    out_img = img.copy()
    for _ in range(np.random.randint(0, 4)):
        shape_type = np.random.randint(0, 2)
        # Random somewhat-dark color to simulate objects
        color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 200))
        
        # Pick a random point near the paper to occlude
        center = pts_2d[np.random.randint(len(pts_2d))]
        cx = int(center[0] + np.random.uniform(-100, 100))
        cy = int(center[1] + np.random.uniform(-100, 100))
        
        if shape_type == 0:
            axes = (np.random.randint(20, 100), np.random.randint(20, 100))
            angle = np.random.randint(0, 180)
            cv2.ellipse(out_img, (cx, cy), axes, angle, 0, 360, color, -1)
        else:
            num_pts = np.random.randint(3, 7)
            poly_pts = []
            for _ in range(num_pts):
                px = cx + np.random.randint(-120, 120)
                py = cy + np.random.randint(-120, 120)
                poly_pts.append([px, py])
            poly_pts = np.array([poly_pts], dtype=np.int32)
            cv2.fillPoly(out_img, poly_pts, color)
            
    # Apply slight blur to occlusions to integrate them better
    mask_diff = np.any(out_img != img, axis=-1).astype(np.float32)
    mask_diff = cv2.GaussianBlur(mask_diff, (7, 7), 2.0)[:, :, np.newaxis]
    blurred_out = cv2.GaussianBlur(out_img, (5, 5), 1.0)
    final = img * (1 - mask_diff) + blurred_out * mask_diff
    return np.clip(final, 0, 255).astype(np.uint8)

def apply_harsh_shadow(img):
    """Apply a random dark shadow casting over the image."""
    h, w = img.shape[:2]
    shadow = np.zeros((h, w), dtype=np.float32)
    
    # Define a random polygon that cuts across the image
    pt1 = (np.random.randint(-w, w*2), np.random.randint(-h, h*2))
    pt2 = (np.random.randint(-w, w*2), np.random.randint(-h, h*2))
    pt3 = (np.random.randint(-w, w*2), np.random.randint(-h, h*2))
    cv2.fillConvexPoly(shadow, np.array([pt1, pt2, pt3, (-w, h*2), (-w, -h)], dtype=np.int32), 1.0)
    
    shadow = cv2.GaussianBlur(shadow, (151, 151), 80)
    shadow_density = np.random.uniform(0.3, 0.7)
    
    img_float = img.astype(np.float32)
    img_float = img_float * (1.0 - shadow[:, :, np.newaxis] * shadow_density)
    return np.clip(img_float, 0, 255).astype(np.uint8)

def get_random_background(bg_dir):
    """Load a random image from bg_dir OR generate a gradient if none found."""
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(bg_dir, e)))
    
    if files:
        bg = cv2.imread(np.random.choice(files))
        if bg is not None:
            return cv2.resize(bg, (IMG_W, IMG_H))
            
    # Fallback Gradient
    bg = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    c1 = np.random.randint(0, 100, 3)
    c2 = np.random.randint(100, 200, 3)
    for y in range(IMG_H):
        alpha = y / IMG_H
        bg[y, :] = (1 - alpha) * c1 + alpha * c2
    return bg


def render_sample(pts_2d, grid_w, grid_h, texture_img, background_img, opacity=1.0):
    """Warp texture using stable triangulation to prevent black artifacts."""
    h, w = background_img.shape[:2]
    tex_h, tex_w = texture_img.shape[:2]
    
    # --- Bleed-Through Effect (Front-side see-through) ---
    if opacity < 1.0:
        # ghost_opacity is proportional to transparency
        ghost_opacity = (1.0 - opacity) * 2.0 # High sensitivity
        back_tex = cv2.flip(texture_img, 1) # Horizontal flip
        
        # Soften the ghost lines to simulate diffusion through paper pulp
        # Using a larger blur for more natural 'back-side' look
        back_tex_blurred = cv2.GaussianBlur(back_tex, (13, 13), 3.5)
        
        # Multiplicative blend: Faded back-side lines darken the front-side colors
        # We fade the back lines towards white (neutral in multiplication)
        back_faded = cv2.addWeighted(back_tex_blurred, ghost_opacity * 0.4, 
                                     np.full_like(back_tex_blurred, 255), 
                                     1.0 - (ghost_opacity * 0.4), 0)
        
        texture_img = (texture_img.astype(np.float32) * back_faded.astype(np.float32) / 255.0)
        texture_img = np.clip(texture_img, 0, 255).astype(np.uint8)

    paper_buffer = np.zeros((h, w, 3), dtype=np.float32)
    paper_counts = np.zeros((h, w), dtype=np.float32)
    
    # Source points on texture: col-major (matching mesh layout)
    xs = np.linspace(0, 1, grid_w)
    ys = np.linspace(0, 1, grid_h)
    
    grid_points_tex = []
    for px in xs:
        for py in ys:
            grid_points_tex.append([px * (tex_w - 1), py * (tex_h - 1)])
    grid_points_tex = np.array(grid_points_tex, dtype=np.float32).reshape(grid_w, grid_h, 2)
    
    # Target points on canvas: col-major
    grid_points_canvas = pts_2d.astype(np.float32).reshape(grid_w, grid_h, 2)
    
    # We split each quad into 2 triangles for stable affine warping
    EXPAND = 1.02
    
    for c in range(grid_w - 1):
        for r in range(grid_h - 1):
            # Col-major quad vertices
            v_tl = (c, r)
            v_tr = (c+1, r)
            v_br = (c+1, r+1)
            v_bl = (c, r+1)
            
            # Two triangles per quad
            triangles = [
                (v_tl, v_tr, v_bl),
                (v_tr, v_br, v_bl)
            ]
            
            for tri_indices in triangles:
                src_tri = np.array([grid_points_tex[i] for i in tri_indices], dtype=np.float32)
                dst_tri = np.array([grid_points_canvas[i] for i in tri_indices], dtype=np.float32)
                
                # Sub-pixel expansion
                center_dst = dst_tri.mean(axis=0)
                dst_expanded = center_dst + (dst_tri - center_dst) * EXPAND
                center_src = src_tri.mean(axis=0)
                src_expanded = center_src + (src_tri - center_src) * EXPAND
                
                M = cv2.getAffineTransform(src_expanded, dst_expanded)
                
                min_x = int(np.floor(dst_expanded[:, 0].min()))
                max_x = int(np.ceil(dst_expanded[:, 0].max()))
                min_y = int(np.floor(dst_expanded[:, 1].min()))
                max_y = int(np.ceil(dst_expanded[:, 1].max()))
                
                min_x, max_x = max(0, min_x), min(w - 1, max_x)
                min_y, max_y = max(0, min_y), min(h - 1, max_y)
                
                if max_x <= min_x or max_y <= min_y: continue
                
                qw, qh = max_x - min_x + 1, max_y - min_y + 1
                M_sub = M.copy()
                M_sub[0, 2] -= min_x
                M_sub[1, 2] -= min_y
                
                warped = cv2.warpAffine(texture_img, M_sub, (qw, qh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                
                mask = np.zeros((qh, qw), dtype=np.float32)
                cv2.fillConvexPoly(mask, (dst_expanded - [min_x, min_y]).astype(np.int32), 1.0)
                # Slight feathering
                mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
                
                paper_buffer[min_y:min_y+qh, min_x:min_x+qw] += warped.astype(np.float32) * mask[:, :, np.newaxis]
                paper_counts[min_y:min_y+qh, min_x:min_x+qw] += mask
                
    # Normalize paper and blend with background
    mask_visited = paper_counts > 0.001
    paper_final = paper_buffer.copy()
    paper_final[mask_visited] /= paper_counts[mask_visited][:, np.newaxis]
    
    paper_alpha = np.clip(paper_counts, 0, 1.0) * opacity
    paper_alpha = cv2.GaussianBlur(paper_alpha, (3, 3), 1.0)[:, :, np.newaxis]
    
    bg_float = background_img.astype(np.float32)
    
    # Optional Drop Shadow behind paper
    if np.random.random() > 0.3:
        shadow_mask = np.clip(paper_counts, 0, 1.0)
        # Offset drop shadow
        dx, dy = np.random.uniform(-15, 30), np.random.uniform(10, 40)
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        shadow_mask = cv2.warpAffine(shadow_mask, M_shift, (w, h))
        # Large blur for soft shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 20)
        
        shadow_intensity = np.random.uniform(0.3, 0.8)
        bg_float = bg_float * (1.0 - shadow_mask[:, :, np.newaxis] * shadow_intensity)
    
    final_img = (bg_float * (1.0 - paper_alpha)) + (paper_final * paper_alpha)
        
    return np.clip(final_img, 0, 255).astype(np.uint8)


def generate_one_sample(idx, out_img_dir, out_label_dir, texture_img, bg_dir, label_grid="9x12"):
    """Generate a single random sample and save it."""
    max_attempts = 40
    for attempt in range(max_attempts):
        # 1. Base mesh (ALWAYS high-res for smooth rendering)
        paper_w = np.random.uniform(200, 230)
        paper_h = paper_w * (298.45 / 215.9) # Keep A4-ish ratio
        mesh = create_flat_paper(RENDER_W, RENDER_H, paper_w, paper_h)
        
        # 2. Deformations (Physics)
        # Apply organic soft-drags (twists/local deformations)
        if np.random.random() < 0.95:
            num_twists = np.random.randint(4, 10)
            mesh = apply_random_twists(mesh, num_twists=num_twists)
            
        # Apply Hard Folds (Creases) -- new 50% chance
        if np.random.random() < 0.5:
            num_folds = np.random.randint(1, 4)
            for _ in range(num_folds):
                mesh = apply_sharp_fold(mesh)
        
        # Apply Micro-Crumple -- new 40% chance
        if np.random.random() < 0.4:
            mesh = apply_crumple(mesh, intensity=np.random.randint(10, 25))

        # Apply structured bends
        if np.random.random() < 0.7:
            mesh = apply_bend_x(mesh, None, np.random.uniform(-80, 80))
        if np.random.random() < 0.7:
            mesh = apply_bend_y(mesh, None, np.random.uniform(-80, 80))
        
        if np.random.random() < 0.4:
            mesh = apply_wave(mesh, None, np.random.uniform(10, 40), np.random.uniform(0.3, 3.0))

        # 3. 3D Pose
        mesh = apply_random_rotation(mesh)
        
        # 4. Perspective Projection (Closer camera allowed for extreme close-ups)
        # Expand Z-range for more distance/distance variety
        cam_z = np.random.uniform(150, 1800)
        fov = np.random.uniform(35, 85)
        pts_2d = project_to_2d(mesh, fov=fov, cam_z=cam_z)
        
        # 5. Cull if points are outside image bounds or paper is too small
        # We allow moderate bleeding (-100px/ +868px) to capture grazing perspectives
        # but require the grid to be mostly visible
        if (pts_2d[:, 0].min() < -100 or pts_2d[:, 0].max() > IMG_W + 100 or
            pts_2d[:, 1].min() < -100 or pts_2d[:, 1].max() > IMG_H + 100):
            continue
            
        # Col-major area: corners are 0(TL), RENDER_H-1(BL), (RENDER_W-1)*RENDER_H(TR), RENDER_W*RENDER_H-1(BR)
        area = cv2.contourArea(pts_2d[[0, RENDER_H-1, RENDER_W*RENDER_H-1, (RENDER_W-1)*RENDER_H]].astype(np.float32))
        if area < (IMG_W * IMG_H * 0.03): # Lowered to 3% for macro/distant shots
            continue
            
        # 6. Render with high-res mesh for smooth edges
        bg = get_random_background(bg_dir)
        texture_var = augment_texture(texture_img)
        # Aggressive range (0.5 to 1.0) to make sure we see clear bleed-through cases
        paper_opacity = np.random.uniform(0.5, 1.0)
        
        final_img = render_sample(pts_2d, RENDER_W, RENDER_H, texture_var, bg, opacity=paper_opacity)
        # 7. Post-processing (Shadows & Occlusions)
        if np.random.random() < 0.6:
            final_img = apply_harsh_shadow(final_img)
        if np.random.random() < 0.6:
            final_img = apply_occlusions(final_img, pts_2d)
        
        # 8. Labels (Sub-sample high-res mesh for the labels)
        img_name = f"synth_{idx:05d}.jpg"
        label_name = f"synth_{idx:05d}.json"
        
        # Reshape to Col-Major (W, H, 2/3)
        grid_2d = pts_2d.reshape(RENDER_W, RENDER_H, 2)
        grid_3d = mesh.reshape(RENDER_W, RENDER_H, 3)
        
        # Col-major corners: [TL, TR, BR, BL]
        # mesh[0]=TL, mesh[H-1]=BL, mesh[(W-1)*H]=TR, mesh[W*H-1]=BR
        corners = [
            grid_2d[0, 0].tolist(),               # TL
            grid_2d[RENDER_W-1, 0].tolist(),      # TR
            grid_2d[RENDER_W-1, RENDER_H-1].tolist(), # BR
            grid_2d[0, RENDER_H-1].tolist()       # BL
        ]

        output_data = {
            "image": img_name,
            "corners": corners
        }

        if label_grid == "dual":
            # 5x5 Grid
            c5 = [0, 6, 12, 18, 24]
            r5 = [0, 8, 16, 24, 33]
            mesh_5x5 = grid_2d[c5][:, r5].reshape(-1, 2).tolist()
            
            # 9x12 Grid
            c9 = [i*3 for i in range(9)]
            r12 = [i*3 for i in range(12)]
            mesh_9x12 = grid_2d[c9][:, r12].reshape(-1, 2).tolist()
            
            output_data.update({
                "mesh_5x5": mesh_5x5,
                "mesh_9x12": mesh_9x12,
                "grid_w_5x5": 5, "grid_h_5x5": 5,
                "grid_w_9x12": 9, "grid_h_9x12": 12
            })
        else:
            if label_grid == "5x5":
                col_indices = [0, 6, 12, 18, 24]
                row_indices = [0, 8, 16, 24, 33]
                out_grid_w, out_grid_h = 5, 5
            else:
                col_indices = [i*3 for i in range(9)]
                row_indices = [i*3 for i in range(12)]
                out_grid_w, out_grid_h = 9, 12
                
            sub_2d = grid_2d[col_indices][:, row_indices]
            
            output_data.update({
                "mesh": sub_2d.reshape(-1, 2).tolist(),
                "grid_w": out_grid_w,
                "grid_h": out_grid_h
            })
        
        # IO
        cv2.imwrite(os.path.join(out_img_dir, img_name), final_img)
        with open(os.path.join(out_label_dir, label_name), 'w') as f:
            json.dump(output_data, f, indent=4)
            
        return True

    print(f"  ⚠ Could not generate valid sample {idx} after {max_attempts} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic paper training data")
    parser.add_argument("--count", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--out_dir", type=str, 
                        default=r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\Synthetic_9x12_768_v3",
                        help="Output directory")
    parser.add_argument("--label_grid", type=str, choices=["5x5", "9x12", "dual"], default="dual", 
                        help="Sub-sample the output labels down to 5x5, or retain 9x12")
    parser.add_argument("--texture", type=str, default="paper_lines_golden.png", help="Path to paper texture image")
    parser.add_argument("--bg_dir", type=str, 
                        default=r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\backgrounds", 
                        help="Path to directory containing background images")
    parser.add_argument("--preview", action="store_true", help="Generate a preview.jpg of the first 16 samples")
    parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers (-1 for all CPUs)")
    args = parser.parse_args()

    # Paths
    out_img_dir = os.path.join(args.out_dir, "images")
    out_label_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    texture_img = None
    if args.texture:
        texture_img = cv2.imread(args.texture)
        if texture_img is None:
            print(f"Error: Could not load texture from {args.texture}")
            return
    
    from multiprocessing import Pool, cpu_count
    
    workers = args.workers if args.workers > 0 else cpu_count()
    print(f"Generating {args.count} samples using {workers} workers into {args.out_dir}...")
    
    tasks = [(i, out_img_dir, out_label_dir, texture_img, args.bg_dir, args.label_grid) for i in range(1, args.count + 1)]
    
    success_count = 0
    with Pool(processes=workers) as pool:
        # Use starmap to pass multiple arguments
        results = list(tqdm(pool.starmap(generate_one_sample, tasks), total=args.count, desc="Generating Samples"))
        success_count = sum(results)

    print(f"Success: {success_count}/{args.count} samples generated.")

    if args.preview and success_count >= 16:
        print("Creating preview.jpg grid...")
        rows, cols = 4, 4
        cell_w, cell_h = 320, 240
        grid_img = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for k in range(16):
            idx = k + 1
            img_path = os.path.join(out_img_dir, f"synth_{idx:05d}.jpg")
            label_path = os.path.join(out_label_dir, f"synth_{idx:05d}.json")
            
            img = cv2.imread(img_path)
            with open(label_path, 'r') as f:
                data = json.load(f)
            
            if img is None:
                continue
            img = cv2.resize(img, (cell_w, cell_h))

            # Draw mesh
            pts = np.array(data.get('mesh', data.get('mesh_9x12')))
            pts[:, 0] *= cell_w / IMG_W
            pts[:, 1] *= cell_h / IMG_H
            w_opt = data.get('grid_w', data.get('grid_w_9x12', GRID_W))
            h_opt = data.get('grid_h', data.get('grid_h_9x12', GRID_H))
            grid = pts.reshape(w_opt, h_opt, 2)

            # Colors from inspect_data.py
            EDGE_COLORS = {
                'top': (0, 255, 255),    # Yellow (BGR)
                'bottom': (255, 0, 0),   # Blue (BGR)
                'left':  (0, 255, 0),    # Green (BGR)
                'right': (0, 0, 255),    # Red (BGR)
                'inner': (80, 80, 80)
            }
            CORNER_MAP = {
                (0, 0): (0, 255, 0, "1"),              # TL (Green)
                (w_opt-1, 0): (255, 0, 0, "2"),       # TR (Blue)
                (w_opt-1, h_opt-1): (0, 0, 255, "3"), # BR (Red)
                (0, h_opt-1): (0, 255, 255, "4")      # BL (Yellow)
            }

            # Draw mesh lines
            for r in range(h_opt):
                for c in range(w_opt):
                    p1 = tuple(grid[c, r].astype(int))
                    # Horizontal lines
                    if c < w_opt - 1:
                        p2 = tuple(grid[c+1, r].astype(int))
                        color = EDGE_COLORS['inner']
                        if r == 0: color = EDGE_COLORS['top']
                        elif r == h_opt-1: color = EDGE_COLORS['bottom']
                        cv2.line(img, p1, p2, color, 1)
                    # Vertical lines
                    if r < h_opt - 1:
                        p2 = tuple(grid[c, r+1].astype(int))
                        color = EDGE_COLORS['inner']
                        if c == 0: color = EDGE_COLORS['left']
                        elif c == w_opt-1: color = EDGE_COLORS['right']
                        cv2.line(img, p1, p2, color, 1)

            # Draw corner markers
            for (c, r), (color, _, _, label) in CORNER_MAP.items():
                pt = tuple(grid[c, r].astype(int))
                cv2.circle(img, pt, 5, color, -1)
                cv2.circle(img, pt, 5, (255, 255, 255), 1)
                # Sub-labels
                cv2.putText(img, label, (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            row_i, col_i = divmod(k, cols)
            grid_img[row_i * cell_h:(row_i + 1) * cell_h, 
                     col_i * cell_w:(col_i + 1) * cell_w] = img
        
        preview_path = os.path.join(args.out_dir, "preview.jpg")
        cv2.imwrite(preview_path, grid_img)
        print(f"   Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
