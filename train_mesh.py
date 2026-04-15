import os
import json
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves to file, no frozen windows
import matplotlib.pyplot as plt

IMAGE_DIR = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\combined_768\images"
MESH_DIR  = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\combined_768\labels"
SYNTH_IMAGE_DIR = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\Synthetic_9x12_768_v4_3k\images"
SYNTH_MESH_DIR  = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\Synthetic_9x12_768_v4_3k\labels"
BATCH_SIZE = 1
ACCUMULATION_STEPS = 32  # Simulate Batch Size 32
EPOCHS = 100             # Full training run
STEPS_PER_EPOCH = 250   # Short trial run for verification
MAX_LR_BACKBONE = 1e-4
MAX_LR_HEAD = 1e-3
IMG_SIZE = 768
HEATMAP_SIZE = 384      # 2x upsample from 192x192 feature map (HRNet-W32 @ 768x768)
GRID_W = 9
GRID_H = 12
NUM_POINTS = GRID_W * GRID_H # 108 points
HEATMAP_SIGMA = 4.0     # Slightly smaller Sigma for denser points
JITTER_STD_PX = 2.0     # Smaller jitter for higher density precision
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────
# Start fresh for the new 9x12 architecture
FRESH_START = False  
# AUTO-HEALER LOGIC REMOVED: Dataset orientations are now verified.

# ──────────────────────────────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────────────────────────────
def generate_gaussian_heatmap(cx, cy, H, W, sigma):
    x = torch.arange(W, dtype=torch.float32)
    y = torch.arange(H, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    heatmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap

class MeshDataset(Dataset):
    def __init__(self, sample_list, transform=None, is_train=False):
        """
        sample_list: list of (img_dir, mesh_dir, filename, folder_idx) tuples
        """
        self.samples = sample_list
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, mesh_dir, filename, folder_idx = self.samples[idx]
        json_path = os.path.join(mesh_dir, filename)
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_name = data['image']
        img_path = os.path.join(img_dir, img_name)
        # Load with PIL WITHOUT exif_transpose — mesh_dragger labels coordinates
        # in the raw pixel space (Image.open without rotation), so we must match.
        pil_img = Image.open(img_path).convert('RGB')
        image = np.array(pil_img)

        # Support both new specific keys and legacy 'mesh' keys
        if 'mesh_9x12' in data:
            keypoints_2d = np.array(data['mesh_9x12'], dtype=np.float32)
        elif 'mesh' in data:
            keypoints_2d = np.array(data['mesh'], dtype=np.float32)
        else:
            raise KeyError(f"Neither 'mesh_9x12' nor 'mesh' found in {json_path}")
        
        kp_2d = keypoints_2d[:, :2]

        # Load 3D Mesh Ground Truth if available (for UVDoc Dual-Task)
        has_3d = 'mesh_3d' in data
        if has_3d:
            gt_3d = np.array(data['mesh_3d'], dtype=np.float32)
        else:
            gt_3d = np.zeros((kp_2d.shape[0], 3), dtype=np.float32)

        # Dataset Audit (Ensure 108-point 9x12 grid compliance)
        num_expected = GRID_W * GRID_H
        if kp_2d.shape[0] != num_expected:
            raise ValueError(f"Dataset has {kp_2d.shape[0]} points but model expects {num_expected}!")

        if self.transform:
            transformed = self.transform(image=image, keypoints=kp_2d.tolist())
            image = transformed['image']
            kp_2d = np.array(transformed['keypoints'], dtype=np.float32)
            
        # Jitter removed. Uncorrelated structural noise actively teaches grid instability.

        # Normalize X, Y to 0-1
        coords_norm = np.zeros((NUM_POINTS, 2), dtype=np.float32)
        coords_norm[:, :2] = kp_2d / IMG_SIZE
        coords_norm = torch.tensor(coords_norm, dtype=torch.float32)

        # Generate ground-truth heatmaps at HEATMAP_SIZE resolution
        scale = HEATMAP_SIZE / IMG_SIZE
        heatmaps = torch.zeros(NUM_POINTS, HEATMAP_SIZE, HEATMAP_SIZE)
        for i, (px, py) in enumerate(kp_2d):
            hx = px * scale
            hy = py * scale
            heatmaps[i] = generate_gaussian_heatmap(hx, hy, HEATMAP_SIZE, HEATMAP_SIZE, HEATMAP_SIGMA)

        # Generate Paper Mask (1.0 inside paper poly, 0.0 outside)
        mask = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
        # Use all perimeter points for accurate bent paper mask (Col-Major 9x12)
        # Formula: w * 12 + h
        # Top (h=0): 0..8 | Right (w=8): 1..11 | Bottom (h=11): 7..0 | Left (w=0): 10..1
        perimeter_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96, 
                         97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                         95, 83, 71, 59, 47, 35, 23, 11,
                         10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        poly = (kp_2d[perimeter_idx] * scale).astype(np.int32)
        cv2.fillPoly(mask, [poly], 1.0)
        paper_mask = torch.from_numpy(mask).unsqueeze(0) # (1, H, W)

        # Generate Edge Boundary Ribbon Mask (the silhouette)
        # We erode the paper mask and subtract it from the original to create a target "ribbon"
        kernel = np.ones((7, 7), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary_mask_np = mask - eroded
        boundary_mask = torch.from_numpy(boundary_mask_np).unsqueeze(0)

        # Metadata for flip-tracking and auto-healing
        img_dir, mesh_dir, fname, folder_idx = self.samples[idx]
        is_synth = "Synthetic" in img_dir
        
        sample_meta = {
            'index': idx,
            'folder_idx': folder_idx, 
            'filename': fname,
            'mesh_dir': mesh_dir,
            'is_synth': is_synth,
            'poison_level': data.get('poison_level', 0)
        }

        return image, heatmaps, coords_norm, paper_mask, boundary_mask, sample_meta

# ──────────────────────────────────────────────────────────────────────
#  AUGMENTATION
# ──────────────────────────────────────────────────────────────────────
def get_train_transforms():
    # Removed Perspective and ShiftScaleRotate: since we predict physical 3D properties relative
    # to the camera, 2D affine warping desyncs the 2D image from the true 3D shape ground truth.
    # The synthetic dataset already rigorously applies 3D pose and camera position augmentations natively.
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.4),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        
        # --- Aggressive Massive-Occlusion Training Augmentations ---
        # Simulates a user's arm/hand completely covering the paper
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(100, 350), hole_width_range=(100, 350), fill=(200, 150, 100), p=0.4), # Flesh tones
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(50, 250), hole_width_range=(50, 250), fill=(20, 20, 20), p=0.4),    # Shadows/Sleeves
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# ──────────────────────────────────────────────────────────────────────
#  MODEL
# ──────────────────────────────────────────────────────────────────────
def dsnt(heatmaps):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    probs = F.softmax(flat / 1.0, dim=-1).reshape(B, K, H, W)
    x_coords = torch.linspace(0, 1, W, device=heatmaps.device)
    y_coords = torch.linspace(0, 1, H, device=heatmaps.device)
    x = (probs.sum(dim=2) * x_coords).sum(dim=-1)
    y = (probs.sum(dim=3) * y_coords).sum(dim=-1)
    return torch.stack([x, y], dim=-1)

class HeatmapMeshRegressor(nn.Module):
    def __init__(self, num_points=25):
        super().__init__()
        self.num_points = num_points

        # HRNet-W32 backbone -> (B, 128, 96, 96)
        self.backbone = timm.create_model(
            'hrnet_w32', pretrained=True, features_only=True,
            out_indices=(1,) 
        )

        # Heatmap head: Upsample 192x192 -> 384x384 (HRNet-W32 @ 768x768)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_points, 1)  # num_points heatmaps @ 384x384
        )

    def forward(self, x):
        features = self.backbone(x)[0]          # (B, 128, 192, 192)
        heatmaps = self.heatmap_head(features)  # (B, num_points, 384, 384)
        coords_xy = dsnt(heatmaps)              # (B, num_points, 2)
        
        return heatmaps, coords_xy

# ──────────────────────────────────────────────────────────────────────
#  MODEL EMA (Removed for Stability)
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
#  LOSS
# ──────────────────────────────────────────────────────────────────────
def calculate_quad_area_norm(coords):
    """
    Calculate area of a quadrilateral using shoelace formula.
    coords: (B, 4, 2) in normalized 0-1 range.
    Returns: (B,) areas.
    """
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    # Shoelace formula: 0.5 * |x0y1 + x1y2 + x2y3 + x3y0 - (y0x1 + y1x2 + y2x3 + y3x0)|
    area = 0.5 * torch.abs(
        x[:, 0]*y[:, 1] + x[:, 1]*y[:, 2] + x[:, 2]*y[:, 3] + x[:, 3]*y[:, 0] -
        (y[:, 0]*x[:, 1] + y[:, 1]*x[:, 2] + y[:, 2]*x[:, 3] + y[:, 3]*x[:, 0])
    )
    return area

class MeshLoss2D(nn.Module):
    def __init__(self, pos_weight=10.0, struct_weight=10.0, sil_weight=1.0):
        super().__init__()
        self.pos_w = pos_weight     # Primary XY / Corner / Edge signals
        self.struct_w = struct_weight # Folding / Expansion / Uniformity
        self.sil_w = sil_weight     # Mask / Silhouette consistency
        
        # Internal Best-Practice Multipliers (No longer need separate tuning)
        self.edge_mult = 5.0
        self.corner_mult = 10.0

    def forward(self, p_hm, g_hm, p_xy, g_xy, g_mask, g_boundary_mask, poison_weights=None):
        B, K, H, W = p_hm.shape

        # 1. Per-Sample Area Weights
        # Use corner points (0, 4, 24, 20) to define the paper quadrilateral area
        with torch.no_grad():
            # Use average of two diagonals for scale-invariant normalization
            # Col-Major 9x12 Corners: 0(TL), 96(TR), 107(BR), 11(BL)
            ptr_tl, ptr_tr, ptr_br, ptr_bl = g_xy[:, 0, :], g_xy[:, 96, :], g_xy[:, 107, :], g_xy[:, 11, :]
            diag1 = torch.norm(ptr_tl - ptr_br, dim=-1)
            diag2 = torch.norm(ptr_tr - ptr_bl, dim=-1)
            paper_scale = (diag1 + diag2) / 2.0
            
            # Prevent division by zero and extreme weighting for tiny/distant fragments
            # Using sqrt(paper_scale) softens the weighting so distant papers get more focus
            # but don't completely drown out the gradient with resolution-limited noise.
            paper_scale_clamped = torch.clamp(paper_scale, min=0.05)
            
            sample_weights = 1.0 / torch.sqrt(paper_scale_clamped) 
            sample_weights = sample_weights / sample_weights.mean() # Normalize batch mean to 1.0
            
            # FACTOR IN POISON TIER LOSS EXCLUSION
            if poison_weights is not None:
                sample_weights = sample_weights * poison_weights

        # Col-Major 9x12 Corners: TL=0, BL=11, TR=96, BR=107
        corner_indices = [0, 11, 96, 107]
        edge_indices = []
        # Left/Right edges (w=0 or w=8, h=1..10)
        for h in range(1, 11):
            edge_indices.extend([h, 96 + h])
        # Top/Bottom edges (h=0 or h=11, w=1..7)
        for w in range(1, 8):
            edge_indices.extend([w * 12 + 0, w * 12 + 11])
            
        point_weights = torch.ones(p_xy.size(1), device=p_xy.device)
        point_weights[edge_indices] = self.edge_mult
        point_weights[corner_indices] = self.corner_mult
        
        # Combined Weight Matrix (B, 108)
        combined_weights = sample_weights.unsqueeze(1) * point_weights.unsqueeze(0)

        # 3. Positional Loss (XY Regression)
        # Probabilistic Heatmap Match (Heatmap loss is removed as XY DSNT is superior)
        # but we keep the logic for mask sampling if needed.
        p_probs = F.softmax(p_hm.view(B, K, -1), dim=-1).view(B, K, H, W)
        xy_loss_raw = F.l1_loss(p_xy, g_xy, reduction='none')        # (B, 108, 2)
        
        # Apply combined weights
        pos_loss = (xy_loss_raw.mean(dim=-1) * combined_weights).mean()
        
        # 4. Silhouette Loss (Stay on Paper)
        # Global: All heatmaps must not be outside the paper mask
        mask_penalty = (p_probs.sum(dim=1, keepdim=True) * (1.0 - g_mask)).mean()
        
        # Boundary Anchor: Edge heatmaps must be strictly ON the boundary ribbon
        perimeter_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96, 
                         97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                         95, 83, 71, 59, 47, 35, 23, 11,
                         10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        edge_probs = p_probs[:, perimeter_idx, :, :]
        boundary_penalty = (edge_probs.mean(dim=1, keepdim=True) * (1.0 - g_boundary_mask)).mean()
        
        sil_loss = mask_penalty + boundary_penalty

        # 5. Structural Loss (Geometric Integrity)
        grid_pts = p_xy.view(B, 9, 12, 2)
        
        # a. Anti-Folding (Signed Area)
        A_pts, B_pts, C_pts, D_pts = grid_pts[:, :-1, :-1, :], grid_pts[:, 1:, :-1, :], grid_pts[:, 1:, 1:, :], grid_pts[:, :-1, 1:, :]
        vAB, vAC, vAD = B_pts - A_pts, C_pts - A_pts, D_pts - A_pts
        area1 = vAB[..., 0] * vAC[..., 1] - vAB[..., 1] * vAC[..., 0]
        area2 = vAC[..., 0] * vAD[..., 1] - vAC[..., 1] * vAD[..., 0]
        fold_penalty = (F.relu(1e-4 - area1)**2 + F.relu(1e-4 - area2)**2).mean()

        # b. Anti-Bunching (Expansion / Variance)
        edge_h = torch.norm(grid_pts[:, 1:, :, :] - grid_pts[:, :-1, :, :], dim=-1)
        edge_v = torch.norm(grid_pts[:, :, 1:, :] - grid_pts[:, :, :-1, :], dim=-1)
        ideal_h_len, ideal_v_len = paper_scale_clamped / 8.0, paper_scale_clamped / 11.0
        
        # Standard Variance
        len_var = (((edge_h - edge_h.mean(dim=(1, 2), keepdim=True))**2).mean() + 
                   ((edge_v - edge_v.mean(dim=(1, 2), keepdim=True))**2).mean()) / (paper_scale_clamped.mean()**2 + 1e-6)
        
        # Expansion Force (Push)
        push_loss = ((F.relu(0.6 * ideal_h_len.view(B, 1, 1) - edge_h)**2).mean() + 
                     (F.relu(0.6 * ideal_v_len.view(B, 1, 1) - edge_v)**2).mean()) / (paper_scale_clamped.mean()**2 + 1e-6)

        # c. Laplacian Smoothness (Anti-Jitter)
        # Actively push internal points toward the average of their 4 orthogonal neighbors
        internal_pts = grid_pts[:, 1:-1, 1:-1, :]
        neighbors_avg = (grid_pts[:, :-2, 1:-1, :] + 
                         grid_pts[:, 2:, 1:-1, :] + 
                         grid_pts[:, 1:-1, :-2, :] + 
                         grid_pts[:, 1:-1, 2:, :]) * 0.25
        laplacian_loss = F.mse_loss(internal_pts, neighbors_avg, reduction='none').mean(dim=-1).mean()
        # Scale properly with paper dimensions
        laplacian_loss = laplacian_loss / (paper_scale_clamped.mean()**2 + 1e-6)

        # Consolidate Structural forces
        struct_loss = fold_penalty + push_loss + 0.1 * len_var + 0.5 * laplacian_loss

        # Final Objective
        total = (self.pos_w * pos_loss + 
                 self.sil_w * sil_loss + 
                 self.struct_w * struct_loss)
        
        return total, pos_loss, sil_loss, struct_loss, fold_penalty, push_loss, laplacian_loss

# ──────────────────────────────────────────────────────────────────────
#  VISUALIZATION — Renders to PNG each epoch (no frozen windows!)
# ──────────────────────────────────────────────────────────────────────
VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_viz")
os.makedirs(VIS_DIR, exist_ok=True)

class DashboardRenderer:
    """Sci-fi training dashboard with glowing neon aesthetics."""

    # ── Color Palette ──
    BG_DARK = '#0a0e1a'
    BG_PANEL = '#0f1629'
    BG_CARD = '#141d33'
    CYAN = '#00f0ff'
    MAGENTA = '#ff00d4'
    LIME = '#39ff14'
    AMBER = '#ffb300'
    CORAL = '#ff6b6b'
    GHOST = '#4a5568'
    TEXT = '#c9d1d9'
    TEXT_DIM = '#6b7280'

    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'xy_err': [], 'pos_loss': [], 'struct_loss': [],
            'sil_loss': [],
            'lr_backbone': [], 'lr_head': [],
            'best_nme': [], 'per_point_err': [],
            'fold_loss': [],
            'push_loss': [],
        }
        self.best_nme_val = float('inf')
        self.best_epoch = 0
        self.start_time = time.time()
        self.epoch_times = []
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _denorm(self, img_tensor):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    def _style_ax(self, ax, title='', xlabel='', ylabel=''):
        """Apply sci-fi styling to an axes."""
        ax.set_facecolor(self.BG_PANEL)
        if title:
            ax.set_title(title, color=self.CYAN, fontsize=11, fontweight='bold', pad=8,
                         fontfamily='monospace')
        if xlabel:
            ax.set_xlabel(xlabel, color=self.TEXT_DIM, fontsize=8, fontfamily='monospace')
        if ylabel:
            ax.set_ylabel(ylabel, color=self.TEXT_DIM, fontsize=8, fontfamily='monospace')
        ax.tick_params(colors=self.TEXT_DIM, labelsize=7)
        ax.grid(True, alpha=0.08, color=self.CYAN, linestyle='-')
        for spine in ax.spines.values():
            spine.set_color(self.CYAN)
            spine.set_alpha(0.2)

    def _glow_line(self, ax, x, y, color, label='', lw=1.5, alpha=1.0, ls='-'):
        """Draw a line with a soft glow effect behind it."""
        ax.plot(x, y, color=color, linewidth=lw + 3, alpha=0.08 * alpha, linestyle=ls)
        ax.plot(x, y, color=color, linewidth=lw + 1.5, alpha=0.15 * alpha, linestyle=ls)
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label, linestyle=ls)

    def _draw_mesh_on_ax(self, ax, img, gt_xy, pred_xy, title=None):
        """Draw GT (cyan) and pred (magenta) mesh overlay with sci-fi styling."""
        ax.imshow(img)
        gt_g = gt_xy.reshape(GRID_W, GRID_H, 2)
        pr_g = pred_xy.reshape(GRID_W, GRID_H, 2)

        # Inner grid lines — subtle scan lines
        for i in range(1, GRID_W - 1):
            ax.plot(gt_g[i, :, 0], gt_g[i, :, 1], color=self.CYAN, alpha=0.15, linewidth=0.6)
            ax.plot(pr_g[i, :, 0], pr_g[i, :, 1], color=self.MAGENTA, alpha=0.15, linewidth=0.6)
        for j in range(1, GRID_H - 1):
            ax.plot(gt_g[:, j, 0], gt_g[:, j, 1], color=self.CYAN, alpha=0.15, linewidth=0.6)
            ax.plot(pr_g[:, j, 0], pr_g[:, j, 1], color=self.MAGENTA, alpha=0.15, linewidth=0.6)

        # Glowing outer edges
        edge_colors = [self.CYAN, self.MAGENTA, self.AMBER, self.LIME]
        edges_gt = [gt_g[:, 0], gt_g[GRID_W-1, :], gt_g[:, GRID_H-1], gt_g[0, :]]
        edges_pr = [pr_g[:, 0], pr_g[GRID_W-1, :], pr_g[:, GRID_H-1], pr_g[0, :]]
        for gt_e, pr_e, c in zip(edges_gt, edges_pr, edge_colors):
            ax.plot(gt_e[:, 0], gt_e[:, 1], color=c, linewidth=3, alpha=0.15)  # glow
            ax.plot(gt_e[:, 0], gt_e[:, 1], color=c, linewidth=1.5, alpha=0.9)
            ax.plot(pr_e[:, 0], pr_e[:, 1], color=c, linewidth=1.5, alpha=0.7, linestyle='--')

        # Error whiskers — fade by distance
        errors = np.sqrt(((pred_xy - gt_xy) ** 2).sum(axis=-1))
        max_err = errors.max() if errors.max() > 0 else 1
        for k, (g, p) in enumerate(zip(gt_xy, pred_xy)):
            intensity = min(errors[k] / max_err, 1.0)
            c = (1.0, intensity * 0.3, intensity * 0.3)  # fade from white to red
            ax.plot([g[0], p[0]], [g[1], p[1]], color=c, alpha=0.5, linewidth=0.6)

        # GT points — cyan glow
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], c=self.CYAN, s=8, zorder=5, edgecolors='none', alpha=0.4)
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], c=self.CYAN, s=4, zorder=6, edgecolors='none')
        # Pred points — magenta glow
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c=self.MAGENTA, s=8, zorder=5, edgecolors='none', alpha=0.4)
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c=self.MAGENTA, s=4, zorder=6, edgecolors='none')

        # Corner highlights with pulsing rings
        corner_colors = [self.LIME, self.AMBER, self.CORAL, self.CYAN]
        for ci, idx in enumerate([0, 96, 107, 11]):
            cc = corner_colors[ci]
            ax.scatter(*gt_xy[idx], c=cc, s=60, zorder=7, edgecolors='none', alpha=0.15)
            ax.scatter(*gt_xy[idx], c=cc, s=25, zorder=8, edgecolors='black', linewidths=0.5)
            ax.scatter(*pred_xy[idx], c=self.MAGENTA, s=25, zorder=8, edgecolors='white', linewidths=0.3)

        # NME badge
        diag = (np.linalg.norm(gt_xy[0] - gt_xy[107]) + np.linalg.norm(gt_xy[96] - gt_xy[11])) / 2
        err_px = np.sqrt(((pred_xy - gt_xy) ** 2).sum(axis=-1)).mean()
        nme = err_px / diag * 100 if diag > 0 else 0
        badge_color = self.LIME if nme < 5 else (self.AMBER if nme < 10 else self.CORAL)
        label = f"▸ NME {nme:.1f}%  ·  {err_px:.1f}px"
        ax.text(5, IMG_SIZE - 8, label, color=badge_color, fontsize=7, fontfamily='monospace',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.85,
                                              edgecolor=badge_color, linewidth=0.5))

        if title:
            ax.set_title(title, color=self.CYAN, fontsize=10, fontweight='bold', fontfamily='monospace')
        ax.axis('off')
        ax.set_facecolor(self.BG_DARK)

    def _draw_speedometer(self, ax, nme_pct, target=2.0):
        """Draw a sci-fi NME arc gauge."""
        ax.set_facecolor(self.BG_PANEL)
        ax.set_aspect('equal')

        # Arc background
        theta = np.linspace(np.pi, 0, 100)
        r = 1.0
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=self.GHOST, linewidth=6, alpha=0.3)

        # Colored arc segments (green → yellow → red)
        segments = [(0, 5, self.LIME), (5, 10, self.AMBER), (10, 40, self.CORAL)]
        for lo, hi, c in segments:
            frac_lo = max(0, min(1, (np.pi - lo / 40 * np.pi)))
            frac_hi = max(0, min(1, (np.pi - hi / 40 * np.pi)))
            seg_theta = np.linspace(frac_lo, frac_hi, 50)
            ax.plot(r * np.cos(seg_theta), r * np.sin(seg_theta), color=c, linewidth=8, alpha=0.4)

        # Needle
        clamped = min(nme_pct, 40)
        needle_angle = np.pi - (clamped / 40) * np.pi
        nx, ny = 0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle)
        needle_color = self.LIME if nme_pct < 5 else (self.AMBER if nme_pct < 10 else self.CORAL)
        ax.plot([0, nx], [0, ny], color=needle_color, linewidth=2.5, alpha=0.9)
        ax.plot([0, nx], [0, ny], color=needle_color, linewidth=5, alpha=0.15)  # glow
        ax.scatter(0, 0, c='white', s=30, zorder=10)

        # Value text
        ax.text(0, -0.15, f'{nme_pct:.2f}%', ha='center', va='center', color=needle_color,
                fontsize=18, fontweight='bold', fontfamily='monospace')
        ax.text(0, -0.35, 'NME', ha='center', va='center', color=self.TEXT_DIM,
                fontsize=9, fontfamily='monospace')

        # Scale labels
        for val in [0, 5, 10, 20, 40]:
            a = np.pi - (val / 40) * np.pi
            tx, ty = 1.15 * np.cos(a), 1.15 * np.sin(a)
            ax.text(tx, ty, str(val), ha='center', va='center', color=self.TEXT_DIM,
                    fontsize=6, fontfamily='monospace')

        # Target line
        ta = np.pi - (target / 40) * np.pi
        tx1, ty1 = 0.75 * np.cos(ta), 0.75 * np.sin(ta)
        tx2, ty2 = 1.0 * np.cos(ta), 1.0 * np.sin(ta)
        ax.plot([tx1, tx2], [ty1, ty2], color=self.LIME, linewidth=2, alpha=0.6, linestyle='--')

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.5, 1.3)
        ax.axis('off')
        ax.set_title('⚡ CONVERGENCE GAUGE', color=self.CYAN, fontsize=9,
                      fontweight='bold', fontfamily='monospace', pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_error_list(self, ax, stagnant_results):
        """Draw a text-based list of samples that are improving the slowest (stagnant)."""
        ax.set_facecolor(self.BG_PANEL)
        ax.axis('off')
        
        if not stagnant_results:
            ax.set_title('◈ SLOWEST IMPROVING SAMPLES', color=self.CORAL, fontsize=9,
                         fontweight='bold', fontfamily='monospace', pad=10)
            ax.text(0.5, 0.5, 'AWAITING DATA', color=self.GHOST, ha='center', va='center',
                    fontfamily='monospace', fontsize=10, transform=ax.transAxes)
            return

        # Sort by improvement ASCENDING (lowest improvement first).
        # We only consider samples with NME > 1% to avoid noise from perfectly converged samples.
        filtered = [x for x in stagnant_results if x[1] > 0.01]
        sorted_results = sorted(filtered, key=lambda x: x[2])[:10]
        
        ax.set_title('◈ SLOWEST IMPROVING (BY IDX)', color=self.CORAL, fontsize=9,
                     fontweight='bold', fontfamily='monospace', pad=10)
        
        y_pos = 0.88
        for i, (idx, nme, imp, dev, filename) in enumerate(sorted_results):
            # Improvement color & arrow
            imp_pct = imp * 100
            if imp_pct < 0: 
                imp_str = f"↑ {abs(imp_pct):4.2f}%" # Error Increasing
                imp_color = self.MAGENTA
            elif imp_pct < 0.1:
                imp_str = f"⁃ {imp_pct:4.2f}%" # Stagnant
                imp_color = self.CORAL
            else:
                imp_str = f"↓ {imp_pct:4.2f}%" # Improving slowly
                imp_color = self.LIME

            # Show original folder index and last 6 of filename
            fname = filename.split('.')[0][-6:] if '.' in filename else filename[-6:]
            ax.text(0.05, y_pos, f"{i+1:02d}. {fname} (ID {idx:04d})", color=self.TEXT, 
                    fontsize=8, fontfamily='monospace', transform=ax.transAxes)
            
            # Show current NME
            ax.text(0.65, y_pos, f"{nme*100:4.1f}%", color=self.TEXT_DIM, 
                    fontsize=7, fontfamily='monospace', transform=ax.transAxes, ha='right')
            
            # Show Improvement Delta
            ax.text(0.95, y_pos, imp_str, color=imp_color, 
                    fontsize=8, fontfamily='monospace', fontweight='bold', 
                    transform=ax.transAxes, ha='right')
            y_pos -= 0.082

    def _draw_error_deviation(self, ax, stagnant_results):
        """Draw a horizontal bar chart showing error deviation from mean NME."""
        ax.set_facecolor(self.BG_PANEL)
        
        if not stagnant_results:
            self._style_ax(ax, title='◈ ERROR DEVIATION (Δ MEAN)')
            ax.text(0.5, 0.5, 'AWAITING DATA', color=self.GHOST, ha='center', va='center',
                    fontfamily='monospace', fontsize=10, transform=ax.transAxes)
            return

        # Use the same top 10 as the list
        filtered = [x for x in stagnant_results if x[1] > 0.01]
        sorted_results = sorted(filtered, key=lambda x: x[2])[:10]
        
        indices = []
        for x in sorted_results:
            fname = x[4].split('.')[0][-6:] if '.' in x[4] else x[4][-6:]
            indices.append(f"{fname} (ID {x[0]:04d})")
        indices = indices[::-1] # Reverse for bottom-up plot
        deviations = [x[3] * 100 for x in sorted_results][::-1]
        
        y_pos = np.arange(len(indices))
        colors = [self.CORAL if d > 0 else self.LIME for d in deviations]
        
        bars = ax.barh(y_pos, deviations, color=colors, alpha=0.6, height=0.6, edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='white', linewidth=0.8, alpha=0.5, linestyle='--')
        
        # Style
        ax.set_yticks(y_pos)
        ax.set_yticklabels(indices, color=self.TEXT, fontsize=7, fontfamily='monospace')
        ax.tick_params(colors=self.TEXT_DIM, labelsize=7)
        ax.set_xlabel('Δ NME % FROM MEAN', color=self.TEXT_DIM, fontsize=7, fontfamily='monospace')
        ax.set_title('◈ ERROR DEVIATION (Δ MEAN)', color=self.CYAN, fontsize=9, 
                     fontweight='bold', fontfamily='monospace', pad=10)
        
        # Grid
        ax.grid(axis='x', color=self.TEXT_DIM, linestyle=':', alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color(self.CYAN); spine.set_alpha(0.2)
        
        # Annotate bars
        for i, d in enumerate(deviations):
            align = 'left' if d > 0 else 'right'
            offset = 0.5 if d > 0 else -0.5
            ax.text(d + offset, i, f"{'+' if d > 0 else ''}{d:.1f}%", 
                    va='center', ha=align, color=colors[i], fontsize=7, fontweight='bold', fontfamily='monospace')

    def _draw_boundary_alignment(self, ax, img, gt_bnd, pred_xy, title=None, loss_val=None):
        """Visualize how well the predicted perimeter aligns with the ground truth boundary ribbon."""
        ax.imshow(img, alpha=0.7)
        
        # Boundary Ribbon Overlay (Neon Magenta)
        bnd_mask = gt_bnd.squeeze().numpy()
        h, w = bnd_mask.shape
        overlay = np.zeros((h, w, 4))
        overlay[bnd_mask > 0.5] = [1.0, 0.0, 0.8, 0.4] # Magenta
        
        # Upsample overlay to IMG_SIZE if needed
        if h != IMG_SIZE:
            overlay = cv2.resize(overlay, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        ax.imshow(overlay)

        # Draw predicted perimeter
        # Draw predicted perimeter
        # Perimeter indices (Col-Major 9x12): Top (0..96) -> Right (97..107) -> Bottom (95..11) -> Left (10..1)
        perimeter_idx = [0, 12, 24, 36, 48, 60, 72, 84, 96, 
                         97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                         95, 83, 71, 59, 47, 35, 23, 11,
                         10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        peri_pts = pred_xy[perimeter_idx]
        peri_pts_closed = np.concatenate([peri_pts, peri_pts[:1]], axis=0)
        
        # Glow for predicted boundary
        ax.plot(peri_pts_closed[:, 0], peri_pts_closed[:, 1], color=self.LIME, linewidth=4, alpha=0.15)
        ax.plot(peri_pts_closed[:, 0], peri_pts_closed[:, 1], color=self.LIME, linewidth=1.5, alpha=0.9)

        if loss_val is not None:
            # Note: loss_val here is the scaled version (x1000) for visibility
            label = f"▸ BND LOSS: {loss_val:.4f}"
            ax.text(0.05, 0.05, label, color=self.MAGENTA, fontsize=8, fontfamily='monospace',
                    fontweight='bold', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7,
                              edgecolor=self.MAGENTA, linewidth=0.5))

        if title:
            ax.set_title(title, color=self.CYAN, fontsize=9, fontweight='bold', fontfamily='monospace')
        ax.axis('off')

    def update(self, epoch, vis_samples, per_point_errors, per_point_vectors, per_point_confs,
               train_loss, val_loss, xy_err, pos_loss, struct_loss, sil_loss,
               fold_loss, push_loss,
               lr_backbone, lr_head):
        epoch_start = time.time()
        
        # Track best
        if xy_err < self.best_nme_val:
            self.best_nme_val = xy_err
            self.best_epoch = epoch

        # Record history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['xy_err'].append(xy_err)
        self.history['pos_loss'].append(pos_loss)
        self.history['struct_loss'].append(struct_loss)
        self.history['sil_loss'].append(sil_loss)
        self.history['fold_loss'].append(fold_loss)
        self.history['push_loss'].append(push_loss)
        self.history['lr_backbone'].append(lr_backbone)
        self.history['lr_head'].append(lr_head)
        self.history['per_point_err'].append(per_point_errors.copy())

        n_samples = len(vis_samples)
        nme_pct = xy_err * 100
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)

        # ETA calculation
        if len(self.epoch_times) > 1:
            avg_epoch_time = elapsed / epoch
            remaining_epochs = EPOCHS - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
        else:
            eta_str = "calculating..."

        elapsed_str = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"

        # Improvement delta
        if len(self.history['xy_err']) > 1:
            delta = (self.history['xy_err'][-2] - self.history['xy_err'][-1]) * 100
            delta_str = f"{'▲' if delta > 0 else '▼'} {abs(delta):.3f}%"
            delta_color = self.LIME if delta > 0 else self.CORAL
        else:
            delta_str = "—"
            delta_color = self.TEXT_DIM

        # ── Figure: 3 rows × 4 cols, ultra-wide ──
        fig = plt.figure(figsize=(24, 16), facecolor=self.BG_DARK)

        # ════════════════════════════════════════════════════════════════
        #  HEADER BAR (manual text placement)
        # ════════════════════════════════════════════════════════════════
        header_y = 0.975
        fig.text(0.02, header_y, '◈ MESH TRAINING COMMAND CENTER', color=self.CYAN,
                 fontsize=16, fontweight='bold', fontfamily='monospace', va='top')
        fig.text(0.02, header_y - 0.025, f'HYBRID Real/Synth · HRNet-W32 · 9x12 Col-Major · {NUM_POINTS} Points',
                 color=self.TEXT_DIM, fontsize=8, fontfamily='monospace', va='top')

        # Status indicators
        status_color = self.LIME if nme_pct < 10 else (self.AMBER if nme_pct < 20 else self.CORAL)
        fig.text(0.50, header_y, f'EPOCH {epoch:03d}/{EPOCHS}', color=self.AMBER,
                 fontsize=14, fontweight='bold', fontfamily='monospace', va='top', ha='center')
        fig.text(0.50, header_y - 0.025, f'◷ {elapsed_str}  ·  ETA {eta_str}  ·  Δ {delta_str}',
                 color=delta_color, fontsize=8, fontfamily='monospace', va='top', ha='center')

        fig.text(0.98, header_y, f'NME {nme_pct:.2f}%', color=status_color,
                 fontsize=16, fontweight='bold', fontfamily='monospace', va='top', ha='right')
        fig.text(0.98, header_y - 0.025,
                 f'★ BEST {self.best_nme_val * 100:.2f}% @ E{self.best_epoch:03d}  ·  '
                 f'T:{train_loss:.3f} V:{val_loss:.3f}',
                 color=self.TEXT_DIM, fontsize=8, fontfamily='monospace', va='top', ha='right')

        # Thin separator line
        line_ax = fig.add_axes([0.01, 0.935, 0.98, 0.002])
        line_ax.set_facecolor(self.CYAN)
        line_ax.set_alpha(0.3)
        line_ax.set_xticks([]); line_ax.set_yticks([])
        for s in line_ax.spines.values(): s.set_visible(False)

        # ════════════════════════════════════════════════════════════════
        #  ROW 1: 3 Val Samples + Speedometer (4 cols)
        # ════════════════════════════════════════════════════════════════
        for s_idx in range(3):
            ax = fig.add_subplot(3, 4, s_idx + 1)
            if s_idx < n_samples:
                sample = vis_samples[s_idx]
                img = self._denorm(sample['img'])
                gt_px = sample['gt_2d'].numpy() * IMG_SIZE
                pr_px = sample['pr_2d'].numpy() * IMG_SIZE
                
                fname = sample['filename']
                f_idx = sample['folder_idx']
                last6 = fname.split('.')[0][-6:] if '.' in fname else fname[-6:]
                title_str = f"Sample: {last6} (ID {f_idx:04d})"
                
                self._draw_mesh_on_ax(ax, img, gt_px, pr_px, title=title_str)
            else:
                ax.set_facecolor(self.BG_DARK)
                ax.axis('off')

        # Speedometer gauge
        ax_speed = fig.add_subplot(3, 4, 4)
        self._draw_speedometer(ax_speed, nme_pct)

        epochs_range = list(range(1, len(self.history['train_loss']) + 1))

        # ════════════════════════════════════════════════════════════════
        #  ROW 2: Loss Curves | NME Trend | Per-Point Heatmap | Histogram
        # ════════════════════════════════════════════════════════════════

        # ── Loss Curves with glowing lines ──
        ax_loss = fig.add_subplot(3, 4, 5)
        self._style_ax(ax_loss, title='◈ GLOBAL CONVERGENCE', xlabel='Epoch', ylabel='Loss')
        
        # Main thick lines for Total Train and Val
        self._glow_line(ax_loss, epochs_range, self.history['train_loss'], self.CYAN, 'Train Total', lw=2)
        self._glow_line(ax_loss, epochs_range, self.history['val_loss'], self.MAGENTA, 'Val Total', lw=2, ls='--')
        
        # Subtle "Ghost" lines for components (so they don't clutter)
        ax_loss.plot(epochs_range, self.history['pos_loss'], color=self.LIME, alpha=0.3, linewidth=0.8, label='pos')
        ax_loss.plot(epochs_range, self.history['struct_loss'], color=self.AMBER, alpha=0.3, linewidth=0.8, label='struct')
        ax_loss.plot(epochs_range, self.history['sil_loss'], color=self.GHOST, alpha=0.3, linewidth=0.8, label='sil')

        # Gradient fill under train curve
        ax_loss.fill_between(epochs_range, self.history['train_loss'], alpha=0.03, color=self.CYAN)
        ax_loss.legend(fontsize=6, facecolor=self.BG_CARD, edgecolor=self.CYAN,
                       labelcolor='white', loc='upper right', framealpha=0.8)

        # ── NME Trend with convergence zone ──
        ax_nme = fig.add_subplot(3, 4, 6)
        self._style_ax(ax_nme, title='◈ NME TRAJECTORY', xlabel='Epoch', ylabel='NME (%)')
        nme_hist = [e * 100 for e in self.history['xy_err']]
        self._glow_line(ax_nme, epochs_range, nme_hist, self.CYAN, 'NME (%)')
        ax_nme.fill_between(epochs_range, nme_hist, alpha=0.08, color=self.CYAN)

        # Target zone
        ax_nme.axhspan(0, 2.0, alpha=0.05, color=self.LIME)
        ax_nme.axhline(y=2.0, color=self.LIME, linestyle='--', alpha=0.4, linewidth=0.8)
        ax_nme.text(len(epochs_range), 2.0, ' TARGET', color=self.LIME, fontsize=6,
                    fontfamily='monospace', va='bottom', alpha=0.7)

        # Best marker with pulse ring
        best_idx = self.history['xy_err'].index(self.best_nme_val)
        ax_nme.scatter(best_idx + 1, self.best_nme_val * 100, c=self.CORAL, s=120, zorder=9,
                       edgecolors='none', alpha=0.15)  # outer glow
        ax_nme.scatter(best_idx + 1, self.best_nme_val * 100, c=self.CORAL, s=50, zorder=10,
                       edgecolors='white', linewidths=0.5, marker='*')
        ax_nme.annotate(f'{self.best_nme_val*100:.2f}%', (best_idx + 1, self.best_nme_val * 100),
                        textcoords='offset points', xytext=(8, 8), fontsize=7,
                        color=self.CORAL, fontweight='bold', fontfamily='monospace')

        # LR overlay
        ax_lr = ax_nme.twinx()
        ax_lr.plot(epochs_range, self.history['lr_head'], color=self.AMBER, linewidth=0.6, alpha=0.3)
        ax_lr.fill_between(epochs_range, self.history['lr_head'], alpha=0.03, color=self.AMBER)
        ax_lr.set_ylabel('LR', color=self.AMBER, fontsize=7, fontfamily='monospace')
        ax_lr.tick_params(colors=self.AMBER, labelsize=6)
        ax_lr.spines['right'].set_color(self.AMBER)
        ax_lr.spines['right'].set_alpha(0.3)

        # ── Per-Point Error Heatmap with glow ──
        ax_pp = fig.add_subplot(3, 4, 7)
        ax_pp.set_facecolor(self.BG_PANEL)
        if per_point_errors is not None and len(per_point_errors) == NUM_POINTS:
            err_grid = (per_point_errors * 100).reshape(GRID_W, GRID_H).T
            # Custom colormap: dark blue → cyan → yellow → red
            from matplotlib.colors import LinearSegmentedColormap
            sci_fi_cmap = LinearSegmentedColormap.from_list('scifi',
                [(0, '#0a0e2a'), (0.2, '#003366'), (0.4, '#00a88f'),
                 (0.6, '#ffe066'), (0.8, '#ff6b35'), (1.0, '#ff1744')])
            im = ax_pp.imshow(err_grid, cmap=sci_fi_cmap, interpolation='gaussian',
                              vmin=0, vmax=max(15, err_grid.max()))
            # Annotate cells with glow
            for r in range(GRID_H):
                for c in range(GRID_W):
                    val = err_grid[r, c]
                    tc = 'white' if val > err_grid.max() * 0.4 else self.CYAN
                    ax_pp.text(c, r, f'{val:.1f}', ha='center', va='center',
                              fontsize=8, fontweight='bold', color=tc, fontfamily='monospace')
            cbar = plt.colorbar(im, ax=ax_pp, shrink=0.7, pad=0.03)
            cbar.set_label('NME %', color=self.CYAN, fontsize=7, fontfamily='monospace')
            cbar.ax.tick_params(colors=self.TEXT_DIM, labelsize=6)
            cbar.outline.set_edgecolor(self.CYAN)
            cbar.outline.set_alpha(0.3)

            ax_pp.set_xticks(range(GRID_W))
            ax_pp.set_yticks(range(GRID_H))
            ax_pp.set_xticklabels([f'C{c}' for c in range(GRID_W)], color=self.TEXT_DIM, fontsize=6, fontfamily='monospace')
            ax_pp.set_yticklabels([f'R{r}' for r in range(GRID_H)], color=self.TEXT_DIM, fontsize=6, fontfamily='monospace')
            # Corner labels
            corners = {(0,0): 'TL', (0, GRID_W-1): 'TR', (GRID_H-1, GRID_W-1): 'BR', (GRID_H-1, 0): 'BL'}
            for (r,c), lbl in corners.items():
                ax_pp.text(c, r - 0.38, lbl, ha='center', va='center', fontsize=5,
                          color=self.AMBER, fontweight='bold', fontfamily='monospace')
        else:
            ax_pp.text(0.5, 0.5, 'AWAITING DATA', transform=ax_pp.transAxes,
                      ha='center', va='center', color=self.GHOST, fontfamily='monospace', fontsize=10)
        ax_pp.set_title('◈ POINT ERROR MATRIX', color=self.CYAN, fontsize=9,
                         fontweight='bold', fontfamily='monospace', pad=6)
        for spine in ax_pp.spines.values():
            spine.set_color(self.CYAN); spine.set_alpha(0.2)

        # ── Confidence Density Scan (Scatter + Trend) ──
        ax_conf = fig.add_subplot(3, 4, 8)
        self._style_ax(ax_conf, title='◈ CONFIDENCE DENSITY', xlabel='Point Index (0-107)', ylabel='H-Map Peak')
        if per_point_confs is not None:
            indices = np.arange(NUM_POINTS)
            # Scatter with glow
            ax_conf.scatter(indices, per_point_confs, c=self.AMBER, s=30, alpha=0.3, edgecolors='none')
            ax_conf.scatter(indices, per_point_confs, c='white', s=5, alpha=0.8)
            ax_conf.plot(indices, per_point_confs, color=self.AMBER, linewidth=1, alpha=0.5, linestyle='--')
            
            # Label corners for 9x12
            for i, lbl in zip([0, 11, 96, 107], ['TL', 'BL', 'TR', 'BR']):
                ax_conf.text(i, per_point_confs[i] + 0.05, lbl, color=self.AMBER, 
                            fontsize=6, ha='center', va='bottom', fontweight='bold')
        
        # ════════════════════════════════════════════════════════════════
        #  ROW 3: Spatial Error Flow | Loss Decomposition | LR Schedule | Stats
        # ════════════════════════════════════════════════════════════════
        

        # ── Top 10 Highest Errors (replaces duplicated speedometer) ──
        ax_err_list = fig.add_subplot(3, 4, 9)
        ax_err_list.set_facecolor(self.BG_PANEL)
        ax_err_list.axis('off')

        # ── Error Deviation from Mean (New Graph) ──
        ax_deviation = fig.add_subplot(3, 4, 10)
        ax_deviation.set_facecolor(self.BG_PANEL)
        ax_deviation.axis('off')

        # ── Boundary Alignment Scan ──
        ax_bnd = fig.add_subplot(3, 4, 11)
        if n_samples > 0:
            sample = vis_samples[0]
            img = self._denorm(sample['img'])
            self._draw_boundary_alignment(ax_bnd, img, sample['gt_bnd'], sample['pr_2d'].detach().numpy() * IMG_SIZE,
                                         title='◈ BOUNDARY ALIGNMENT', loss_val=sil_loss)
        else:
            self._style_ax(ax_bnd, title='◈ BOUNDARY ALIGNMENT')

        # ── Stats Panel (text-based telemetry readout) ──
        ax_stats = fig.add_subplot(3, 4, 12)
        ax_stats.set_facecolor(self.BG_PANEL)
        ax_stats.axis('off')
        for spine in ax_stats.spines.values():
            spine.set_color(self.CYAN); spine.set_alpha(0.2)

        # Build stats text
        improvement = (self.history['xy_err'][0] - self.history['xy_err'][-1]) / self.history['xy_err'][0] * 100 if self.history['xy_err'][0] > 0 else 0
        stats_lines = [
            ('SYSTEM STATUS', '', self.CYAN, 12),
            ('─' * 28, '', self.GHOST, 8),
            ('▸ EPOCH', f'{epoch:03d} / {EPOCHS}', self.AMBER, 9),
            ('▸ TOTAL LOSS', f'{train_loss:.4f}', self.CYAN, 9),
            ('▸ NME', f'{nme_pct:.2f}%', status_color, 9),
            ('▸ BEST NME', f'{self.best_nme_val * 100:.2f}% @ E{self.best_epoch:03d}', self.LIME, 9),
            ('▸ IMPROVEMENT', f'{improvement:.1f}% from start', self.LIME if improvement > 0 else self.CORAL, 9),
            ('─' * 28, '', self.GHOST, 8),
            ('▸ LR BACKBONE', f'{lr_backbone:.2e}', self.AMBER, 9),
            ('▸ LR HEAD', f'{lr_head:.2e}', self.AMBER, 9),
            ('▸ ELAPSED', elapsed_str, self.TEXT, 9),
            ('▸ ETA', eta_str, self.TEXT, 9),
            ('─' * 28, '', self.GHOST, 8),
            ('▸ POSITIONAL', f'{pos_loss:.4f}', self.LIME, 9),
            ('▸ STRUCTURAL', f'{struct_loss:.4f}', self.AMBER, 9),
            ('▸ SILHOUETTE', f'{sil_loss:.4f}', self.MAGENTA, 9),
        ]
        y_pos = 0.95
        for label, value, color, fs in stats_lines:
            if value:
                ax_stats.text(0.05, y_pos, label, color=self.TEXT_DIM, fontsize=fs - 1,
                             fontfamily='monospace', transform=ax_stats.transAxes, va='top')
                ax_stats.text(0.95, y_pos, value, color=color, fontsize=fs,
                             fontfamily='monospace', fontweight='bold',
                             transform=ax_stats.transAxes, va='top', ha='right')
            else:
                ax_stats.text(0.5, y_pos, label, color=color, fontsize=fs,
                             fontfamily='monospace', fontweight='bold',
                             transform=ax_stats.transAxes, va='top', ha='center')
            y_pos -= 0.065

        ax_stats.set_title('◈ TELEMETRY', color=self.CYAN, fontsize=9,
                           fontweight='bold', fontfamily='monospace', pad=6)

        # ── Save ──
        plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.5, w_pad=1.0)
        fig.savefig(os.path.join(VIS_DIR, "training_dashboard.png"), dpi=150, facecolor=fig.get_facecolor())
        fig.savefig(os.path.join(VIS_DIR, f"epoch_{epoch:03d}.png"), dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────
def train():
    print(f"🚀 Mesh Training v6 — Hybrid Real+HD Training | Device: {DEVICE}")
    print(f"   Model: HRNet-W32 + {HEATMAP_SIZE}px Heatmap (DSNT)")
    print(f"   📊 Dashboard: {VIS_DIR}/training_dashboard.png (updates every epoch)")

    # --- 1. Gather Real Samples (Using ALL 749 samples) ---
    real_samples = []
    if os.path.exists(IMAGE_DIR) and os.path.exists(MESH_DIR):
        files = sorted([f for f in os.listdir(MESH_DIR) if f.endswith('.json')])
        real_samples = [(IMAGE_DIR, MESH_DIR, f, i) for i, f in enumerate(files)]
        print(f"   Using {len(real_samples)} REAL samples (HD 768).")
    else:
        print("   ❌ ERROR: Real data directories not found!")
        return

    # --- 2. Gather Synthetic Samples ---
    synth_samples = []
    if os.path.exists(SYNTH_IMAGE_DIR) and os.path.exists(SYNTH_MESH_DIR):
        files = sorted([f for f in os.listdir(SYNTH_MESH_DIR) if f.endswith('.json')])
        synth_samples = [(SYNTH_IMAGE_DIR, SYNTH_MESH_DIR, f, i) for i, f in enumerate(files)]
        print(f"   Using {len(synth_samples)} SYNTHETIC samples (HD 768).")
    else:
        print("   ⚠️ WARNING: Synthetic data directories not found!")

    # --- 3. Split & Combine (Randomized for better representation) ---
    import random
    random.seed(42) # Fixed seed for stable val set WITHIN a run
    all_indices = list(range(len(real_samples)))
    random.shuffle(all_indices)
    
    n_val = int(len(real_samples) * 0.1)
    val_indices = set(all_indices[:n_val])
    
    r_train = [real_samples[i] for i in all_indices[n_val:]]
    r_val = [real_samples[i] for i in all_indices[:n_val]]
    # Hybrid Mix: Combine real training data with the golden synthetic samples
    all_train_samples = r_train + synth_samples
    all_val_samples = r_val

    # Normalize/Mix — 50/50 Real/Synth Training Pool
    train_ds = MeshDataset(all_train_samples, transform=get_train_transforms(), is_train=True)
    val_ds   = MeshDataset(all_val_samples, transform=get_val_transforms(), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    print(f"   ✅ Hybrid Training Active: {len(all_train_samples)} total training samples")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"   🚀 DATA POOL: Train: {len(all_train_samples)} | Val: {len(all_val_samples)}")

    model = HeatmapMeshRegressor(num_points=NUM_POINTS).to(DEVICE)

    if not FRESH_START:
        checkpoint_path = "backup_best_9x12_768.pth"
        if os.path.exists(checkpoint_path):
            print(f"   Loading {checkpoint_path} (Resuming 9x12 training)...")
            try:
                state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state, strict=False)
                print(f"   ✅ Weights loaded from {checkpoint_path}!")
            except Exception as e:
                print(f"   Could not load fallback weights: {e}")
        else:
            print("   Starting from FRESH weights (No checkpoint found).")
    else:
        print("   ⚡ FRESH START ENABLED: Starting with random weights (ignoring checkpoints).")
    dashboard = DashboardRenderer()
    sample_stats = {} # Persistent history for each real sample filename

    torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {total_params:,}")

    criterion = MeshLoss2D(
        pos_weight=10.0,      # MAIN ACCURACY SIGNAL
        struct_weight=5.0,    # GUARDRAIL: Prevents criss-crossing/tangling
        sil_weight=0.0        # Silhouette remains disabled
    )
    
    # Verify multipliers (DEBUG PRINT)
    print(f"   Multipliers: Edge={criterion.edge_mult}, Corner={criterion.corner_mult}")

    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': MAX_LR_BACKBONE},
        {'params': model.heatmap_head.parameters(), 'lr': MAX_LR_HEAD}
    ], weight_decay=1e-4)

    # Standard ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    scaler = torch.amp.GradScaler('cuda')
    best_val_nme = float('inf')

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss_sum = 0.0
        loop = tqdm(train_loader, total=min(len(train_loader), STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}")

        optimizer.zero_grad()
        for i, (images, gt_heatmaps, gt_coords, gt_mask, gt_boundary, meta) in enumerate(loop):
            images = images.to(DEVICE)
            gt_heatmaps = gt_heatmaps.to(DEVICE)
            gt_coords = gt_coords.to(DEVICE)
            gt_mask = gt_mask.to(DEVICE)
            gt_boundary = gt_boundary.to(DEVICE)
            
            # Tier 3 (Poisoned) samples get weight 0.0 to exclude them from training
            p_weights = (meta['poison_level'] < 3).float().to(DEVICE)
 
            with torch.amp.autocast('cuda'):
                pred_heatmaps, pred_coords = model(images)
                loss, pos_loss, sil_loss, struct_loss, fold_loss, push_loss, lapl_loss = criterion(
                    pred_heatmaps, gt_heatmaps, pred_coords, gt_coords, 
                    gt_mask, gt_boundary, poison_weights=p_weights
                )

            scaler.scale(loss / ACCUMULATION_STEPS).backward()

            if ((i + 1) % ACCUMULATION_STEPS == 0) or (i + 1 == min(len(train_loader), STEPS_PER_EPOCH)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                
                optimizer.zero_grad()

            train_loss_sum += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", str=f"{struct_loss.item():.4f}", 
                            psh=f"{push_loss.item():.4f}", lap=f"{lapl_loss.item():.4f}")

            if i >= STEPS_PER_EPOCH - 1:
                break

        n = min(len(train_loader), STEPS_PER_EPOCH)
        avg_train = train_loss_sum / n

        # --- VALIDATE ---
        torch.cuda.empty_cache()  # Free VRAM before validation
        model.eval()
        val_loss_sum = 0.0
        val_xy_errors = []
        val_pos_sum, val_struct_sum, val_sil_sum = 0.0, 0.0, 0.0
        val_fold_sum, val_push_sum = 0.0, 0.0

        # Collect up to 3 random vis samples from the entire validation set
        num_vis = min(3, len(val_loader))
        # Ensure we pick different batches every epoch to see new images
        vis_batch_indices = sorted(np.random.choice(len(val_loader), num_vis, replace=False))
        vis_samples = []

        # Per-point error accumulator (108 points)
        per_point_nme_sum = np.zeros(NUM_POINTS)
        per_point_vector_sum = np.zeros((NUM_POINTS, 2))
        per_point_conf_sum = np.zeros(NUM_POINTS)
        per_batch_errors = {} # Track error per filename this epoch
        per_point_count = 0

        with torch.no_grad():
            for i, (images, gt_heatmaps, gt_coords, gt_mask, gt_boundary, meta) in enumerate(tqdm(val_loader, desc="Evaluating", leave=False)):
                images = images.to(DEVICE)
                gt_heatmaps = gt_heatmaps.to(DEVICE)
                gt_coords = gt_coords.to(DEVICE)
                gt_mask = gt_mask.to(DEVICE)
                gt_boundary = gt_boundary.to(DEVICE)

                with torch.amp.autocast('cuda'):
                    pred_heatmaps, pred_coords = model(images)
                    loss, posl, sill, strl, fldl, pshl, lapll = criterion(pred_heatmaps, gt_heatmaps, pred_coords, gt_coords, gt_mask, gt_boundary)
                
                filenames = meta['filename']
                val_loss_sum += loss.item()
                val_pos_sum += posl.item()
                val_sil_sum += sill.item()
                val_struct_sum += strl.item()
                val_fold_sum += fldl.item()
                val_push_sum += pshl.item()

                pred_px = pred_coords.cpu().numpy() * IMG_SIZE
                gt_px = gt_coords.cpu().numpy() * IMG_SIZE
                
                # Scale-invariant error (Col-Major 9x12 Corners: 0, 96, 107, 11)
                ptr_tl, ptr_tr, ptr_br, ptr_bl = gt_px[:, 0, :], gt_px[:, 96, :], gt_px[:, 107, :], gt_px[:, 11, :]
                diag1 = np.linalg.norm(ptr_tl - ptr_br, axis=-1)
                diag2 = np.linalg.norm(ptr_tr - ptr_bl, axis=-1)
                scale = (diag1 + diag2) / 2.0
                
                errors = np.sqrt(((pred_px - gt_px) ** 2).sum(axis=-1))  # (B, 108)
                pct_errors = errors / scale[:, None]  # (B, 108)
                val_xy_errors.extend(pct_errors.mean(axis=1).tolist())

                # Track per-sample errors and coordinates for validated auto-healing
                for b_idx in range(len(meta['index'])):
                    s_idx = meta['index'][b_idx].item()
                    f_idx = meta['folder_idx'][b_idx].item()
                    nme_val = pct_errors[b_idx].mean()
                    # Per-batch errors disabled for stability

                # Per-point errors & vectors
                per_point_nme_sum += pct_errors.sum(axis=0)
                per_point_vector_sum += (pred_px - gt_px).sum(axis=0)
                
                # Confidence peaks
                confs = pred_heatmaps.view(pred_heatmaps.size(0), NUM_POINTS, -1).max(dim=-1)[0]
                per_point_conf_sum += confs.detach().cpu().numpy().sum(axis=0)
                
                per_point_count += len(images)

                # Collect visualization samples (ensure we use meta correctly)
                for b_idx in range(len(images)):
                    if (i * BATCH_SIZE + b_idx) in vis_batch_indices: # Approximation for simplicity
                        vis_samples.append({
                            'img': images[b_idx].cpu(),
                            'gt_2d': gt_coords[b_idx].cpu(),
                            'pr_2d': pred_coords[b_idx].cpu(),
                            'gt_mask': gt_mask[b_idx].cpu(),
                            'gt_bnd': gt_boundary[b_idx].cpu(),
                            'filename': filenames[b_idx],
                            'folder_idx': meta['folder_idx'][b_idx].item()
                        })
                        if len(vis_samples) >= 3: break # Limit to 3

        vn = len(val_loader)
        avg_val = val_loss_sum / vn
        avg_pct = np.mean(val_xy_errors)
        
        # Remove the CPU offload for EMA module since it stays on GPU now
        torch.cuda.empty_cache()
        per_point_nme = per_point_nme_sum / max(per_point_count, 1)
        per_point_vectors = per_point_vector_sum / max(per_point_count, 1)
        per_point_confs = per_point_conf_sum / max(per_point_count, 1)
        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']

        print(f"  Train: {avg_train:.4f} | Val: {avg_val:.4f} | NME: {avg_pct * 100:.2f}% | LR: bb={lr_bb:.2e} hd={lr_hd:.2e}")

        # Ensure we step the scheduler passing validation metric
        scheduler.step(avg_pct)

        dashboard.update(
            epoch=epoch + 1,
            vis_samples=vis_samples,
            per_point_errors=per_point_nme,
            per_point_vectors=per_point_vectors,
            per_point_confs=per_point_confs,
            train_loss=avg_train,
            val_loss=avg_val,
            xy_err=avg_pct,
            pos_loss=val_pos_sum / vn,
            struct_loss=val_struct_sum / vn,
            sil_loss=val_sil_sum / vn,
            fold_loss=val_fold_sum / vn,
            push_loss=val_push_sum / vn,
            lr_backbone=lr_bb,
            lr_head=lr_hd
        )
        
        if avg_pct < best_val_nme:
            best_val_nme = avg_pct
            torch.save(model.state_dict(), "best_9x12_768.pth")
            try:
                import shutil
                shutil.copy("best_9x12_768.pth", "backup_best_9x12_768.pth")
            except Exception as e:
                print(f"   Could not create backup checkpoint: {e}")
            print(f"  ✅ New best 9x12_768 model! (val: {avg_val:.4f}, NME: {avg_pct * 100:.2f}%)")

    print(f"\n🏁 Done! Best NME: {best_val_nme * 100:.2f}%")
    print(f"   Dashboard history: {VIS_DIR}/")



if __name__ == "__main__":
    train()

