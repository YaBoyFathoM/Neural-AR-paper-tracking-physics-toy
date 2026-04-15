import os
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F

# --- CONFIGURATION ---
IMAGE_DIR = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\combined_768\images"
LABEL_DIR = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\combined_768\labels"
MESH_DIR  = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\data\combined_768\labels" 
CANDIDATES_PATH = None

# --- DATA STANDARDS ---
# Corner Order: [TL, TR, BR, BL] (Clockwise)
# Colors: 1:Green, 2:Blue, 3:Red, 4:Yellow
# Orientation: Labels are stored against RAW image pixels (NO EXIF transpose).
# Always load images with PIL.Image.open() without transposing for label alignment.

# Grid Configuration
GRID_W = 9  # columns (Horizontal)
GRID_H = 12 # rows (Vertical)
GRID_COUNT = GRID_W * GRID_H
MODEL_PATH = "best_9x12_768.pth"
IMG_SIZE = 768
HEATMAP_SIZE = 384

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Mesh Dragger UI: Using device {DEVICE}")

# Standardizing to linear proportions for faithful texture mapping
GRID_X_PROPS = np.linspace(0.0, 1.0, GRID_W)
GRID_Y_PROPS = np.linspace(0.0, 1.0, GRID_H)


class MeshDragger:
    def __init__(self, root):
        self.root = root
        self.root.title("Topology Mesh Dragger")
        
        os.makedirs(MESH_DIR, exist_ok=True)
        
        # Mode Selection
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--candidates", action="store_true")
        args, _ = parser.parse_known_args()

        if args.candidates and os.path.exists(CANDIDATES_PATH):
            with open(CANDIDATES_PATH, 'r') as f:
                self.label_files = json.load(f)
            print(f"🎯 LOADING {len(self.label_files)} CANDIDATES FOR MANUAL CORRECTION")
        else:
            self.label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith('.json')])
        
        self.current_idx = 0
        
        self.mesh_points = [] # 108 points in original image coordinates
        self.manual_corners = [] # Temp store for 4-click manual placement
        self.active_node = None # Index of the node currently being dragged
        self.texture_rotation_k = 1 # 90-degree CCW steps (Default CCW 1)
        # Standard colors: TL, TR, BR, BL
        self.point_colors = ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"]
        self.corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        
        # --- TEXTURE OVERLAY ---
        self.texture_path = r"C:\Users\Camer\OneDrive\Desktop\OpenCamera\paper_lines_golden.png"
        self.texture_img = cv2.imread(self.texture_path)
        if self.texture_img is not None:
            self.texture_img = cv2.cvtColor(self.texture_img, cv2.COLOR_BGR2RGB)
            self.tex_h, self.tex_w = self.texture_img.shape[:2]
        
        self.show_texture = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True) # Toggle for grid/node overlay
        self.texture_rotation = 1 # Matches user's manual fix
        self.texture_alpha = tk.DoubleVar(value=0.5)
        self.blend_mode = tk.StringVar(value="MULTIPLY") # NORMAL, MULTIPLY, DIFF, SCREEN, DIVIDE
        self.luma_mask = tk.BooleanVar(value=True) # Makes white in texture transparent
        self.texture_mirror = tk.BooleanVar(value=False)
        self.soft_drag = tk.BooleanVar(value=True)
        self.soft_radius = tk.DoubleVar(value=0.5) # Default for 5x5
        self.working_texture = None # Cached rotated texture
        
        self.setup_model()
        self.setup_ui()
        
        if not self.label_files:
            messagebox.showerror("Error", f"No base labels found in {LABEL_DIR}")
            self.root.destroy()
            return
            
        self.root.state('zoomed')
        self.root.after(100, self.load_data)

    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Navigation & Actions
        nav_frame = tk.Frame(control_frame, bg="#2c3e50")
        nav_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Button(nav_frame, text="< Back", command=self.prev_image, bg="#34495e", fg="white", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="Next >", command=self.next_image, bg="#34495e", fg="white", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="RESET (Esc)", command=self.reset_mesh, bg="#34495e", fg="#e67e22", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="TRANSPOSE (T)", command=self.transpose_corners, bg="#34495e", fg="#e67e22", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="ROTATE (F)", command=self.rotate_corners_cw, bg="#34495e", fg="#e67e22", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_frame, text="DELETE", command=self.delete_sample, bg="#c0392b", fg="white", font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=15)

        # --- INDEX JUMP PANEL ---
        jump_frame = tk.Frame(control_frame, bg="#2c3e50")
        jump_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(jump_frame, text="[", font=("Consolas", 14, "bold"), fg="white", bg="#2c3e50").pack(side=tk.LEFT)
        
        self.idx_entry = tk.Entry(jump_frame, width=4, font=("Consolas", 12, "bold"), justify='center', bg="#34495e", fg="#f1c40f", insertbackground="white", borderwidth=0)
        self.idx_entry.pack(side=tk.LEFT, padx=2)
        self.idx_entry.bind("<Return>", lambda e: self.jump_to_idx())
        
        self.total_label = tk.Label(jump_frame, text="/ 0 ]", font=("Consolas", 14, "bold"), fg="white", bg="#2c3e50")
        self.total_label.pack(side=tk.LEFT)
        
        self.info_label = tk.Label(control_frame, text="Loading filename...", font=("Consolas", 11), fg="#bdc3c7", bg="#2c3e50")
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(control_frame, text="", font=("Consolas", 10, "italic"), fg="#ecf0f1", bg="#2c3e50")
        self.status_label.pack(side=tk.LEFT, padx=10)

        instructions = tk.Label(control_frame, text="Drag Nodes | Esc = Reset | SPACE = Save | F = Rot90 | Shift+F = Flip180", 
                                font=("Consolas", 10, "bold"), fg="#f1c40f", bg="#2c3e50")
        instructions.pack(side=tk.RIGHT, padx=15)

        tk.Button(control_frame, text="ROTATE TEX", command=self.rotate_texture, 
                  bg="#34495e", fg="#f1c40f", font=("Consolas", 10, "bold")).pack(side=tk.RIGHT, padx=5)
        
        tk.OptionMenu(control_frame, self.blend_mode, "NORMAL", "MULTIPLY", "DIFF", "SCREEN", "DIVIDE",
                      command=lambda e: self.draw_mesh()).pack(side=tk.RIGHT, padx=5)
        tk.Label(control_frame, text="BLEND:", bg="#2c3e50", fg="#f1c40f", font=("Consolas", 8, "bold")).pack(side=tk.RIGHT)

        tk.Checkbutton(control_frame, text="LUMA-MSK", variable=self.luma_mask, 
                       command=self.draw_mesh, bg="#2c3e50", fg="#f1c40f", 
                       selectcolor="#34495e", activebackground="#2c3e50").pack(side=tk.RIGHT, padx=5)
        tk.Label(control_frame, text="|", bg="#2c3e50", fg="#34495e").pack(side=tk.RIGHT, padx=5)

        tk.Checkbutton(control_frame, text="SOFT-DRAG", variable=self.soft_drag, 
                       bg="#2c3e50", fg="#f1c40f", 
                       selectcolor="#34495e", activebackground="#2c3e50").pack(side=tk.RIGHT, padx=5)

        tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self.texture_alpha, command=lambda e: self.draw_mesh(),
                 label="ALPHA", bg="#2c3e50", fg="#f1c40f", font=("Consolas", 8),
                 troughcolor="#34495e", highlightthickness=0, width=10).pack(side=tk.RIGHT, padx=5)

        tk.Scale(control_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.soft_radius,
                 label="S-RAD", bg="#2c3e50", fg="#f1c40f", font=("Consolas", 8),
                 troughcolor="#34495e", highlightthickness=0, width=10).pack(side=tk.RIGHT, padx=5)

        tk.Checkbutton(control_frame, text="TEXTURE", variable=self.show_texture, 
                       command=self.draw_mesh, bg="#2c3e50", fg="#f1c40f", 
                       selectcolor="#34495e", activebackground="#2c3e50").pack(side=tk.RIGHT, padx=5)

        tk.Checkbutton(control_frame, text="LABELS", variable=self.show_labels, 
                       command=self.draw_mesh, bg="#2c3e50", fg="#f1c40f", 
                       selectcolor="#34495e", activebackground="#2c3e50").pack(side=tk.RIGHT, padx=5)

        self.canvas = tk.Canvas(self.root, bg="#1e272e", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Mouse & Keyboard Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Escape>", lambda e: self.reset_mesh())
        self.root.bind("<space>", lambda e: self.save_and_next())
        self.root.bind("r", self.rotate_texture)
        self.root.bind("R", self.rotate_texture)
        self.root.bind("t", lambda e: self.transpose_corners())
        self.root.bind("T", lambda e: self.transpose_corners())
        self.root.bind("f", lambda e: self.rotate_corners_cw())
        self.root.bind("F", lambda e: self.flip_mesh_180())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.delete_sample())
        
        # Scroll wheel for S-RAD
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("v", lambda e: self.toggle_overlays())
        self.root.bind("V", lambda e: self.toggle_overlays())

    def reset_mesh(self):
        self.mesh_points = []
        self.manual_corners = []
        self.status_label.config(text=f"PLEASE CLICK: {self.corner_names[0]}", fg="#f1c40f")
        self.draw_mesh()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_data()

    def next_image(self):
        base_range = len(self.label_files) - 1
        if self.current_idx < base_range:
            self.current_idx += 1
            self.load_data()

    def jump_to_idx(self):
        try:
            val = int(self.idx_entry.get())
            # Convert 1-based UI index to 0-based internal index
            new_idx = val - 1 
            if 0 <= new_idx < len(self.label_files):
                self.current_idx = new_idx
                self.load_data()
            else:
                self.status_label.config(text=f"OUT OF RANGE: 1 to {len(self.label_files)}", fg="#e74c3c")
                # Reset to current
                self.idx_entry.delete(0, tk.END)
                self.idx_entry.insert(0, str(self.current_idx + 1))
        except ValueError:
            self.status_label.config(text="INVALID NUMBER", fg="#e74c3c")
            self.idx_entry.delete(0, tk.END)
            self.idx_entry.insert(0, str(self.current_idx + 1))

    def rotate_texture(self):
        self.texture_rotation = (self.texture_rotation + 1) % 4
        self.working_texture = None # Force cache refresh
        self.draw_mesh()

    def update_texture_cache(self):
        if self.texture_img is None: return
        
        # Apply Rotation
        working = self.texture_img.copy()
        if self.texture_rotation == 1:
            working = cv2.rotate(working, cv2.ROTATE_90_CLOCKWISE)
        elif self.texture_rotation == 2:
            working = cv2.rotate(working, cv2.ROTATE_180)
        elif self.texture_rotation == 3:
            working = cv2.rotate(working, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        if self.texture_mirror.get():
            working = cv2.flip(working, 1)

        # PRE-RESIZE texture to reasonable quality for the canvas 
        # (e.g. 1024 max dim) to make warping much faster
        max_dim = 1024
        th, tw = working.shape[:2]
        if max(th, tw) > max_dim:
            scale = max_dim / max(th, tw)
            working = cv2.resize(working, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA)

        self.working_texture = working

    def delete_sample(self):
        if not self.label_files: return
        
        label_name = self.label_files[self.current_idx]
        if not messagebox.askyesno("Delete", f"Are you sure you want to delete {label_name} and its image?"):
            return
            
        try:
            # Paths
            label_path = os.path.join(LABEL_DIR, label_name)
            mesh_path = os.path.join(MESH_DIR, label_name)
            
            with open(label_path, 'r') as f:
                img_name = json.load(f)['image']
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            # Delete files
            for p in [label_path, mesh_path, img_path]:
                if os.path.exists(p): os.remove(p)
                
            # Update list and move on
            self.label_files.pop(self.current_idx)
            if self.current_idx >= len(self.label_files):
                self.current_idx = max(0, len(self.label_files) - 1)
            
            self.load_data()
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete: {e}")

    def generate_initial_mesh(self, corners):
        # corners are ordered: TL, TR, BR, BL (Standard)
        src_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        dst_pts = np.array(corners, dtype=np.float32)
        
        # Calculate Homography to map a perfect square to the paper's perspective
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Project the grid onto the paper
        grid_pts = []
        for x in range(GRID_W):
            for y in range(GRID_H): # Top-to-Bottom
                px = GRID_X_PROPS[x]
                py = GRID_Y_PROPS[y]
                grid_pts.append([px, py])
                
        grid_pts = np.array([grid_pts], dtype=np.float32)
        warped_pts = cv2.perspectiveTransform(grid_pts, H)[0]
        return warped_pts.tolist()

    def interpolate_mesh(self, old_mesh, old_w, old_h, new_w, new_h):
        """High-precision manual bilinear interpolation to keep corners locked."""
        try:
            grid = np.array(old_mesh).reshape(old_w, old_h, 2)
            
            # Map new grid (0..new-1) to old grid coordinates (0..old-1)
            new_xs = np.linspace(0, old_w - 1, new_w)
            new_ys = np.linspace(0, old_h - 1, new_h)
            
            new_mesh = np.zeros((new_w, new_h, 2), dtype=np.float32)
            
            for c_idx, ox in enumerate(new_xs):
                for r_idx, oy in enumerate(new_ys):
                    # Find floor index for base point, capping to old-2 for safe +1 lookup
                    x_base = int(np.floor(ox)); x_base = min(x_base, old_w - 2)
                    y_base = int(np.floor(oy)); y_base = min(y_base, old_h - 2)
                    
                    dx = ox - x_base
                    dy = oy - y_base
                    
                    # 4-point neighborhood
                    p00, p10 = grid[x_base, y_base], grid[x_base + 1, y_base]
                    p01, p11 = grid[x_base, y_base + 1], grid[x_base + 1, y_base + 1]
                    
                    # Perform bilinear interpolation
                    top = (1 - dx) * p00 + dx * p10
                    bot = (1 - dx) * p01 + dx * p11
                    new_mesh[c_idx, r_idx] = (1 - dy) * top + dy * bot
                    
            return new_mesh.reshape(-1, 2).tolist()
        except Exception as e:
            print(f"Bilinear interpolation failed: {e}. Falling back.")
            return None

    def setup_model(self):
        try:
            import timm
            import torch.nn as nn
            
            def dsnt(heatmaps):
                B, K, H, W = heatmaps.shape
                device = heatmaps.device
                y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)
                x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
                expected_y = torch.sum(heatmaps * y_coords, dim=[2, 3])
                expected_x = torch.sum(heatmaps * x_coords, dim=[2, 3])
                return torch.stack([expected_x, expected_y], dim=-1)

            class HeatmapMeshRegressor(nn.Module):
                def __init__(self, num_points=108):
                    super().__init__()
                    self.backbone = timm.create_model('hrnet_w32', pretrained=False, features_only=True)
                    self.final_layer = nn.Sequential(
                        nn.Conv2d(480, num_points, kernel_size=1),
                        nn.Softmax(dim=-1)
                    )
                def forward(self, x):
                    feats = self.backbone(x)[-1]
                    B, C, H, W = feats.shape
                    logits = self.final_layer(feats)
                    heatmaps = logits.view(B, -1, H * W)
                    heatmaps = F.softmax(heatmaps, dim=-1)
                    heatmaps = heatmaps.view(B, -1, H, W)
                    return dsnt(heatmaps)

            self.model = HeatmapMeshRegressor(num_points=GRID_COUNT).to(DEVICE)
            if os.path.exists(MODEL_PATH):
                ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(ckpt, strict=False)
                self.model.eval()
                print(f"✅ Loaded Model: {MODEL_PATH}")
            else:
                self.model = None
                print("⚠️ Model not found. Live NME disabled.")
        except Exception as e:
            print(f"❌ Model Setup Failed: {e}")
            self.model = None

    def get_live_nme(self):
        if self.model is None or not self.mesh_points: return 0.0
        try:
            # 1. Prep Image
            img_rgb = np.array(self.original_img.convert('RGB'))
            img_t = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            img_t = torch.from_numpy(img_t).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
            
            # 2. Inference
            with torch.no_grad():
                pred = self.model(img_t)[0] # [108, 2] in [-1, 1]
            
            # 3. Ground Truth (Current Mesh)
            gt = np.array(self.mesh_points, dtype=np.float32)
            gt[:, 0] = (gt[:, 0] / self.orig_w) * 2 - 1
            gt[:, 1] = (gt[:, 1] / self.orig_h) * 2 - 1
            gt_t = torch.from_numpy(gt).to(DEVICE)
            
            # 4. Score (NME)
            error = torch.norm(pred - gt_t, dim=-1).mean().item()
            return error
        except:
            return 0.0

    def load_data(self):
        if self.current_idx < 0 or self.current_idx >= len(self.label_files):
            return
            
        self.canvas.delete("all")
        
        # Read the source JSON
        label_name = self.label_files[self.current_idx]
        with open(os.path.join(LABEL_DIR, label_name), 'r') as f:
            self.base_data = json.load(f)
            
        img_name = self.base_data['image']
        self.img_path = os.path.join(IMAGE_DIR, img_name)
        
        # Check if we already have a mesh saved for this, otherwise generate one
        mesh_path = os.path.join(MESH_DIR, label_name)
        corners = self.base_data.get('corners') or self.base_data.get('points')
        
        if os.path.exists(mesh_path):
            with open(mesh_path, 'r') as f:
                saved_data = json.load(f)
                self.mesh_points = saved_data.get('mesh', [])
                
                # If it's a 9x12 grid, we may need to handle it, but for now we just mismatch
                if len(self.mesh_points) != GRID_COUNT:
                    print(f"Grid size mismatch ({len(self.mesh_points)} != {GRID_COUNT}). Regenerating.")
                    self.mesh_points = self.generate_initial_mesh(corners)
        else:
            if corners is None:
                print(f"Warning: No 'corners' or 'points' found in {label_name}")
                self.current_idx += 1
                self.load_data()
                return
            self.mesh_points = self.generate_initial_mesh(corners)

        # Load Image
        self.original_img = Image.open(self.img_path)
        self.orig_w, self.orig_h = self.original_img.size
        
        # Update UI Counters
        self.idx_entry.delete(0, tk.END)
        self.idx_entry.insert(0, str(self.current_idx + 1))
        self.total_label.config(text=f"/ {len(self.label_files)} ]")
        self.info_label.config(text=img_name)
        
        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        self.scale = min(canvas_w / self.orig_w, canvas_h / self.orig_h) * 0.95
        
        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)
        
        self.display_img = self.original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.display_img)
        
        self.x_offset = (canvas_w - new_w) // 2
        self.y_offset = (canvas_h - new_h) // 2
        
        self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.tk_img)
        self.draw_mesh()

    def img_to_canvas(self, x, y):
        return (x * self.scale) + self.x_offset, (y * self.scale) + self.y_offset

    def canvas_to_img(self, x, y):
        return (x - self.x_offset) / self.scale, (y - self.y_offset) / self.scale

    def draw_mesh(self):
        self.canvas.delete("mesh")
        self.canvas.delete("texture_layer")
        
        # 1. Draw Texture Overlay (Organic Bending)
        if self.show_texture.get() and self.texture_img is not None and len(self.mesh_points) == GRID_COUNT:
             self.overlay_texture()
        
        # 2. Live NME Scoring
        nme = self.get_live_nme()
        color = "#2ecc71" if nme < 0.05 else ("#f1c40f" if nme < 0.1 else "#e74c3c")
        self.root.title(f"Topology Mesh Dragger | {self.label_files[self.current_idx]} | NME: {nme*100:.2f}%")
        self.status_label.config(text=f"MODEL ERROR: {nme*100:.2f}%", fg=color)
        
        # If we are in manual placement mode, show current clicks
        if not self.mesh_points and self.manual_corners:
            for i, pt in enumerate(self.manual_corners):
                cx, cy = self.img_to_canvas(*pt)
                self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill=self.point_colors[i], outline="white", tags="mesh")
            return

        if not self.mesh_points:
            return

        # 2. Draw Labels (Grid & Nodes)
        if self.show_labels.get():
            # Draw Lines (Column-Major traversal)
            for c in range(GRID_W):
                for r in range(GRID_H):
                    idx = c * GRID_H + r
                    cx, cy = self.img_to_canvas(*self.mesh_points[idx])
                    
                    # Horizontal lines (connect to next column)
                    if c < GRID_W - 1:
                        nx, ny = self.img_to_canvas(*self.mesh_points[idx + GRID_H])
                        # Color coding: Top(r=0)=Yellow, Bottom(r=GRID_H-1)=Blue
                        color = "#34495e" # Default inner
                        width = 1
                        if r == 0: 
                            color = "#f1c40f" # Yellow Top
                            width = 2
                        elif r == GRID_H - 1: 
                            color = "#3498db" # Blue Bottom
                            width = 2
                            
                        self.canvas.create_line(cx, cy, nx, ny, fill=color, width=width, tags="mesh")
                    
                    # Vertical lines (connect to next row in same column)
                    if r < GRID_H - 1:
                        nx, ny = self.img_to_canvas(*self.mesh_points[idx + 1])
                        # Color coding: Left(c=0)=Green, Right(c=GRID_W-1)=Red
                        color = "#34495e" # Default inner
                        width = 1
                        if c == 0: 
                            color = "#2ecc71" # Green Left
                            width = 2
                        elif c == GRID_W - 1: 
                            color = "#e74c3c" # Red Right
                            width = 2
                            
                        self.canvas.create_line(cx, cy, nx, ny, fill=color, width=width, tags="mesh")

            # Draw Nodes
            # corner_indices match column-major: TL(0,0)=0, BL(0,H-1)=H-1, TR(W-1,0)=(W-1)*H, BR(W-1,H-1)=COUNT-1
            # Order in point_colors: TL, TR, BR, BL
            corner_indices_map = {
                0: 0,                   # TL
                (GRID_W-1)*GRID_H: 1,   # TR
                GRID_COUNT-1: 2,        # BR
                GRID_H-1: 3             # BL
            }
            for i, pt in enumerate(self.mesh_points):
                cx, cy = self.img_to_canvas(*pt)
                if i == self.active_node:
                    color, rad = "white", 6
                elif i in corner_indices_map:
                    color, rad = self.point_colors[corner_indices_map[i]], 5
                else:
                    color, rad = "#95a5a6", 3
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, 
                                        fill=color, outline="white", width=1, tags="mesh")
                self.canvas.create_text(cx + 8, cy - 8, text=str(i), fill="white", font=("Arial", 8, "bold"), tags="mesh")

    def overlay_texture(self):
        if self.working_texture is None:
            self.update_texture_cache()
            
        working_texture = np.rot90(self.working_texture, k=self.texture_rotation_k).copy()
        t_h, t_w = working_texture.shape[:2]
        canvas_img = np.array(self.display_img).copy()
        h, w = canvas_img.shape[:2]
        
        # 1. Prepare Texture Tensor (Cached)
        # Convert to (1, 3, H, W) and normalize to [-1, 1] for grid_sample if needed? 
        # Actually grid_sample samples from [0, 255] just fine if we pass as float.
        tex_t = torch.from_numpy(working_texture).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        
        # 2. Build the Control Grid [1, GRID_H, GRID_W, 2]
        # We need to map Canvas Pixels -> Texture Pixels.
        # This is the Inverse Map.
        # mesh_points stores (x, y) on Canvas. 
        # We want to associate these with (u, v) in Texture.
        
        # mesh_points in col-major: [c*H + r]
        # We'll build a grid of where each mesh point IS on the canvas.
        mesh_grid_canvas = np.array(self.mesh_points).reshape(GRID_W, GRID_H, 2)
        
        # Get bounding box of mesh on canvas to minimize processing area
        pts_canvas = (mesh_grid_canvas * self.scale).astype(np.float32)
        x_min, y_min = np.min(pts_canvas, axis=(0, 1))
        x_max, y_max = np.max(pts_canvas, axis=(0, 1))
        
        # Pad slightly
        x_min, y_min = max(0, int(x_min - 5)), max(0, int(y_min - 5))
        x_max, y_max = min(w, int(x_max + 5)), min(h, int(y_max + 5))
        
        crop_w, crop_h = x_max - x_min, y_max - y_min
        if crop_w <= 0 or crop_h <= 0: return

        # To use grid_sample, we need values in [-1, 1]
        # We want to find for each pixel in the crop [x_min..x_max, y_min..y_max], its (u, v) in texture.
        # This is an INVERSE mapping.
        # But we only have a FORWARD mapping (SourceProp -> CanvasPos).
        # Trick: Use a small grid of (u, v) relative to [x, y] and interpolate.
        
        # More robust: use the piecewise warp logic but vectorised if possible.
        # Short-term fix for "Laggy": The piecewise warping was slow because it warped full-canvas 64 times.
        # Let's do the CROPPED piecewise warp first, it's 100x faster and requires no complex inverse mapping.
        
        warped_full = np.zeros_like(canvas_img)
        mask_full = np.zeros((h, w), dtype=np.uint8)

        # SUB-GRID Piecewise Warp with ROI CROP (Silky Smooth)
        SUB_X, SUB_Y = 8, 8
        sx_new = np.linspace(0, 1.0, SUB_X + 1)
        sy_new = np.linspace(0, 1.0, SUB_Y + 1)
        x_orig = np.linspace(0, 1.0, GRID_W)
        y_orig = np.linspace(0, 1.0, GRID_H)
        
        sub_x_props = np.interp(sx_new, x_orig, GRID_X_PROPS)
        sub_y_props = np.interp(sy_new, y_orig, GRID_Y_PROPS)
        
        # Pre-calc sub_grid
        grid = np.array(self.mesh_points).reshape(GRID_W, GRID_H, 2) * self.scale
        sub_grid = np.zeros((SUB_Y + 1, SUB_X + 1, 2), dtype=np.float32)
        for y_idx in range(SUB_Y + 1):
            for x_idx in range(SUB_X + 1):
                ix = min(np.searchsorted(x_orig, sx_new[x_idx], side='right') - 1, GRID_W - 2)
                iy = min(np.searchsorted(y_orig, sy_new[y_idx], side='right') - 1, GRID_H - 2)
                wx = (sx_new[x_idx] - x_orig[ix]) / (x_orig[ix+1] - x_orig[ix])
                wy = (sy_new[y_idx] - y_orig[iy]) / (y_orig[iy+1] - y_orig[iy])
                p00, p10, p01, p11 = grid[ix, iy], grid[ix+1, iy], grid[ix, iy+1], grid[ix+1, iy+1]
                sub_grid[y_idx, x_idx] = (1-wx)*(1-wy)*p00 + wx*(1-wy)*p10 + (1-wx)*wy*p01 + wx*wy*p11

        for y in range(SUB_Y):
            for x in range(SUB_X):
                lx, rx = sub_x_props[x] * t_w, sub_x_props[x+1] * t_w
                yt, yb = sub_y_props[y] * t_h, sub_y_props[y+1] * t_h
                
                # Source quad points (Order: TL, BL, BR, TR)
                src = np.array([
                    [lx, yt], [lx, yb], [rx, yb], [rx, yt]
                ], dtype=np.float32)
                
                # Destination quad points on canvas
                dst = np.array([
                    sub_grid[y, x],      # TL
                    sub_grid[y + 1, x],  # BL
                    sub_grid[y + 1, x + 1], # BR
                    sub_grid[y, x + 1]   # TR
                ], dtype=np.float32)
                
                # ROI Crop Optimization
                bx, by, bw_roi, bh_roi = cv2.boundingRect(dst.astype(np.int32))
                if bw_roi <= 2 or bh_roi <= 2: continue # Ignore degenerate quads
                
                try:
                    # Clean the ROI destination
                    dst_roi = (dst - np.array([bx, by])).astype(np.float32)
                    M = cv2.getPerspectiveTransform(src, dst_roi)
                    roi_warped = cv2.warpPerspective(working_texture, M, (bw_roi, bh_roi))
                    
                    mask_roi = np.zeros((bh_roi, bw_roi), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_roi, dst_roi.astype(np.int32), 255)
                    
                    # Safe Paste into result
                    r_y1, r_y2 = max(0, by), min(h, by + bh_roi)
                    r_x1, r_x2 = max(0, bx), min(w, bx + bw_roi)
                    m_y1, m_y2 = r_y1 - by, r_y2 - by
                    m_x1, m_x2 = r_x1 - bx, r_x2 - bx
                    
                    if r_y2 > r_y1 and r_x2 > r_x1:
                        target_roi = warped_full[r_y1:r_y2, r_x1:r_x2]
                        source_roi = roi_warped[m_y1:m_y2, m_x1:m_x2]
                        mask_bool = mask_roi[m_y1:m_y2, m_x1:m_x2] > 0
                        target_roi[mask_bool] = source_roi[mask_bool]
                        mask_full[r_y1:r_y2, r_x1:r_x2][mask_bool] = 255
                except cv2.error:
                    continue # Skip quads that fail transforms

        # 3. Blending Logic
        alpha = self.texture_alpha.get()
        if alpha <= 0:
            # Short-circuit: 0 alpha = No Overlay
            self.tk_img_overlay = ImageTk.PhotoImage(Image.fromarray(canvas_img))
            self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.tk_img_overlay, tags="texture_layer")
            return

        idx = mask_full > 0
        mode = self.blend_mode.get()
        
        # Prepare Overlay
        ovl = warped_full.copy()
        
        # Apply Luma Mask: Multiply overlay by its own darkness
        # (Darker pixels in overlay stay opaque, lighter pixels become transparent)
        if self.luma_mask.get():
            gray = cv2.cvtColor(ovl, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            luma_alpha = 1.0 - gray # 1.0 for black, 0.0 for white
            # Expand to 3 channels
            luma_alpha_3 = cv2.merge([luma_alpha, luma_alpha, luma_alpha])
            # The "Overlay" itself is now masked
            # We'll use this to weigh the overlay in the final blend
            final_alpha = alpha * luma_alpha
        else:
            final_alpha = alpha

        def blend_op(bg, fg, m):
            if m == "MULTIPLY":
                return (bg.astype(np.float32) * fg.astype(np.float32) / 255.0).astype(np.uint8)
            elif m == "DIFF":
                return cv2.absdiff(bg, fg)
            elif m == "SCREEN":
                return (255 - ((255 - bg.astype(np.float32)) * (255 - fg.astype(np.float32)) / 255.0)).astype(np.uint8)
            elif m == "DIVIDE":
                return cv2.divide(bg, fg, scale=255)
            else:
                return fg

        # Apply Blend Operation
        blended = blend_op(canvas_img[idx], ovl[idx], mode)
        
        # Final weighting based on Alpha (and Luma if enabled)
        if isinstance(final_alpha, np.ndarray):
            # Per-pixel alpha weighting
            fa = final_alpha[mask_full > 0][:, None]
            canvas_img[idx] = (canvas_img[idx].astype(np.float32) * (1.0 - fa) + blended.astype(np.float32) * fa).astype(np.uint8)
        else:
            canvas_img[idx] = cv2.addWeighted(canvas_img[idx], 1.0 - final_alpha, blended, final_alpha, 0)
        
        self.tk_img_overlay = ImageTk.PhotoImage(Image.fromarray(canvas_img))
        self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.tk_img_overlay, tags="texture_layer")

    def toggle_overlays(self):
        # Toggle both texture and labels for a clean view
        new_state = not self.show_labels.get()
        self.show_labels.set(new_state)
        self.show_texture.set(new_state)
        self.draw_mesh()
    def rotate_texture(self, event=None):
        self.texture_rotation_k = (self.texture_rotation_k + 1) % 4
        self.status_label.config(text=f"Texture Rotated (k={self.texture_rotation_k})", fg="#3498db")
        self.draw_mesh()

    def transpose_corners(self):
        """Swaps (col, row) -> (row, col). For non-square grids, this regenerates from the 4 corners."""
        if len(self.mesh_points) == GRID_COUNT:
            # Extract current 4 corners from mesh
            # [TL, TR, BR, BL]
            c = [
                self.mesh_points[0],                        # TL
                self.mesh_points[(GRID_W-1)*GRID_H],        # TR
                self.mesh_points[GRID_COUNT-1],             # BR
                self.mesh_points[GRID_H-1]                  # BL
            ]
            # Transpose: Swap TR and BL
            c[1], c[3] = c[3], c[1]
            self.mesh_points = self.generate_initial_mesh(c)
            self.status_label.config(text="Mesh TRANSPOSED (Regenerated)", fg="#e67e22")
            self.draw_mesh()
        elif len(self.manual_corners) == 4:
            self.manual_corners[1], self.manual_corners[3] = self.manual_corners[3], self.manual_corners[1]
            self.draw_mesh()

    def rotate_corners_cw(self):
        """Rotates grid 90 deg CW. For non-square grids, this regenerates from the 4 corners."""
        if len(self.mesh_points) == GRID_COUNT:
            # Extract current 4 corners from mesh
            # [TL, TR, BR, BL]
            c = [
                self.mesh_points[0],                        # TL
                self.mesh_points[(GRID_W-1)*GRID_H],        # TR
                self.mesh_points[GRID_COUNT-1],             # BR
                self.mesh_points[GRID_H-1]                  # BL
            ]
            # Rotate CW: New TL = Old BL, New TR = Old TL, etc.
            c = [c[3]] + c[:3]
            self.mesh_points = self.generate_initial_mesh(c)
            self.status_label.config(text="Mesh ROTATED CW (Regenerated)", fg="#9b59b6")
            self.draw_mesh()
        elif len(self.manual_corners) == 4:
            self.manual_corners = [self.manual_corners[3]] + self.manual_corners[:3]
            self.draw_mesh()

    def flip_mesh_180(self):
        """180-degree flip (Reverses point order). Preserves all manual deformations."""
        if len(self.mesh_points) == GRID_COUNT:
            self.mesh_points = self.mesh_points[::-1]
            self.status_label.config(text="Mesh FLIPPED 180 (Points Reversed)", fg="#9b59b6")
            self.draw_mesh()
        elif len(self.manual_corners) == 4:
            self.manual_corners = self.manual_corners[::-1]
            self.draw_mesh()

    def on_press(self, event):
        # 1. Handle Manual Corner Placement
        if not self.mesh_points:
            img_x, img_y = self.canvas_to_img(event.x, event.y)
            self.manual_corners.append([img_x, img_y])
            
            num_clicks = len(self.manual_corners)
            if num_clicks < 4:
                self.status_label.config(text=f"PLEASE CLICK: {self.corner_names[num_clicks]}", fg="#f1c40f")
                self.draw_mesh()
            else:
                # We have 4 corners! Generate mesh
                self.mesh_points = self.generate_initial_mesh(self.manual_corners)
                self.manual_corners = []
                self.status_label.config(text="Mesh generated. Fine-tune by dragging.", fg="#2ecc71")
                self.draw_mesh()
            return

        # 2. Find the closest node within a click radius
        click_radius = 15 # pixels on screen
        min_dist = float('inf')
        self.active_node = None
        
        for i, pt in enumerate(self.mesh_points):
            cx, cy = self.img_to_canvas(*pt)
            dist = np.hypot(event.x - cx, event.y - cy)
            if dist < click_radius and dist < min_dist:
                min_dist = dist
                self.active_node = i
                
        if self.active_node is not None:
            self.draw_mesh()

    def on_drag(self, event):
        if self.active_node is not None:
            img_x, img_y = self.canvas_to_img(event.x, event.y)
            
            # Calculate Movement Vector
            old_x, old_y = self.mesh_points[self.active_node]
            dx, dy = img_x - old_x, img_y - old_y
            
            if self.soft_drag.get():
                # Hierarchical Soft-Drag (Strict Perimeter Pinning):
                # 1. Perimeter points ONLY move if they are the direct node being dragged.
                # 2. Interior points move via Gaussian influence from any dragged node.
                r_active, c_active = self.active_node % GRID_H, self.active_node // GRID_H
                
                sigma = self.soft_radius.get()
                if sigma <= 0:
                    # Fallback to direct drag if radius is 0
                    self.mesh_points[self.active_node] = [img_x, img_y]
                    self.draw_mesh()
                    return

                for i in range(len(self.mesh_points)):
                    r, c = i % GRID_H, i // GRID_H
                    target_on_edge = (r == 0 or r == GRID_H - 1 or c == 0 or c == GRID_W - 1)
                    
                    # STRICT RULE: Perimeter nodes never move unless they are the active node
                    if target_on_edge and i != self.active_node:
                        continue
                        
                    dist_grid = np.hypot(r - r_active, c - c_active)
                    influence = np.exp(- (dist_grid**2) / (2 * sigma**2 + 1e-9))
                    
                    self.mesh_points[i][0] += dx * influence
                    self.mesh_points[i][1] += dy * influence
            else:
                # Direct drag only
                self.mesh_points[self.active_node] = [img_x, img_y]
                
            self.draw_mesh()

    def on_release(self, event):
        if self.active_node is not None:
            self.active_node = None
            self.draw_mesh()

    def on_mousewheel(self, event):
        # Adjust S-RAD with mouse wheel
        delta = 0.05 if event.delta > 0 else -0.05
        new_val = self.soft_radius.get() + delta
        self.soft_radius.set(np.clip(new_val, 0.0, 1.0))
        self.status_label.config(text=f"S-RAD: {self.soft_radius.get():.2f}", fg="#f1c40f")

    def save_and_next(self):
        if not self.mesh_points:
            messagebox.showwarning("Warning", "Please finish placing the 4 corners first!")
            return
            
        label_name = self.label_files[self.current_idx]
        save_path = os.path.join(MESH_DIR, label_name)
        
        # Combine base data with the new dense mesh
        output_data = self.base_data.copy()
        output_data["mesh"] = self.mesh_points
        output_data["grid_w"] = GRID_W
        output_data["grid_h"] = GRID_H
        
        # Extract corners from mesh for standardized key
        # Col-major: TL=0, TR=(W-1)*H, BR=COUNT-1, BL=H-1
        if len(self.mesh_points) == GRID_COUNT:
            output_data["corners"] = [
                self.mesh_points[0],
                self.mesh_points[(GRID_W-1)*GRID_H],
                self.mesh_points[GRID_COUNT-1],
                self.mesh_points[GRID_H-1]
            ]
        
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        self.current_idx += 1
        self.load_data()

if __name__ == "__main__":
    root = tk.Tk()
    app = MeshDragger(root)
    root.mainloop()