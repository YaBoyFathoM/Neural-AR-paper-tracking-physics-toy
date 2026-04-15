# OpenCamera: Neural AR Physics Sandbox

OpenCamera is a real-time, high-fidelity Augmented Reality physics sandbox. It uses a custom-trained HRNet mesh regressor to detect a physical piece of paper through a webcam, projects drawn shapes onto it using piecewise-affine warping, transferring real-world specular lighting and shadows, and runs a full physical simulation (complete with tilt-based gravity based on paper pose) directly on top.

![Webcam Demo](examples/webcam_demo_recording.gif)

## Features
- **Real-Time Mesh Tracking**: Uses HRNet to regress 108 structural points for perfectly curved projection mapping.
- **Topological Gravity**: PyMunk physics engine derives directional gravity from the paper's 3D tilt.
- **Real-World Lighting Transfer**: Employs live, dynamic occlusion matting and specular highlight tracking to mix the virtual objects with real-world shadows seamlessly.
- **High Performance**: Optimizations including FP16 inference, bounding box isolation, and multithreaded 1-Euro temporal filtering for 60+ FPS bounds.

---

## Installation

You need Python 3.9+ and a CUDA-compatible GPU for real-time tracking (although CPU fallback is supported).

```bash
git clone https://github.com/your-username/OpenCamera
cd OpenCamera
pip install -r requirements.txt
```

> **Note:** The pre-trained weights (`best_9x12_768.pth`) are too large for GitHub. You must generate synthetic data and train the model yourself using the included pipeline, or download the weights from the releases page (if provided).

---

## Core Components

The repository is stripped down to the primary 4 scripts necessary to run, train, and modify the AR pipeline.

### 1. `webcam_demo.py`
The main application. Opens a webcam stream, tracks the paper, and projects the Pygame/PyMunk physics rendering onto the feed.
```bash
python webcam_demo.py --width 1280 --height 720
```
**Controls:**
- **Mouse**: Draw shapes onto the paper.
- **c**: Clear canvas
- **b**: Cycle brush color
- **+/-**: Change brush size
- **r**: Rotate canvas overlay
- **t**: Toggle AR texture / raw camera
- **f**: Toggle performance HUD
- **q / ESC**: Quit

### 2. `generate_synthetic.py`
The entire dataset used to train the neural network can be synthetically generated. This script creates photorealistic synthetic images of paper at thousands of different angles, occlusion levels, and lighting environments, automatically creating the ground-truth coordinates needed for training.
```bash
python generate_synthetic.py
```

### 3. `train_mesh.py`
The PyTorch neural network training loop. It consumes the dataset containing the paper meshes and trains an HRNet model to predict those 108 coordinate nodes directly from webcam frames. It features dynamic augmentations, curriculum learning, and real-time visualization dashboards.
```bash
python train_mesh.py
```

**Training Visualizer (Dashboard Example):**
![Training Visualizer](examples/epoch_075.png)

### 4. `mesh_dragger.py`
A visual debugging utility used heavily during development. Allows you to load an image, predict the mesh, and actively click/drag the internal or external nodes of the mesh grid around to verify piecewise-affine curvature and AR warp projections manually perfectly.
```bash
python mesh_dragger.py
```

---

## License
MIT License
