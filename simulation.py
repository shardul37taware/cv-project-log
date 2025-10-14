# Generating a drone-survey simulation (top-down camera footprint at 15 m) and saving an animated GIF.
# This code will:
# - Create a rectangular survey area (meters)
# - Scatter random shapes (rectangles, circles, polygons) as "objects"
# - Add a disaster zone polygon and a marked target inside it
# - Compute a lawnmower path (back-and-forth) for a drone flying at 15 m altitude
# - Simulate a camera footprint using a chosen horizontal FOV and produce frames centered on the drone
# - Save frames and output an animated GIF to /mnt/data/drone_survey.gif
# No external web access required.
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend (no GUI)
import matplotlib.pyplot as plt

from matplotlib import patches
from PIL import Image
import math
import random
import imageio

random.seed(42)
np.random.seed(42)

# --- Parameters ---
AREA_W = 80.0   # meters (x direction)
AREA_H = 50.0   # meters (y direction)
NUM_OBJECTS = 30
DRONE_HEIGHT = 15.0  # meters
CAM_FOV_DEG = 60.0   # horizontal field of view in degrees (estimate)
FRAME_W = 640
FRAME_H = 480
ALTITUDE = DRONE_HEIGHT

# camera footprint width (meters) at altitude using simple pinhole model: width = 2 * h * tan(fov/2)
cam_fov_rad = math.radians(CAM_FOV_DEG)
footprint_w = 2 * ALTITUDE * math.tan(cam_fov_rad / 2)
# Maintain aspect ratio of frame to compute footprint_h
aspect = FRAME_H / FRAME_W
footprint_h = footprint_w * aspect

# Lawn mower path parameters
strip_spacing = footprint_h * 0.6  # overlap ~40%
num_strips = max(1, int(math.ceil(AREA_H / strip_spacing)))
strip_spacing = AREA_H / num_strips  # adjust to cover exactly

# Build lawnmower waypoints (center positions)
waypoints = []
y = strip_spacing / 2.0
direction = 1
while y < AREA_H + 1e-6:
    if direction == 1:
        x_positions = np.linspace(footprint_w/2.0, AREA_W - footprint_w/2.0, max(2, int(AREA_W/(footprint_w/2))+1))
    else:
        x_positions = np.linspace(AREA_W - footprint_w/2.0, footprint_w/2.0, max(2, int(AREA_W/(footprint_w/2))+1))
    for x in x_positions:
        waypoints.append((x, y))
    y += strip_spacing
    direction *= -1

# Create random obstacles
objects = []
for i in range(NUM_OBJECTS):
    typ = random.choice(['circle', 'rect', 'poly'])
    cx = random.uniform(0, AREA_W)
    cy = random.uniform(0, AREA_H)
    if typ == 'circle':
        r = random.uniform(0.4, 3.0)
        objects.append(('circle', (cx, cy, r)))
    elif typ == 'rect':
        w = random.uniform(0.8, 6.0)
        h = random.uniform(0.8, 6.0)
        angle = random.uniform(0, 360)
        objects.append(('rect', (cx, cy, w, h, angle)))
    else:
        # polygon: triangle or quad
        pts = []
        for _ in range(random.choice([3,4,5])):
            rx = random.uniform(-2.5, 2.5)
            ry = random.uniform(-2.5, 2.5)
            pts.append((cx + rx, cy + ry))
        objects.append(('poly', pts))

# Add a disaster zone polygon (clustered irregular polygon)
dz_center = (AREA_W * 0.7, AREA_H * 0.6)
dz_pts = []
for ang in np.linspace(0, 2*math.pi, 10)[:-1]:
    r = random.uniform(3.0, 10.0)
    dz_pts.append((dz_center[0] + r*math.cos(ang), dz_center[1] + r*math.sin(ang)))
disaster_zone = dz_pts

# Target inside disaster zone
# Choose a point near the center with some offset
target = (dz_center[0] + random.uniform(-2,2), dz_center[1] + random.uniform(-2,2))

# Utility: convert world coords to image crop of camera footprint
def render_frame(center_x, center_y):
    fig, ax = plt.subplots(figsize=(FRAME_W/100, FRAME_H/100), dpi=100)
    # Set limits so plotted area is footprint centered
    half_w = footprint_w/2.0
    half_h = footprint_h/2.0
    ax.set_xlim(center_x - half_w, center_x + half_w)
    ax.set_ylim(center_y - half_h, center_y + half_h)
    ax.set_aspect('equal', adjustable='box')
    # Draw ground (just a rectangle background)
    ax.add_patch(patches.Rectangle((center_x-half_w, center_y-half_h), footprint_w, footprint_h, fill=True, alpha=1.0))
    # Plot objects that intersect the footprint
    def bbox_intersects(xmin,xmax,ymin,ymax):
        return not (xmax < center_x-half_w or xmin > center_x+half_w or ymax < center_y-half_h or ymin > center_y+half_h)
    for obj in objects:
        if obj[0] == 'circle':
            cx,cy,r = obj[1]
            if bbox_intersects(cx-r, cx+r, cy-r, cy+r):
                circ = patches.Circle((cx,cy), r, alpha=0.9)
                ax.add_patch(circ)
        elif obj[0] == 'rect':
            cx,cy,w,h,angle = obj[1]
            xmin, xmax = cx - w/2, cx + w/2
            ymin, ymax = cy - h/2, cy + h/2
            if bbox_intersects(xmin, xmax, ymin, ymax):
                rect = patches.Rectangle((cx-w/2, cy-h/2), w, h, angle=angle, alpha=0.9)
                t = mpl.transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)
        else:
            pts = obj[1]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
            if bbox_intersects(xmin, xmax, ymin, ymax):
                poly = patches.Polygon(pts, closed=True, alpha=0.9)
                ax.add_patch(poly)
    # Draw disaster zone (as polygon) if intersects
    xs = [p[0] for p in disaster_zone]; ys = [p[1] for p in disaster_zone]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    if bbox_intersects(xmin, xmax, ymin, ymax):
        dz = patches.Polygon(disaster_zone, closed=True, alpha=0.5)
        ax.add_patch(dz)
    # Mark target
    tx, ty = target
    if (center_x-half_w <= tx <= center_x+half_w) and (center_y-half_h <= ty <= center_y+half_h):
        ax.plot(tx, ty, marker='*', markersize=12)
    # Drone position marker (center of image) - show small crosshair
    ax.plot(center_x, center_y, marker='+', markersize=6)
    ax.axis('off')
    fig.tight_layout(pad=0)
    # Save to PIL image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

# Fix: import mpl here to use transforms
import matplotlib as mpl

# Generate frames along waypoints (sample rate)
frames = []
for i, (x,y) in enumerate(waypoints):
    img = render_frame(x, y)
    frames.append(img)
    # limit frames if too many
    if len(frames) >= 200:
        break

# Save GIF
out_dir = Path('/mnt/data')
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / 'drone_survey.gif'
imageio.mimsave(str(gif_path), frames, fps=6)

# Save first frame as preview
preview_path = out_dir / 'drone_frame0.png'
Image.fromarray(frames[0]).save(preview_path)

print(f"Saved animated GIF to: {gif_path}")
print(f"Saved preview image to: {preview_path}")
