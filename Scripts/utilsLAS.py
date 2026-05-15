
import os
import numpy as np
import laspy
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pandas as pd

import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt



def inspect_header(las_path):
    """
    Inspects a LAS/LAZ file header and prints a summary:
    - Version, point format, and total point count
    - Scales, offsets, mins and maxs (X, Y, Z)
    - Covered area (width x height in metres)
    - Real extent: X and Y mins/maxs computed from the points

    Args:
        las_path (str | Path): Path to the LAS/LAZ file.

    Effects:
        Prints info to the console. Returns early if the file does not exist.
    """
    if not Path(las_path).exists():
        print("Archivo no encontrado.")
        return

    with laspy.open(las_path) as f:
        header = f.header
        
        print(f"--- INFO: {Path(las_path).name} ---")
        print(f"LAS Version: {header.version}")
        print(f"Point Format ID: {header.point_format.id}")
        print(f"Total Points: {header.point_count:,}")
        
        print("\n--- GEOMETRY ---")
        print(f"Scales (Precision): {header.scales}")
        print(f"Offsets (Origin):    {header.offsets}")
        print(f"Mins (X,Y,Z):        {header.min}")
        print(f"Maxs (X,Y,Z):        {header.max}")
        
        width = header.max[0] - header.min[0]
        height = header.max[1] - header.min[1]
        print(f"Covered area: {width:.2f}m x {height:.2f}m")

    las = laspy.read(las_path)
    x_min = las.x.min()
    x_max = las.x.max()
    y_min = las.y.min()
    y_max = las.y.max()
    print("\n--- REAL COORDINATES ---")
    print(f"X Min: {x_min:.2f}, X Max: {x_max:.2f}")
    print(f"Y Min: {y_min:.2f}, Y Max: {y_max:.2f}")

def visualize_lidar_3d(file_path: str):
    """
    Visualizes a LAS/LAZ point cloud in 3D using Open3D.
    - Centers the cloud at the origin for better visualization.
    - Colours by elevation (Z) using a “jet” colour map.

    Args:
        file_path (str): Path to the LAS/LAZ file.

    Effects:
        Opens an interactive 3D window with the rendered point cloud.
    """

    print(f"Loading point cloud: {Path(file_path).name}...")
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    points = np.vstack((las.x, las.y, las.z)).T
    center = points.mean(axis=0)
    points = points - center

    z_norm = (las.z - las.z.min()) / (las.z.max() - las.z.min())
    colors = plt.get_cmap("jet")(z_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("-" * 30)
    print("Opening 3D viewer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"LiDAR Viewer: {Path(file_path).name}")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()

def visualize_lidar_rgb(file_path: str):
    import laspy
    import open3d as o3d
    import numpy as np
    from pathlib import Path

    print(f"Loading RGB from: {Path(file_path).name}...")
    
    las = laspy.read(file_path)
    limit = 1_000_000

    print(f"Total points in file: {len(las)}. Visualizing the first {limit} points...")
    
    # Get Points and Center
    points = las.xyz[:limit].copy()
    points -= points.mean(axis=0)

    # Get RGB and Scale
    # Check if the file actually has color attributes
    if hasattr(las, 'red'):
        red = las.red[:limit]
        green = las.green[:limit]
        blue = las.blue[:limit]
        print(f"Max Red: {las.red.max()}, Max Green: {las.green.max()}, Max Blue: {las.blue.max()}")
        
        # Scale from 16-bit to 0.0-1.0 float for Open3D
        colors = np.vstack((red, green, blue)).T / 65535.0
    else:
        print("No RGB data found in file. Falling back to gray.")
        colors = np.ones((len(points), 3)) * 0.5

    # 3. Open3D Setup
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR RGB Viewer")
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Dark gray background
    opt.point_size = 3.0
    
    vis.run()
    vis.destroy_window()


def convert_las_to_assets(las_path: str, output_folder: str):
    las = laspy.read(las_path)
    os.makedirs(output_folder, exist_ok=True)

    # COORDINATES: float64 to avoid precision loss
    # Read as float64 (double) so subtraction is exact
    xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    
    np.save(os.path.join(output_folder, "coord.npy"), xyz_raw)

    # INSTANCE: derived from classification
    # Trees start at ID 3; classes 0,1,2 are non-tree
    instance = np.array(las.classification).astype(np.int32)
    instance[instance < 3] = -1  # Ground and low vegetation have no instance
    np.save(os.path.join(output_folder, "instance.npy"), instance)

    # SEMANTIC: semantic class based on classification
    # Points with classification >= 3 → label 1, others → label 0
    segment = np.array(las.classification).astype(np.int32)
    segment[instance >= 3] = 1  # Any instanced point (tree) is class 1
    segment[instance < 3] = 0  # Ground and low vegetation are class 0
    np.save(os.path.join(output_folder, "segment.npy"), segment)

    # INTENSITY: normalize to [0, 1] (not used for training by default)
    intensity = np.array(las.intensity).reshape(-1, 1).astype(np.float32)
    if intensity.max() > 0:
        intensity = intensity / intensity.max()  # Robust normalization
    # Saved as 'intensity.npy' to match feat_keys in the config
    np.save(os.path.join(output_folder, "intensity.npy"), intensity) 

    # COLOUR: normalize to [0, 1] as in the papers; saved as 'color.npy' (RGB)
    # Extract RGB
    red = las.red
    green = las.green
    blue = las.blue
    color = np.vstack((red, green, blue)).transpose().astype(np.float32)
    
    np.save(os.path.join(output_folder, "color.npy"), color)

    print(f" Assets created in: {output_folder} ({xyz_raw.shape[0]} points)")


def convert_las_to_assets_conference(las_path: str, output_folder: str):
    las = laspy.read(las_path)
    os.makedirs(output_folder, exist_ok=True)

    # COORDINATES: float64 to avoid precision loss
    # Read as float64 (double) so subtraction is exact
    xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    
    np.save(os.path.join(output_folder, "coord.npy"), xyz_raw)

    # INSTANCE: derived from classification
    # Trees start at ID 3; classes 0,1,2 are non-tree
    instance = np.array(las['intermediate_segs']).astype(np.int32)
    instance[instance == 0] = -1  # Ground and low vegetation have no instance
    np.save(os.path.join(output_folder, "instance.npy"), instance)

    # SEMANTIC: semantic class based on classification
    # Points with classification >= 3 → label 1, others → label 0
    segment = np.array(las['intermediate_segs']).astype(np.int32)
    segment[instance > 0] = 1  # Any instanced point (tree) is class 1
    segment[instance == 0] = 0  # Ground and low vegetation are class 0
    np.save(os.path.join(output_folder, "segment.npy"), segment)

    # INTENSITY: normalize to [0, 1] (not used for training by default)
    intensity = np.array(las.intensity).reshape(-1, 1).astype(np.float32)
    if intensity.max() > 0:
        intensity = intensity / intensity.max()  # Robust normalization
    # Saved as 'intensity.npy' to match feat_keys in the config
    np.save(os.path.join(output_folder, "intensity.npy"), intensity) 

    # COLOUR: normalize to [0, 1] as in the papers; saved as 'color.npy' (RGB)
    # Extract RGB
    red = las.red
    green = las.green
    blue = las.blue
    color = np.vstack((red, green, blue)).transpose().astype(np.float32)
    
    np.save(os.path.join(output_folder, "color.npy"), color)

    print(f" Assets created in: {output_folder} ({xyz_raw.shape[0]} points)")