# %%
import laspy
import numpy as np
import CSF
import pickle
import open3d as o3d
from scipy.interpolate import griddata
from scipy import ndimage as ndi
from pathlib import Path
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import alphashape
from scipy.interpolate import NearestNDInterpolator

# %% [markdown]
# # Tree and terrain point creation

# %%

def get_variable_window_size(h):
    """Computes the window diameter (in metres) based on height."""
    width = 3 + 0.00901 * (h**2)
    #width = 1.5 + 0.05 * h
    return width

def apply_vwf(chm, resolution, min_tree_height, min_distance):
    """Applies the variable window filter to detect peaks."""
    # 1. Identify initial candidates (fast local maxima)
    local_max = ndi.maximum_filter(chm, size=min_distance) == chm
    local_max = local_max & (chm > min_tree_height)

    y_coords, x_coords = np.where(local_max)
    final_peaks = []

    # 2. Filter candidates using the variable window
    # Sort descending by height
    if len(y_coords) > 0:
        vals = chm[y_coords, x_coords]
        sorted_indices = np.argsort(vals)[::-1]

        for i in sorted_indices:
            y, x = y_coords[i], x_coords[i]
            h = chm[y, x]

            # Compute radius
            window_width_m = get_variable_window_size(h)
            radius_px = int((window_width_m) / resolution)
            if radius_px < 1: radius_px = 1

            # Bounds
            y_min, y_max = max(0, y - radius_px), min(chm.shape[0], y + radius_px + 1)
            x_min, x_max = max(0, x - radius_px), min(chm.shape[1], x + radius_px + 1)

            window = chm[y_min:y_max, x_min:x_max]

            # Circular mask
            y_indices, x_indices = np.ogrid[y_min - y: y_max - y, x_min - x: x_max - x]
            dist_sq = y_indices**2 + x_indices**2
            mask_circular = dist_sq <= radius_px**2

            # Check: is the central point the maximum inside the circle?
            window_circular = np.where(mask_circular, window, -np.inf)  # -inf is safer than -1

            if h == np.max(window_circular):
                final_peaks.append([y, x])

    return np.array(final_peaks)

# %%
def read_las(las_path):
    """
    Reads a LAS file and returns the point cloud as a NumPy array.

    :param las_path: Path to the LAS file.
    :return: NumPy array of shape (N, 3) containing the point cloud coordinates.
    """
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points

# %%
def segment_terrain_points(points, resolution=0.5, min_tree_height=4.0):
    """
    Receives a point cloud and segments terrain points using CSF,
    then normalizes heights to create a CHM.
    """
    # 1. DTM and height normalization
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Build grid ensuring full coverage
    grid_x, grid_y = np.mgrid[min_x:max_x:resolution, min_y:max_y:resolution]

    csf = CSF.CSF()
    csf.params.bSloopSmooth = True
    csf.params.rigidness = 1
    csf.params.cloth_resolution = 0.75
    csf.params.time_step = 0.75
    csf.params.iterations = 500
    csf.setPointCloud(points)
    ground_indices = CSF.VecInt(); non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices)

    csf_ground_points = points[ground_indices]

    interpolated_dtm = griddata(
        points=csf_ground_points[:, :2],
        values=csf_ground_points[:, 2],
        xi=(grid_x, grid_y),
        method='linear',
    )

    if np.isnan(interpolated_dtm).any():
        nan_mask = np.isnan(interpolated_dtm)
        nan_filler = NearestNDInterpolator(csf_ground_points[:, :2], csf_ground_points[:, 2])
        nan_coords = np.vstack((grid_x[nan_mask], grid_y[nan_mask])).T
        interpolated_dtm[nan_mask] = nan_filler(nan_coords)

    # terrain_points is saved optionally; kept for reference
    terrain_points = np.vstack((grid_x.ravel(), grid_y.ravel(), interpolated_dtm.ravel())).T
    np.save("terrain_points.npy", terrain_points)

    # Safe indices (clamped)
    idx_x_all = np.clip(((points[:, 0] - min_x) / resolution).astype(int), 0, interpolated_dtm.shape[0] - 1)
    idx_y_all = np.clip(((points[:, 1] - min_y) / resolution).astype(int), 0, interpolated_dtm.shape[1] - 1)

    ground_z = interpolated_dtm[idx_x_all, idx_y_all]
    above_ground_height = points[:, 2] - ground_z

    candidate_mask = above_ground_height > min_tree_height
    candidate_tree_points = points[candidate_mask]
    candidate_heights = above_ground_height[candidate_mask]
    remaining_points = points[~candidate_mask]

    return candidate_tree_points, candidate_heights, remaining_points, grid_x, grid_y, idx_x_all, idx_y_all, candidate_mask, min_x, min_y, above_ground_height, csf_ground_points

# %%
def create_chm(candidate_tree_points, candidate_heights, grid_x, grid_y):
    """
    Creates a Canopy Height Model (CHM) by interpolating candidate tree points
    and applying a Gaussian filter for smoothing.
    """
    chm_grid = griddata(
        points=candidate_tree_points[:, :2],
        values=candidate_heights,
        xi=(grid_x, grid_y),
        method='linear',
        fill_value=0
    )
    chm_grid = np.nan_to_num(chm_grid, nan=0)
    chm_smooth = ndi.gaussian_filter(chm_grid, sigma=1)

    return chm_grid, chm_smooth

# %%
def apply_watershed(resolution, min_tree_height, chm_smooth, idx_x_all, idx_y_all, candidate_mask, candidate_tree_points):
    """
    Uses the smoothed CHM to apply the watershed algorithm for initial tree
    segmentation, then assigns labels to candidate points.
    """
    min_crown_dist_m = 3.0
    dist_px = int(min_crown_dist_m / resolution)

    peak_coords = apply_vwf(chm_smooth, resolution, min_tree_height, dist_px)

    mask_watershed = chm_smooth > min_tree_height
    markers = np.zeros(chm_smooth.shape, dtype=int)

    if len(peak_coords) > 0:
        for i, (y, x) in enumerate(peak_coords):
            markers[y, x] = i + 1

    labels_raster = watershed(-chm_smooth, markers, mask=mask_watershed)

    # Assign raster labels to 3D points
    idx_x_cand = idx_x_all[candidate_mask]
    idx_y_cand = idx_y_all[candidate_mask]
    labels_points = labels_raster[idx_x_cand, idx_y_cand]

    df_candidates = pd.DataFrame(candidate_tree_points, columns=['X', 'Y', 'Z'])
    df_candidates['label'] = labels_points
    # Label 0 is background/non-tree in watershed
    df_candidates.loc[df_candidates['label'] == 0, 'label'] = -1

    unique_labels_final = df_candidates['label'].unique()
    valid_tree_indices = []
    rejected_indices = []

    # Extract noise points
    df_noise_ws = df_candidates[df_candidates['label'] == -1]
    noise_points = df_noise_ws[['X', 'Y', 'Z']].values

    return df_candidates, noise_points, labels_raster

# %%
def analyze_geometric_features(df_candidates, sphericity_threshold):
    """
    Checks the geometric features of watershed clusters to filter out
    non-tree clusters based on their sphericity.
    """
    valid_tree_indices = []
    rejected_list = []

    groups = df_candidates[df_candidates['label'] != -1].groupby('label')

    for label, group in groups:
        cluster_3d = group[['X', 'Y', 'Z']].values
        if len(cluster_3d) < 5: continue

        try:
            cov = np.cov(cluster_3d, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(cov)
            l1, l2, l3 = np.sort(eigenvalues)
            sphericity = l1 / l3 if l3 > 0 else 0

            if sphericity > sphericity_threshold:
                valid_tree_indices.append(cluster_3d)
            else:
                rejected_list.append(cluster_3d)
        except:
            rejected_list.append(cluster_3d)

    # Merge rejected clusters into one point block to avoid visualization gaps
    rejected_points = np.vstack(rejected_list) if rejected_list else np.empty((0, 3))
    return valid_tree_indices, rejected_points

# %%
def create_csv(valid_tree_indices, chm_grid, min_x, min_y, resolution):
    """
    Creates a final DataFrame with tree points and their heights, then saves it
    as a CSV. Also prepares data for visualization.
    """
    # --- OPTIMIZATION: build final DataFrame without a slow loop ---
    tree_dfs = []
    tree_vis_points = []
    tree_vis_colors = []

    for idx, cluster in enumerate(valid_tree_indices):
        df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z'])

        # VECTORIZED HEIGHT LOOKUP (much faster)
        # Convert X,Y coords to grid indices
        rows = np.clip(((cluster[:, 0] - min_x) / resolution).astype(int), 0, chm_grid.shape[0]-1)
        cols = np.clip(((cluster[:, 1] - min_y) / resolution).astype(int), 0, chm_grid.shape[1]-1)

        # Direct extraction
        relative_heights = chm_grid[rows, cols]

        df_cluster['Tree_Height'] = relative_heights
        df_cluster['label'] = idx
        tree_dfs.append(df_cluster)

        # Prepare visualization
        tree_vis_points.append(cluster)
        color = np.random.rand(3)
        tree_vis_colors.append(np.tile(color, (len(cluster), 1)))

    if tree_dfs:
        trees_xyz = pd.concat(tree_dfs, ignore_index=True)
        trees_xyz.to_csv("trees_detected.csv", index=False)

        # Merge lists into single arrays
        tree_vis_points = np.vstack(tree_vis_points)
        tree_vis_colors = np.vstack(tree_vis_colors)
    else:
        trees_xyz = pd.DataFrame()
        tree_vis_points = np.array([])
        tree_vis_colors = np.array([])

    return trees_xyz, tree_vis_points, tree_vis_colors

# %%
def prepare_visualization_data(valid_tree_indices, chm_grid, min_x, min_y, resolution):
    """
    Prepares the data for visualization by creating a DataFrame for tree points
    and their heights, and compiles points and colours for Open3D.
    """
    dfs_list = []
    points_list = []
    colors_list = []

    for idx, cluster in enumerate(valid_tree_indices):
        rows = np.clip(((cluster[:, 0] - min_x) / resolution).astype(int), 0, chm_grid.shape[0]-1)
        cols = np.clip(((cluster[:, 1] - min_y) / resolution).astype(int), 0, chm_grid.shape[1]-1)

        df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z'])
        df_cluster['Tree_Height'] = chm_grid[rows, cols]
        df_cluster['label'] = idx
        dfs_list.append(df_cluster)

        points_list.append(cluster)
        color = np.random.rand(3)
        colors_list.append(np.tile(color, (len(cluster), 1)))

    if dfs_list:
        trees_df = pd.concat(dfs_list, ignore_index=True)
        # Merge here to avoid ValueError in visualization
        vis_points = np.vstack(points_list)
        vis_colors = np.vstack(colors_list)
    else:
        trees_df = pd.DataFrame()
        vis_points = np.empty((0, 3))
        vis_colors = np.empty((0, 3))

    return trees_df, vis_points, vis_colors

def combine_all_background_points(remaining_points, noise_points, rejected_points):
    """
    Combines all points that are NOT trees into a single block to avoid gaps.
    """
    bases = [remaining_points]
    if len(noise_points) > 0: bases.append(noise_points)
    if len(rejected_points) > 0: bases.append(rejected_points)
    return np.vstack(bases)


# %%
def classify_tree_watershed(trees_las_path, terrain_las_path, resolution=0.5, min_tree_height=4.0, sphericity_threshold=0.05):

    # 1. Load
    tree_points = read_las(trees_las_path)
    terrain_points = read_las(terrain_las_path)
    points = np.vstack((tree_points, terrain_points))

    # 2. Terrain and normalization
    candidate_tree_points, candidate_heights, remaining_points, grid_x, grid_y, idx_x_all, idx_y_all, candidate_mask, min_x, min_y, above_ground_height, csf_ground_points = segment_terrain_points(points, resolution, min_tree_height)

    # 3. CHM
    chm_grid, chm_smooth = create_chm(candidate_tree_points, candidate_heights, grid_x, grid_y)

    # 4. Watershed segmentation
    df_candidates, noise_points, _ = apply_watershed(resolution, min_tree_height, chm_smooth, idx_x_all, idx_y_all, candidate_mask, candidate_tree_points)

    # 5. Geometric filtering (returns valid tree list and rejected point block)
    valid_indices, rejected_points = analyze_geometric_features(df_candidates, sphericity_threshold)

    # 6. Prepare results (unifies point and colour arrays)
    trees_df, vis_tree_points, vis_tree_colors = prepare_visualization_data(valid_indices, chm_grid, min_x, min_y, resolution)

    # 7. Combine background (avoids visualization gaps)
    background_points = combine_all_background_points(remaining_points, noise_points, rejected_points)

    df_background = pd.DataFrame(background_points, columns=['X', 'Y', 'Z'])
    df_background['label'] = -1

    trees_df = pd.concat([trees_df, df_background], ignore_index=True)

    return trees_df
