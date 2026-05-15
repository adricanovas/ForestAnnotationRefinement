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
# # Creación puntos arbol y suelo

# %%

def get_variable_window_size(h):
    """Calcula el diámetro de la ventana (en metros) basado en la altura."""
    width = 3 + 0.00901 * (h**2)
    #width = 1.5 + 0.05 * h
    return width

def apply_vwf(chm, resolucion, altura_min_arbol, distancia_minima):
    """Aplica el filtro de ventana variable para detectar picos."""
    # 1. Identificar candidatos iniciales (máximos locales rápidos)
    local_max = ndi.maximum_filter(chm, size=distancia_minima) == chm
    local_max = local_max & (chm > altura_min_arbol)
    
    y_coords, x_coords = np.where(local_max)
    picos_finales = []

    # 2. Filtrar candidatos usando la ventana variable
    # Ordenamos descendente por altura
    if len(y_coords) > 0:
        vals = chm[y_coords, x_coords]
        sorted_indices = np.argsort(vals)[::-1]
        
        for i in sorted_indices:
            y, x = y_coords[i], x_coords[i]
            h = chm[y, x]
            
            # Calcular radio
            window_width_m = get_variable_window_size(h)
            radius_px = int((window_width_m) / resolucion)
            if radius_px < 1: radius_px = 1
            
            # Límites
            y_min, y_max = max(0, y - radius_px), min(chm.shape[0], y + radius_px + 1)
            x_min, x_max = max(0, x - radius_px), min(chm.shape[1], x + radius_px + 1)
            
            window = chm[y_min:y_max, x_min:x_max]
            
            # Máscara circular
            y_indices, x_indices = np.ogrid[y_min - y : y_max - y, x_min - x : x_max - x]
            dist_sq = y_indices**2 + x_indices**2
            mask_circular = dist_sq <= radius_px**2
            
            # Comprobación: si el punto central es el máximo dentro del círculo
            window_circular = np.where(mask_circular, window, -np.inf) # Usar -inf es más seguro que -1
            
            if h == np.max(window_circular):
                picos_finales.append([y, x])
                
    return np.array(picos_finales)

# %%
def read_las(ruta_las):
    """
    Reads a LAS file and returns the point cloud as a NumPy array.

    :param ruta_las: Path to the LAS file.
    :return: NumPy array of shape (N, 3) containing the point cloud coordinates.
    """
    las = laspy.read(ruta_las)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points

# %%
def segment_terrain_points(points, resolution = 0.5, altura_min_arbol = 4.0):
    """
    Receives a point cloud and segments terrain points using CSF, then normalizes heights to create a CHM.
    """
    # 1. MDT y NORMALIZACIÓN DE ALTURA
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Crear grid asegurando que cubra todo
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

    puntos_suelo_csf = points[ground_indices]

    mdt_interpolado = griddata(
        points=puntos_suelo_csf[:, :2],
        values=puntos_suelo_csf[:, 2],
        xi=(grid_x, grid_y),
        method='linear',
    )

    if np.isnan(mdt_interpolado).any():
        mask_nan = np.isnan(mdt_interpolado)
        rellenador = NearestNDInterpolator(puntos_suelo_csf[:, :2], puntos_suelo_csf[:, 2])
        coords_nan = np.vstack((grid_x[mask_nan], grid_y[mask_nan])).T
        mdt_interpolado[mask_nan] = rellenador(coords_nan)

    # Nota: Guardar terrain_points es opcional, lo dejo comentado para limpieza
    terrain_points = np.vstack((grid_x.ravel(), grid_y.ravel(), mdt_interpolado.ravel())).T
    np.save("terrain_points.npy", terrain_points)

    # Índices seguros
    idx_x_all = np.clip(((points[:, 0] - min_x) / resolution).astype(int), 0, mdt_interpolado.shape[0] - 1)
    idx_y_all = np.clip(((points[:, 1] - min_y) / resolution).astype(int), 0, mdt_interpolado.shape[1] - 1)

    z_suelo_puntos = mdt_interpolado[idx_x_all, idx_y_all]
    altura_sobre_suelo = points[:, 2] - z_suelo_puntos

    mask_candidatos = altura_sobre_suelo > altura_min_arbol
    puntos_arbol_candidatos = points[mask_candidatos]
    alturas_candidatos = altura_sobre_suelo[mask_candidatos]
    puntos_resto = points[~mask_candidatos]

    return puntos_arbol_candidatos, alturas_candidatos, puntos_resto, grid_x, grid_y, idx_x_all, idx_y_all, mask_candidatos, min_x, min_y, altura_sobre_suelo, puntos_suelo_csf

# %%
def create_chm(puntos_arbol_candidatos, alturas_candidatos, grid_x, grid_y):
    """
    Creates a Canopy Height Model (CHM) by interpolating candidate tree points and applying a Gaussian filter for smoothing.
    """
    chm_grid = griddata(
        points=puntos_arbol_candidatos[:, :2],
        values=alturas_candidatos,
        xi=(grid_x, grid_y),
        method='linear',
        fill_value=0
    )
    chm_grid = np.nan_to_num(chm_grid, nan=0)
    chm_smooth = ndi.gaussian_filter(chm_grid, sigma=1)

    return chm_grid, chm_smooth

# %%
def apply_watershed(resolucion, altura_min_arbol, chm_smooth, idx_x_all, idx_y_all, mask_candidatos, puntos_arbol_candidatos):
    """
    Uses the smoothed CHM to apply the watershed algorithm for initial tree segmentation, then assigns labels to candidate points.
    """
    distancia_min_copas_metros = 3.0
    dist_px = int(distancia_min_copas_metros / resolucion)

    coords_picos = apply_vwf(chm_smooth, resolucion, altura_min_arbol, dist_px)

    mask_watershed = chm_smooth > altura_min_arbol
    markers = np.zeros(chm_smooth.shape, dtype=int)

    if len(coords_picos) > 0:
        for i, (y, x) in enumerate(coords_picos):
            markers[y, x] = i + 1

    labels_raster = watershed(-chm_smooth, markers, mask=mask_watershed)


    # Asignar etiquetas del raster a los puntos 3D
    idx_x_cand = idx_x_all[mask_candidatos]
    idx_y_cand = idx_y_all[mask_candidatos]
    labels_puntos = labels_raster[idx_x_cand, idx_y_cand]

    df_candidatos = pd.DataFrame(puntos_arbol_candidatos, columns=['X', 'Y', 'Z'])
    df_candidatos['label'] = labels_puntos
    # Etiqueta 0 es fondo/no árbol en watershed
    df_candidatos.loc[df_candidatos['label'] == 0, 'label'] = -1

    unique_labels_final = df_candidatos['label'].unique()
    indices_arboles_validos = []
    indices_falsos = []

    # Extraer ruido
    df_ruido_ws = df_candidatos[df_candidatos['label'] == -1]
    puntos_ruido = df_ruido_ws[['X', 'Y', 'Z']].values

    return df_candidatos, puntos_ruido, labels_raster

# %%
def analyze_geometric_features(df_candidatos, umbral_esfericidad):
    """
    Checks the geometric features of the clusters obtained from watershed to filter out non-tree clusters based on their sphericity.
    """
    indices_arboles_validos = []
    lista_falsos = []
    
    grupos = df_candidatos[df_candidatos['label'] != -1].groupby('label')

    for label, group in grupos:
        cluster_3d = group[['X', 'Y', 'Z']].values
        if len(cluster_3d) < 5: continue

        try:
            cov = np.cov(cluster_3d, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(cov)
            l1, l2, l3 = np.sort(eigenvalues)
            esfericidad = l1 / l3 if l3 > 0 else 0

            if esfericidad > umbral_esfericidad:
                indices_arboles_validos.append(cluster_3d)
            else:
                lista_falsos.append(cluster_3d)
        except:
            lista_falsos.append(cluster_3d)

    # Convertimos los descartados en un solo bloque de puntos para evitar huecos
    puntos_descartados = np.vstack(lista_falsos) if lista_falsos else np.empty((0, 3))
    return indices_arboles_validos, puntos_descartados

# %%
def create_csv(indices_arboles_validos, chm_grid, min_x, min_y, resolucion):
    """
    Creates a final DataFrame with tree points and their heights, then saves it as a CSV. Also prepares data for visualization.
    """
    # --- OPTIMIZACIÓN: Crear DataFrame final sin bucle lento ---
    lista_dfs_arboles = []
    puntos_vis_arbol = []
    colores_vis_arbol = []

    for idx, cluster in enumerate(indices_arboles_validos):
        df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z'])
        
        # VECTORIZACIÓN DE ALTURAS (Mucho más rápido)
        # Convertir coord X,Y a indices de grilla
        rows = np.clip(((cluster[:, 0] - min_x) / resolucion).astype(int), 0, chm_grid.shape[0]-1)
        cols = np.clip(((cluster[:, 1] - min_y) / resolucion).astype(int), 0, chm_grid.shape[1]-1)
        
        # Extracción directa
        alturas_relativas = chm_grid[rows, cols]
        
        df_cluster['Tree_Height'] = alturas_relativas
        df_cluster['label'] = idx
        lista_dfs_arboles.append(df_cluster)
        
        # Preparar visualización
        puntos_vis_arbol.append(cluster)
        color = np.random.rand(3)
        colores_vis_arbol.append(np.tile(color, (len(cluster), 1)))

    if lista_dfs_arboles:
        arbolesXYZ = pd.concat(lista_dfs_arboles, ignore_index=True)
        arbolesXYZ.to_csv("arboles_detectados.csv", index=False)
        
        # Unificar las listas en un solo array ---
        puntos_vis_arbol = np.vstack(puntos_vis_arbol) 
        colores_vis_arbol = np.vstack(colores_vis_arbol)
    else:
        arbolesXYZ = pd.DataFrame()
        puntos_vis_arbol = np.array([])
        colores_vis_arbol = np.array([])

    return arbolesXYZ, puntos_vis_arbol, colores_vis_arbol

# %%
def prepare_visualization_data(indices_arboles_validos, chm_grid, min_x, min_y, resolucion):
    """
    Prepares the data for visualization by creating a DataFrame for tree points and their heights, and also compiles the points and colors for Open3D visualization.
    """
    lista_dfs = []
    puntos_list = []
    colores_list = []

    for idx, cluster in enumerate(indices_arboles_validos):
        rows = np.clip(((cluster[:, 0] - min_x) / resolucion).astype(int), 0, chm_grid.shape[0]-1)
        cols = np.clip(((cluster[:, 1] - min_y) / resolucion).astype(int), 0, chm_grid.shape[1]-1)
        
        df_cluster = pd.DataFrame(cluster, columns=['X', 'Y', 'Z'])
        df_cluster['Tree_Height'] = chm_grid[rows, cols]
        df_cluster['label'] = idx
        lista_dfs.append(df_cluster)
        
        puntos_list.append(cluster)
        color = np.random.rand(3)
        colores_list.append(np.tile(color, (len(cluster), 1)))

    if lista_dfs:
        arboles_df = pd.concat(lista_dfs, ignore_index=True)
        # Unificamos aquí para evitar el ValueError en la visualización
        puntos_vis = np.vstack(puntos_list)
        colores_vis = np.vstack(colores_list)
    else:
        arboles_df = pd.DataFrame()
        puntos_vis = np.empty((0, 3))
        colores_vis = np.empty((0, 3))

    return arboles_df, puntos_vis, colores_vis

def combine_all_background_points(puntos_resto, puntos_ruido, puntos_descartados):
    """
    Combines all points that are NOT trees into a single block to avoid gaps.
    """
    bases = [puntos_resto]
    if len(puntos_ruido) > 0: bases.append(puntos_ruido)
    if len(puntos_descartados) > 0: bases.append(puntos_descartados)
    return np.vstack(bases)


# %%
def clasify_tree_watershed(ruta_las_trees, ruta_las_terrain, resolucion=0.5, altura_min_arbol=4.0, umbral_esfericidad=0.05):
    
    # 1. Carga
    points_trees = read_las(ruta_las_trees)

    points_terreno = read_las(ruta_las_terrain)
    points = np.vstack((points_trees, points_terreno))

    # 2. Terreno y Normalización
    puntos_arbol_candidatos, alturas_candidatos, puntos_resto, grid_x, grid_y, idx_x_all, idx_y_all, mask_candidatos, min_x, min_y, altura_sobre_suelo, puntos_suelo_csf = segment_terrain_points(points, resolucion, altura_min_arbol)

    # 3. CHM
    #chm_grid, chm_smooth = create_chm_fast(points, altura_sobre_suelo, gx, gy, resolucion) #Este método solo funciona con nubes de puntos densas, no con PNOA
    chm_grid, chm_smooth = create_chm(puntos_arbol_candidatos, alturas_candidatos, grid_x, grid_y)

    # 4. Segmentación Watershed
    df_candidatos, puntos_ruido, _ = apply_watershed(resolucion, altura_min_arbol, chm_smooth, idx_x_all, idx_y_all, mask_candidatos, puntos_arbol_candidatos)

    # 5. Filtrado Geométrico (Devuelve lista de árboles y bloque de descartados)
    indices_validos, puntos_descartados = analyze_geometric_features(df_candidatos, umbral_esfericidad)

    # 6. Preparación de resultados (Ya unifica arrays de puntos y colores)
    arboles_df, puntos_vis_arbol, colores_vis_arbol = prepare_visualization_data(indices_validos, chm_grid, min_x, min_y, resolucion)

    # 7. Unión de fondo (Evita los huecos en la visualización)
    puntos_fondo = combine_all_background_points(puntos_resto, puntos_ruido, puntos_descartados)

    df_fondo = pd.DataFrame(puntos_fondo, columns=['X', 'Y', 'Z'])
    df_fondo['label'] = -1

    arboles_df = pd.concat([arboles_df, df_fondo], ignore_index=True)


    return arboles_df
  