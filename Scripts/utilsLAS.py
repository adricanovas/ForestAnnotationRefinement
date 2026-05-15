
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



def inspeccionar_header(ruta_las):
    """
    Inspecciona el encabezado de un archivo LAS/LAZ y resume:
    - Versión, formato de puntos y total de puntos
    - Escalas, offsets, mínimos y máximos (X, Y, Z)
    - Área cubierta (ancho x alto en metros)
    - Extensión real: mínimos y máximos de X e Y calculados sobre los puntos

    Parámetros:
        ruta_las (str | Path): Ruta al archivo LAS/LAZ.

    Efectos:
        Imprime la información por consola. Termina si el archivo no existe.
    """
    if not Path(ruta_las).exists():
        print("Archivo no encontrado.")
        return

    with laspy.open(ruta_las) as f:
        header = f.header
        
        print(f"--- INFORMACIÓN DE {Path(ruta_las).name} ---")
        print(f"Versión LAS: {header.version}")
        print(f"Formato de Puntos ID: {header.point_format.id}")
        print(f"Total de Puntos: {header.point_count:,}")
        
        print("\n--- GEOMETRÍA ---")
        print(f"Escalas (Precisión): {header.scales}")
        print(f"Offsets (Origen):    {header.offsets}")
        print(f"Mínimos (X,Y,Z):     {header.min}")
        print(f"Máximos (X,Y,Z):     {header.max}")
        
        ancho = header.max[0] - header.min[0]
        alto = header.max[1] - header.min[1]
        print(f"Área cubierta: {ancho:.2f}m x {alto:.2f}m")

    las = laspy.read(ruta_las)
    x_min = las.x.min()
    x_max = las.x.max()
    y_min = las.y.min()
    y_max = las.y.max()
    print("\n--- COORDENADAS REALES ---")
    print(f"🌍 X Min: {x_min:.2f}, X Max: {x_max:.2f}")
    print(f"🌍 Y Min: {y_min:.2f}, Y Max: {y_max:.2f}")

def visualizar_lidar_3d(ruta_archivo: str):
    """
    Visualiza una nube de puntos LAS/LAZ en 3D usando Open3D.
    - Centra la nube en el origen para mejor visualización.
    - Colorea por elevación (Z) con mapa de color tipo “jet”.

    Parámetros:
        ruta_archivo (str): Ruta al archivo LAS/LAZ.

    Efectos:
        Abre una ventana interactiva 3D con la nube de puntos renderizada.
    """

    print(f"📡 Cargando nube de puntos: {Path(ruta_archivo).name}...")
    try:
        las = laspy.read(ruta_archivo)
    except Exception as e:
        print(f"❌ Error leyendo el archivo: {e}")
        return

    points = np.vstack((las.x, las.y, las.z)).T
    centro = points.mean(axis=0)
    points = points - centro

    z_norm = (las.z - las.z.min()) / (las.z.max() - las.z.min())
    colors = plt.get_cmap("jet")(z_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("-" * 30)
    print("Abriendo visor 3D...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Visor LiDAR: {Path(ruta_archivo).name}")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()

def visualizar_lidar_rgb(ruta_archivo: str):
    import laspy
    import open3d as o3d
    import numpy as np
    from pathlib import Path

    print(f"📸 Cargando RGB de: {Path(ruta_archivo).name}...")
    
    las = laspy.read(ruta_archivo)
    limit = 1_000_000

    print(f"Total de puntos en el archivo: {len(las)}. Visualizando los primeros {limit} puntos...")
    
    # 1. Get Points and Center
    points = las.xyz[:limit].copy()
    points -= points.mean(axis=0)

    # 2. Get RGB and Scale
    # Check if the file actually has color attributes
    if hasattr(las, 'red'):
        red = las.red[:limit]
        green = las.green[:limit]
        blue = las.blue[:limit]
        print(f"Max Red: {las.red.max()}, Max Green: {las.green.max()}, Max Blue: {las.blue.max()}")
        
        # Scale from 16-bit to 0.0-1.0 float for Open3D
        colors = np.vstack((red, green, blue)).T / 65535.0
    else:
        print("⚠️ No RGB data found in file. Falling back to gray.")
        colors = np.ones((len(points), 3)) * 0.5

    # 3. Open3D Setup
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visor LiDAR RGB")
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Dark gray background
    opt.point_size = 3.0
    
    vis.run()
    vis.destroy_window()


def convert_las_to_assets(las_path: str, output_folder: str):
    las = laspy.read(las_path)
    os.makedirs(output_folder, exist_ok=True)

    # 1. COORDENADAS: Centrado en float64 para evitar pérdida de precisión
    # Leemos en float64 (double) para que la resta sea exacta
    xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    
    np.save(os.path.join(output_folder, "coord.npy"), xyz_raw)

    # 2. INSTANCIA: Obtenido de classification
    # Los árboles empiezan con ID 3, mientras que en el 0,1,2 están el resto de puntos
    instance = np.array(las.classification).astype(np.int32)
    instance[instance < 3] = -1  # Suelo y vegetación baja no tienen instancia
    np.save(os.path.join(output_folder, "instance.npy"), instance)

    # 3. SEMÁNTICA: Clase semántica de los puntos basada en classification
    # Los puntos con classification >= 3 tendrán la etiqueta 1 y los que no, etiqueta 0
    segment = np.array(las.classification).astype(np.int32)
    segment[instance >= 3] = 1  # Cualquier punto con instancia (árbol) es clase 1
    segment[instance < 3] = 0 # Suelo y vegetación baja es clase 0
    np.save(os.path.join(output_folder, "segment.npy"), segment)

    # 4. INTENSIDAD: Normalización a [0, 1]. En principio no se usará
    intensity = np.array(las.intensity).reshape(-1, 1).astype(np.float32)
    if intensity.max() > 0:
        intensity = intensity / intensity.max() # Normalización robusta
    # Guardamos como 'intensity.npy' para coincidir con feat_keys del config
    np.save(os.path.join(output_folder, "intensity.npy"), intensity) 

    # 5. COLOR: Normalizamos a [0, 1] como en los artículos y guardamos como 'color.npy' (RGB)
    # extraer RGB
    red = las.red
    green = las.green
    blue = las.blue
    color = np.vstack((red, green, blue)).transpose().astype(np.float32)
    
    np.save(os.path.join(output_folder, "color.npy"), color)

    print(f" Assets creados en: {output_folder} ({xyz_raw.shape[0]} puntos)")


def convert_las_to_assets_conference(las_path: str, output_folder: str):
    las = laspy.read(las_path)
    os.makedirs(output_folder, exist_ok=True)

    # 1. COORDENADAS: Centrado en float64 para evitar pérdida de precisión
    # Leemos en float64 (double) para que la resta sea exacta
    xyz_raw = np.vstack((las.x, las.y, las.z)).transpose()
    
    np.save(os.path.join(output_folder, "coord.npy"), xyz_raw)

    # 2. INSTANCIA: Obtenido de classification
    # Los árboles empiezan con ID 3, mientras que en el 0,1,2 están el resto de puntos
    instance = np.array(las['intermediate_segs']).astype(np.int32)
    instance[instance == 0] = -1  # Suelo y vegetación baja no tienen instancia
    np.save(os.path.join(output_folder, "instance.npy"), instance)

    # 3. SEMÁNTICA: Clase semántica de los puntos basada en classification
    # Los puntos con classification >= 3 tendrán la etiqueta 1 y los que no, etiqueta 0
    segment = np.array(las['intermediate_segs']).astype(np.int32)
    segment[instance > 0] = 1  # Cualquier punto con instancia (árbol) es clase 1
    segment[instance == 0] = 0 # Suelo y vegetación baja es clase 0
    np.save(os.path.join(output_folder, "segment.npy"), segment)

    # 4. INTENSIDAD: Normalización a [0, 1]. En principio no se usará
    intensity = np.array(las.intensity).reshape(-1, 1).astype(np.float32)
    if intensity.max() > 0:
        intensity = intensity / intensity.max() # Normalización robusta
    # Guardamos como 'intensity.npy' para coincidir con feat_keys del config
    np.save(os.path.join(output_folder, "intensity.npy"), intensity) 

    # 5. COLOR: Normalizamos a [0, 1] como en los artículos y guardamos como 'color.npy' (RGB)
    # extraer RGB
    red = las.red
    green = las.green
    blue = las.blue
    color = np.vstack((red, green, blue)).transpose().astype(np.float32)
    
    np.save(os.path.join(output_folder, "color.npy"), color)

    print(f" Assets creados en: {output_folder} ({xyz_raw.shape[0]} puntos)")