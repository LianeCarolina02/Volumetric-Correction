import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
points = []
for i in range(117, 118):
    with open(f'Manequin/AVA_{i}.xml', 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    b_skin = str(Bs_data.find_all('skin'))[1:-1]
    root = ET.fromstring(b_skin)

    for ptx in root.findall('ptx'):
        coordinates = ptx.text.strip().split()
        points.append([float(coord) for coord in coordinates])

# Convert the list of points to a NumPy array
points_array = np.array(points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_array)

points = points_array

# Create colormap starting from blue
color_map = plt.get_cmap('Blues')

# Get color values from the colormap based on z-coordinate
z_min = np.min(points[:, 2])
z_max = np.max(points[:, 2])
colors = color_map(((1 - (points[:, 2] - z_min) / (z_max - z_min)) + 0.5))[:, :3]

# Apply colors to point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)


o3d.visualization.draw_geometries([pcd], window_name= "Breast", width=1600, height=1200, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
