import open3d as o3d


folder = "Pacients"
number = 74
patient = "BR0" + f"{number}"
mesh_path = f"{folder}/{patient}/Segment_4.stl"

print("::   Read Mesh")
mesh = o3d.io.read_triangle_mesh(mesh_path)


voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 256

print(f'voxel_size = {voxel_size:e}')

print(
    f'Simplified mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
mesh_smp = mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)

print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)

print("::   Compute normals simplified")
mesh_smp.compute_vertex_normals()
print("::   Compute normals original")
mesh.compute_vertex_normals()

print("::   Paint meshes")
mesh_smp.paint_uniform_color((1, 0.706, 0))
mesh.paint_uniform_color((1, 0.706, 0))


print("::   Visualization")
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

visualizer.add_geometry(mesh_smp)
visualizer.run()
visualizer.destroy_window()

# o3d.io.write_triangle_mesh(f"{folder}/{patient}/try.stl", mesh_smp, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=True)
