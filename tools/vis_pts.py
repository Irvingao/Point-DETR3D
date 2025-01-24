import os
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import argparse
import numpy as np
# import trimesh
from line_mesh import LineMesh

def decode_obj_to_xyz(file):
    with open(file, 'r') as f:
        points = f.readlines()

    with open(file[:-3] + 'txt', 'w+') as f:
        for point in points:
            point = point.strip()
            _, x, y, z = point.split(' ')
            line = x + ' ' + y + ' ' + z + '\n'
            # line = z + ' ' + y + ' ' + x + '\n'
            f.write(line)

vis = o3d.visualization.Visualizer()
vis.create_window()

render_option = vis.get_render_option()
render_option.background_color = np.array([0, 0, 0])
render_option.point_size = 2.0
render_option.line_width = 50.0

folder_path = 'data'
# folder_path = 'data'
folder_list = os.listdir(folder_path)
file = folder_list[10]
# file = '1616100803399'
# pred = '{}/{}/{}_pred.obj'.format(folder_path, file, file)
pc = '{}/{}/{}_points.npy'.format(folder_path, file, file)
gt = '{}/{}/{}_gt.npy'.format(folder_path, file, file)

points = np.load(pc)
gt_boxes = np.load(gt)

pts = o3d.geometry.PointCloud()
pts.points = o3d.utility.Vector3dVector(points[:, :3])
pts.paint_uniform_color([0.9, 0.9, 0.9])

# pred = o3d.io.read_triangle_mesh(pred)
# pred = o3d.geometry.LineSet.create_from_triangle_mesh(pred)
# pred.paint_uniform_color([0.2, 0.9, 0.8])
# vis.add_geometry(pred)

# gt = o3d.io.read_triangle_mesh(gt_boxes)

# gt = o3d.geometry.LineSet.create_from_triangle_mesh(gt_boxes)
# gt.paint_uniform_color([0.5, 0.0, 1])



# vis.add_geometry(pts)

vis.add_geometry(gt)

vis.run()
# vis.poll_events()
# vis.update_renderer()

# vis.capture_screen_image('{}/{}/{}_draw3d.jpg'.format(folder_path, file, file))
# vis.clear_geometries()

# draw3d = '{}/{}/{}_draw3d.jpg'.format(folder_path, file, file)

# vis.destroy_window()