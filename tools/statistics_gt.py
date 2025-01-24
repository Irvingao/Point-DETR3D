import mmcv
import numpy as np
from mmdet3d.core.points import BasePoints, get_points_type
import torch
from mmdet3d.core.bbox import box_np_ops as box_np_ops

import matplotlib.pyplot as plt


print("loading pkl...")
input_path = "data/nuscenes/nuscenes_infos_train.pkl"

data = mmcv.load(input_path)

data_info = data['infos']

label_wlh = {'car':[],'truck':[],'construction_vehicle':[],'bus':[],'trailer':[],'barrier':[],'motorcycle':[],'bicycle':[],'pedestrian':[],'traffic_cone':[]}

print("statistics start.")
for info_id in range(len(data_info)):
    # print(info_id)
    mask = data_info[info_id]['valid_flag']
    gt_bboxes_3d = data_info[info_id]['gt_boxes'][mask]
    gt_names_3d = data_info[info_id]['gt_names'][mask]

    for gt_id in range(gt_names_3d.shape[0]):

        gt_name = gt_names_3d[gt_id]
        wlh = np.expand_dims(gt_bboxes_3d[gt_id][3:6],axis=0)
        if gt_name in label_wlh.keys():
            label_wlh[gt_name].append(wlh)


from tabulate import tabulate
headers = ["category", "w(mean)", "w(std)", "l(mean)", "l(std)", "h(mean)", "h(std)"]
# data = [["Alice", 25, "Female"], ["Bob", 30, "Male"], ["Charlie", 35, "Male"], ["David", 40, "Male"]]
table_data = []

data_dict = {}

for key in label_wlh:
    
    wlh = label_wlh[key]
    wlh = np.concatenate(wlh,axis=0)

    w_mean = wlh[:,0].mean()
    w_std = wlh[:,0].std()
    l_mean = wlh[:, 1].mean()
    l_std = wlh[:, 1].std()
    h_mean = wlh[:, 2].mean()
    h_std = wlh[:, 2].std()
    
    table_data.append([key, w_mean, w_std, l_mean, l_std, h_mean, h_std])
    
    data_dict[key] = [round(w_mean, 2), round(w_std, 2), round(l_mean, 2), round(l_std, 2), round(h_mean, 2), round(h_std, 2)]
    # plt.hist(wlh[:,0])
    # plt.title("%s w_mean:%.2f,w_std:%.2f" % (key,w_mean,w_std))
    # plt.savefig('pic-{}-w.png'.format(key))
    # plt.cla()
    
    # plt.hist(wlh[:,1])
    # plt.title("%s l_mean:%.2f,l_std:%.2f" % (key,l_mean,l_std))
    # plt.savefig('pic-{}-l.png'.format(key))
    # plt.cla()
    
    # r = np.maximum(wlh[:, 0], wlh[:, 1])
    # r_mean = r.mean()
    # r_std = r.std()
    # plt.hist(r)
    # plt.title("%s r_mean:%.2f,r_std:%.2f" % (key, r_mean, r_std))
    # plt.savefig('img/pic-{}-10%-r.png'.format(key))
    # plt.cla()

    # plt.hist(wlh[:,2])
    # plt.title("%s h_mean:%.2f,h_std:%.2f" % (key,h_mean,h_std))
    # plt.savefig('pic-{}-10%-h.png'.format(key))
    # plt.cla()
    
print(data_dict)

print(tabulate(table_data, headers=headers, floatfmt=".3f", numalign="right", stralign="center", tablefmt="grid"))


'''
{'car': [1.96, 0.19, 4.63, 0.47, 1.74, 0.25], 
'truck': [2.52, 0.45, 6.94, 2.11, 2.85, 0.84], 
'construction_vehicle': [2.82, 1.06, 6.56, 3.17, 3.2, 0.94], 
'bus': [2.95, 0.32, 11.19, 2.06, 3.49, 0.49], 
'trailer': [2.92, 0.55, 12.28, 4.6, 3.87, 0.77], 
'barrier': [2.51, 0.62, 0.5, 0.17, 0.99, 0.15], 
'motorcycle': [0.77, 0.16, 2.11, 0.31, 1.46, 0.23], 
'bicycle': [0.61, 0.16, 1.7, 0.25, 1.3, 0.35], 
'pedestrian': [0.67, 0.14, 0.73, 0.19, 1.77, 0.19], 
'traffic_cone': [0.41, 0.14, 0.42, 0.15, 1.08, 0.27]}
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|       category       |   w(mean) |   w(std) |   l(mean) |   l(std) |   h(mean) |   h(std) |
+======================+===========+==========+===========+==========+===========+==========+
|         car          |     1.960 |    0.188 |     4.634 |    0.469 |     1.738 |    0.250 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|        truck         |     2.518 |    0.449 |     6.936 |    2.108 |     2.851 |    0.839 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
| construction_vehicle |     2.825 |    1.059 |     6.562 |    3.172 |     3.198 |    0.943 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|         bus          |     2.950 |    0.323 |    11.195 |    2.059 |     3.492 |    0.487 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|       trailer        |     2.923 |    0.547 |    12.276 |    4.596 |     3.871 |    0.772 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|       barrier        |     2.508 |    0.620 |     0.497 |    0.166 |     0.989 |    0.150 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|      motorcycle      |     0.773 |    0.161 |     2.107 |    0.305 |     1.464 |    0.228 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|       bicycle        |     0.608 |    0.163 |     1.701 |    0.252 |     1.303 |    0.352 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|      pedestrian      |     0.669 |    0.139 |     0.729 |    0.190 |     1.770 |    0.188 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
|     traffic_cone     |     0.410 |    0.137 |     0.416 |    0.149 |     1.077 |    0.268 |
+----------------------+-----------+----------+-----------+----------+-----------+----------+
'''






