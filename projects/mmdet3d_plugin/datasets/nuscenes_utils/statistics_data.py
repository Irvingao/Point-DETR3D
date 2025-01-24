
gtlabels2names = {-1: 'car',
    0: 'car', 1: 'truck', 2: 'construction_vehicle', 
    3: 'bus', 4: 'trailer', 5: 'barrier', 6: 'motorcycle', 
    7: 'bicycle', 8: 'pedestrian', 9: 'traffic_cone'}

# [w_mean, w_std, l_mean, l_std, h_mean, h_std]
dict_wlh = {
    'car': [1.96, 0.19, 4.63, 0.47, 1.74, 0.25], 
    'truck': [2.52, 0.45, 6.94, 2.11, 2.85, 0.84], 
    'construction_vehicle': [2.82, 1.06, 6.56, 3.17, 3.2, 0.94], 
    'bus': [2.95, 0.32, 11.19, 2.06, 3.49, 0.49], 
    'trailer': [2.92, 0.55, 12.28, 4.6, 3.87, 0.77], 
    'barrier': [2.51, 0.62, 0.5, 0.17, 0.99, 0.15], 
    'motorcycle': [0.77, 0.16, 2.11, 0.31, 1.46, 0.23], 
    'bicycle': [0.61, 0.16, 1.7, 0.25, 1.3, 0.35], 
    'pedestrian': [0.67, 0.14, 0.73, 0.19, 1.77, 0.19], 
    'traffic_cone': [0.41, 0.14, 0.42, 0.15, 1.08, 0.27]}

color_dict = {
    'car':                  [0  , 0.5, 0 ],     # 绿色
    'truck':                [0.75, 0.25, 0.75], # 紫粉
    'construction_vehicle': [0.5, 0.5, 1.],     # 浅紫
    'bus':                  [0.25, 0.10, 0.92], # 深蓝
    'trailer':              [0.93, 0.78, 0.05], # 深黄
    'barrier':              [0.93, 0.46, 0.31], # 橘黄
    'motorcycle':           [0.10, 0.79, 0.59], # 蓝绿
    'bicycle':              [0.49, 0.49, 0.72], # 蓝灰
    'pedestrian':           [0.54, 0., 0.],     # 深红
    'traffic_cone':         [0.12, 0.67, 0.25], # 青绿
}
