import numpy as np
 
def gradient_point_cloud_color_map(points):
    # 根据距离生成色彩
    colors = np.zeros([points.shape[0], 3])
    
    dist = np.sqrt(np.square(points[:,0]) + np.square(points[:,1]))
    
    dist_max = np.max(dist)
    # dist_min = np.min(dist)
    print(f"dist_max: {dist_max}")
    # 调整渐变半径
    dist = dist / 51.2
    dist = dist / 1.3
    
    # RGB
    min = [127,0,255]   # 紫色
    # mid = []
    max = [255,255,0]   # 黄色
    
    # 最近处为紫色
    # colors[:,0] = 127
    # colors[:,2] = 255
    
    # 减R(127 -> 0),加G(0->255),再减B(255->0)，再加R(0 -> 255)
    # 127+255+255+255
    all_color_value = 127+255+255+255
    dist_color = dist * all_color_value
    # if dist_color < 127:
    
    # 减R (127 -> 0)
    clr_1 = 127
    dy_r = 127-dist_color
    tmp = np.zeros([colors[dist_color<clr_1].shape[0], 3])
    tmp[:, 0] = dy_r[dist_color<clr_1]
    tmp[:, 1] = 0
    tmp[:, 2] = 255
    colors[dist_color<clr_1] = tmp
    
    # 加G (0->255)
    clr_2 = 127+255
    dy_g = dist_color-clr_1
    tmp = np.zeros([colors[(dist_color>=clr_1) & (dist_color<clr_2)].shape[0], 3])
    tmp[:, 0] = 0
    tmp[:, 1] = dy_g[(dist_color>=clr_1) & (dist_color<clr_2)]
    tmp[:, 2] = 255
    colors[(dist_color>=clr_1) & (dist_color<clr_2)] = tmp
    
    # 减B (255->0)
    clr_3 = 127+255+255
    dy_b = dist_color-clr_2
    tmp = np.zeros([colors[(dist_color>=clr_2) & (dist_color<clr_3)].shape[0], 3])
    tmp[:, 0] = 0
    tmp[:, 1] = 255
    tmp[:, 2] = dy_b[(dist_color>=clr_2) & (dist_color<clr_3)]
    colors[(dist_color>=clr_2) & (dist_color<clr_3)] = tmp
    
    # 加R(0 -> 255)
    clr_4 = 127+255+255+255
    dy_r = dist_color-clr_3
    tmp = np.zeros([colors[(dist_color>=clr_3) & (dist_color<clr_4)].shape[0], 3])
    tmp[:, 0] = dy_r[(dist_color>=clr_3) & (dist_color<clr_4)]
    tmp[:, 1] = 255
    tmp[:, 2] = 0
    colors[(dist_color>=clr_3) & (dist_color<clr_4)] = tmp
    
    '''
    '''
    # 外围都为黄色
    tmp = np.zeros([colors[dist_color>clr_4].shape[0], 3])
    tmp[:, 0] = 255
    tmp[:, 1] = 255
    tmp[:, 2] = 0
    colors[dist_color>clr_4] = tmp
    
    print(f"color: {colors.all() == 0.}")
    points = np.concatenate((points[:,:3], colors),axis=1)
    print(points.shape)

    return points

def circle_distance_point_cloud_color_map(points):
    # 根据距离生成色彩
    colors = np.zeros([points.shape[0], 3])
    
    dist = np.sqrt(np.square(points[:,0]) + np.square(points[:,1]))
    
    dist_max = np.max(dist)
    # dist_min = np.min(dist)
    print(f"dist_max: {dist_max}")
    # 调整距离半径
    # dist = dist / 51.2
    # dist = dist / 2
    
    # RGB
    # 最近处为紫色
    dist_color = [
        [127,0,127],    # 紫
        [0,0,255],      # 蓝
        [0,255,0],      # 绿
        [255,255,0],    # 黄
        [255,140,0],    # 橙
    ]
    
    dist_1 = 10
    tmp = np.zeros([colors[dist<dist_1].shape[0], 3])
    tmp[:, 0] = dist_color[0][0]
    tmp[:, 1] = dist_color[0][1]
    tmp[:, 2] = dist_color[0][2]
    colors[dist<dist_1] = tmp
    
    # 10-20
    dist_2 = 20
    tmp = np.zeros([colors[(dist>=dist_1) & (dist<dist_2)].shape[0], 3])
    tmp[:, 0] = dist_color[1][0]
    tmp[:, 1] = dist_color[1][1]
    tmp[:, 2] = dist_color[1][2]
    colors[(dist>=dist_1) & (dist<dist_2)] = tmp
    
    # 20-30
    dist_3 = 30
    tmp = np.zeros([colors[(dist>=dist_2) & (dist<dist_3)].shape[0], 3])
    tmp[:, 0] = dist_color[2][0]
    tmp[:, 1] = dist_color[2][1]
    tmp[:, 2] = dist_color[2][2]
    colors[(dist>=dist_2) & (dist<dist_3)] = tmp
    
    # 30-40
    dist_4 = 40
    tmp = np.zeros([colors[(dist>=dist_3) & (dist<dist_4)].shape[0], 3])
    tmp[:, 0] = dist_color[3][0]
    tmp[:, 1] = dist_color[3][1]
    tmp[:, 2] = dist_color[3][2]
    colors[(dist>=dist_3) & (dist<dist_4)] = tmp
    
    '''
    '''
    # 外围都为黄色
    tmp = np.zeros([colors[dist>dist_4].shape[0], 3])
    tmp[:, 0] = dist_color[4][0]
    tmp[:, 1] = dist_color[4][1]
    tmp[:, 2] = dist_color[4][2]
    colors[dist>dist_4] = tmp
    
    print(f"color: {colors.all() == 0.}")
    points = np.concatenate((points[:,:3], colors),axis=1)
    print(points.shape)

    return points


