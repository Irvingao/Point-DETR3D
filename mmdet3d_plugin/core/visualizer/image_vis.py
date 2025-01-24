# Copyright (c) OpenMMLab. All rights reserved.
import copy
from random import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def project_pts_on_img(points,
                       raw_img,
                       lidar2img_rt,
                       max_distance=70,
                       thickness=-1):
    """Project the 3D points cloud on 2D image.

    Args:
        points (numpy.array): 3D points cloud (x, y, z) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        max_distance (float, optional): the max distance of the points cloud.
            Default: 70.
        thickness (int, optional): The thickness of 2D points. Default: -1.
    """
    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

 
    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]  # u, v, d

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=10,
            color=tuple(color),
            thickness=thickness,
        )
    # cv2.imwrite('/home/pc/Workspaces/Reaserch/mmdetection3d/mmdet3d/core/visualizer/project_pts_img.jpg', img)
    cv2.imshow('project_pts_img', img.astype(np.uint8))
    cv2.waitKey(0)

'''
def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)

'''
import cv2
def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    h,w,c = img.shape
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for idx, corner in enumerate(corners):
            corners[idx][0] = w if corner[0] > w else corner[0]
            corners[idx][0] = 0 if corner[0] < 0 else corner[0]
            corners[idx][1] = h if corner[1] > h else corner[1]
            corners[idx][1] = 0 if corner[1] < 0 else corner[1]
        # corners[corners[:,0] > w] = w
        # corners[corners[:,0] < 0] = 0
        # corners[corners[:,1] > h] = h
        # corners[corners[:,1] < 0] = 0
        for rect in rect_corners[i].reshape(-1):
            if rect < 0:
                continue
        print(corners.dtype)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)

def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


# TODO: remove third parameter in all functions here in favour of img_metas
def draw_depth_bbox3d_on_img(bboxes3d,
                             raw_img,
                             calibs,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`DepthInstance3DBoxes`, shape=[M, 7]):
            3d bbox in depth coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        calibs (dict): Camera calibration information, Rt and K.
        img_metas (dict): Used in coordinates transformation.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img
    from mmdet3d.models import apply_3d_transformation

    img = raw_img.copy()
    img_metas = copy.deepcopy(img_metas)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    # first reverse the data transformations
    xyz_depth = apply_3d_transformation(
        points_3d, 'DEPTH', img_metas, reverse=True)

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(xyz_depth,
                               xyz_depth.new_tensor(img_metas['depth2img']))
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_camera_bbox3d_on_img(bboxes3d,
                              raw_img,
                              cam2img,
                              img_metas,
                              color=(0, 255, 0),
                              thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    # corners_3d = bboxes3d[:, :3]
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_camera_pts3d_on_img(pts3d,
                             raw_img,
                             cam2img,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1,
                             show=False):
    """Project the 3D points on 2D plane and draw on input image.

    Args:
        pts3d (:obj:`CameraPoints`, shape=[M, 3]):
            3d points in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.bbox import points_cam2img

    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    num_pts = pts3d.shape[0]
    points_3d = pts3d.coord
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img, with_depth=True)
    uv_origin = (uv_origin - 1).round()
    
    fov_inds = ((uv_origin[:, 0] < img.shape[1])
                & (uv_origin[:, 0] >= 0)
                & (uv_origin[:, 1] < img.shape[0])
                & (uv_origin[:, 1] >= 0))

    imgfov_pts_2d = uv_origin[fov_inds, :3]  # u, v, d
    # print(f"imgfov_pts_2d | {imgfov_pts_2d}")
    max_distance=70
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        # color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=3,
            # color=tuple(color),
            color=color,
            thickness=thickness,
        )
    # cv2.imwrite('/home/pc/Workspaces/Reaserch/mmdetection3d/project_pts_img.jpg',img)
    # print('----------------------------------------------------')
    # cv2.imwrite('/home/pc/Workspaces/Reaserch/mmdetection3d/project_pts_img.jpg', img)
    if show:
        cv2.imshow('project_pts_img', img.astype(np.uint8))
        cv2.waitKey(0)
    else:
        return img
    
def draw_ego_pts3d_on_img(pts_ego3d,
                            raw_img,
                            cam2img,
                            cam2ego_trans,
                            cam2ego_rot,
                            color=(0, 255, 0),
                            thickness=1,
                            show=False):
    """Project the 3D points on 2D plane and draw on input image.

    Args:
        pts3d (list[array], shape=[M, 3]):
            3d points in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (array): Camera intrinsic matrix,
        cam2ego_trans ():
            denoted as `K` in depth bbox coordinate system.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    from mmdet3d.core.points.cam_points import CameraPoints
    if isinstance(pts_ego3d, list):
        if len(pts_ego3d) == 0:
            return raw_img
        pts_ego3d = torch.stack([torch.from_numpy(x) for x in pts_ego3d], dim=0)
    elif isinstance(pts_ego3d, np.ndarray):
        pts_ego3d = torch.tensor(pts_ego3d)
        
    if isinstance(pts_ego3d, torch.Tensor):
        if len(pts_ego3d.size()) == 3:
            if pts_ego3d.size(0) == 0:
                return raw_img
            pts_ego3d = torch.cat([*pts_ego3d],dim=0)
        num_pts, coords = pts_ego3d.size()
        if pts_ego3d.size(-1) == 2 and len(pts_ego3d.size()) == 2:
            pts_ego3d = torch.cat([pts_ego3d, pts_ego3d.new_zeros(num_pts, 1)], dim=-1)
        pts_ego3d = CameraPoints(pts_ego3d.cpu()) 
    
    
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img, dtype=np.float32))
    if not isinstance(cam2ego_trans, torch.Tensor):
        cam2ego_trans = torch.from_numpy(np.array(cam2ego_trans, dtype=np.float32))
    if not isinstance(cam2ego_rot, torch.Tensor):
        cam2ego_rot = torch.from_numpy(np.array(cam2ego_rot, dtype=np.float32))
        assert cam2ego_rot.size() == (3,3)
    from mmdet3d.core.bbox import points_cam2img

    # print(pts_ego3d)
    img = raw_img.copy()
    # ego coords to cam coords.
    pts_ego3d.translate(-cam2ego_trans)
    pts_ego3d.rotate(cam2ego_rot)

    
    cam2img = copy.deepcopy(cam2img)
    num_pts = pts_ego3d.shape[0]
    points_3d = pts_ego3d.coord
    
    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img, with_depth=True)
    uv_origin = (uv_origin - 1).round()
    
    fov_inds = ((uv_origin[:, 0] < img.shape[1])
                & (uv_origin[:, 0] >= 0)
                & (uv_origin[:, 1] < img.shape[0])
                & (uv_origin[:, 1] >= 0))

    imgfov_pts_2d = uv_origin[fov_inds, :3]  # u, v, d
    # print(f"imgfov_pts_2d | {imgfov_pts_2d}")
    max_distance=70
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pts_2d[i, 2]
        if imgfov_pts_2d[i, 0] > 1600 or imgfov_pts_2d[i, 0] < 0:
            continue
        if imgfov_pts_2d[i, 1] > 900 or imgfov_pts_2d[i, 1] < 0:
            continue
        if depth > max_distance:
            continue
        # color = cmap[np.clip(int(max_distance * 10 / depth), 0, 255), :]
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=thickness,
            # color=tuple(color),
            color=color,
            thickness=thickness,
        )
    return img



def draw_ego_pts3d_on_bev(pts_ego3d,
                          bev_img,
                          scale_rate=0.15,
                          color=(0, 255, 0),
                          thickness=1):
    if isinstance(pts_ego3d, list):
        if len(pts_ego3d) == 0:
            return bev_img
        if not isinstance(pts_ego3d[0], np.ndarray):
            if isinstance(pts_ego3d[0], torch.Tensor):
                new_pts_ego3d = [pts.numpy() for pts in pts_ego3d]
                pts_ego3d = new_pts_ego3d
    if isinstance(pts_ego3d, torch.Tensor):
        if len(pts_ego3d.size()) == 3:
            if pts_ego3d.size(0) == 0:
                return bev_img
            else:
                new_pts_ego3d = [pts.cpu().numpy() for pts in pts_ego3d]
                pts_ego3d = new_pts_ego3d
        elif len(pts_ego3d.size()) == 2:
            pts_ego3d = [pts_ego3d.cpu().numpy()]

    
    # tensor list(array[pts, 2])
    for inst_pts in pts_ego3d:
        inst_pts[:,1] = inst_pts[:,1] + 7.5
        inst_pts = inst_pts / scale_rate    # 放缩到BEV尺度
        bev_img = cv2.polylines(bev_img, [inst_pts.astype(np.int)], False, color, thickness)
        # bev_img = cv2.polylines(bev_img, [np.array(inst_pts, dtype=np.int8)], False, color, thickness)
        # for idx in range(len(inst_pts)-1):
        #     bev_img = cv2.line(bev_img, (int(inst_pts[idx][0]), int(inst_pts[idx][1])),
        #              (int(inst_pts[idx+1][0]), int(inst_pts[idx+1][1])), color, thickness,
        #              cv2.LINE_AA)
    return bev_img



import matplotlib.pyplot as plt


def draw_pts_on_bev_matplotlib(img, vec, cls=None, save_img=True):

    plt.figure()
    
    plt.subplot(1,2,1)
    '''
    car_img = get_car_icon()
    car_img = np.rot90(car_img.astype(np.uint8), k=-θ/(np.pi/2))
    if θ/(np.pi/2) == 0:
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
    else:
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
    '''
    if cls:
        cls = int(cls)
        if cls == 0:
            color = 'g'
        elif cls == 1:
            color = 'b'
        elif cls == 2:
            color = 'r'
    if isinstance(vec, list):
        if len(vec) < 1:
            return 
        if isinstance(vec[0], dict):
            # show all pts in dataset
            for pts_dict in vec:
                pts = np.array(pts_dict['pts_ego3d'])
                cls = int(pts_dict['class_id'])
                # if len(pts) != 20 and cls == 2:
                    # print(f"------------- {pts}")
                if cls == 0:
                    color = 'g'
                elif cls == 1:
                    color = 'b'
                elif cls == 2:
                    color = 'r'
                
                x_values = pts[:,0] 
                y_values = pts[:,1] 
                # 画出起始点
                plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
                
                plt.scatter(x_values[1:], y_values[1:],c=color)
                plt.plot(x_values, y_values,color=color)
    elif isinstance(vec, torch.Tensor):
        # show all pred pts 
        # torch.Size([4, 20, 2])
        num_instance = vec.size(0)
        for idx in range(num_instance):
            pts = vec[idx].numpy()
            x_values = pts[:,0] 
            y_values = pts[:,1] 
            plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
            plt.scatter(x_values[1:], y_values[1:],c=color)
            plt.plot(x_values, y_values,color=color)
    elif isinstance(vec, list) and isinstance(vec[0], np.ndarray):
        # show all gt pts 
        for pts in vec:
            x_values = pts[:,0] 
            y_values = pts[:,1] 
            plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
            plt.scatter(x_values[1:], y_values[1:],c=color)
            plt.plot(x_values, y_values,color=color)
    plt.subplot(1,2,2)
    
    if not isinstance(img, list):
        img = [img]
    if len(img) == 6:
        surr_img_top = cv2.hconcat(img[0:3])    # 水平拼接
        surr_img_btm = cv2.hconcat(img[3:6])    # 水平拼接
        surr_img = cv2.vconcat([surr_img_top, surr_img_btm])
        surr_img = cv2.cvtColor(surr_img, cv2.COLOR_RGB2BGR)
        plt.imshow(surr_img)
    else:
        img[0] = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
        plt.imshow(img[0])
    
    if save_img:
        plt.savefig('bev_pred.png')
    else:
        plt.show()
    # plt.show(block=False)
    # plt.pause(0.1)



def draw_img_pts(img, imgfov_pts_2d, thickness=3, color=[0,255,0]):
    if isinstance(imgfov_pts_2d, torch.Tensor):
        imgfov_pts_2d = imgfov_pts_2d.detach().cpu().numpy()
    for i in range(imgfov_pts_2d.shape[0]):
        if imgfov_pts_2d[i, 0] > 1600 or imgfov_pts_2d[i, 0] < 0:
            continue
        if imgfov_pts_2d[i, 1] > 900 or imgfov_pts_2d[i, 1] < 0:
            continue
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=thickness,
            color=color,
            thickness=thickness,
        )
    return img


def draw_instance_on_bev(img, gt_insts, pred_insts, save_img=True, img_id=None, transpose=False):
    
    color = ['g', 'b', 'r']
    # if img_id == 0:
    plt.figure()
    
    '''
    car_img = get_car_icon()
    car_img = np.rot90(car_img.astype(np.uint8), k=-θ/(np.pi/2))
    if θ/(np.pi/2) == 0:
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
    else:
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
    '''
    plt.subplot(1,3,1)
    # show all pred pts [tensor,tensor,tensor]
    # torch.Size([4, 20, 2])
    for cls, vec in enumerate(pred_insts):
        num_instance = vec.size(0)
        for idx in range(num_instance):
            pts = vec[idx].numpy()
            if transpose:
                # 将x, y 互换显示
                x_values = pts[:,1] 
                y_values = pts[:,0] 
            else:
                x_values = pts[:,0] 
                y_values = pts[:,1] 
            plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
            plt.scatter(x_values[1:], y_values[1:],c=color[cls])
            plt.plot(x_values, y_values,color=color[cls])
            plt.grid(True)  # 显示网格线

    plt.subplot(1,3,2)
    # show all gt pts [np.array, np.array, np.array]
    for cls, vec in enumerate(gt_insts):
        for pts in vec:
            if transpose:
                # 将x, y 互换显示
                x_values = pts[:,1] 
                y_values = pts[:,0] 
            else:
                x_values = pts[:,0] 
                y_values = pts[:,1] 
            plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
            plt.scatter(x_values[1:], y_values[1:],c=color[cls])
            plt.plot(x_values, y_values,color=color[cls])
            plt.grid(True)  # 显示网格线
    
    plt.subplot(1,3,3)
    if not isinstance(img, list):
        img = [img]
    if len(img) == 6:
        surr_img_top = cv2.hconcat(img[0:3])    # 水平拼接
        surr_img_btm = cv2.hconcat(img[3:6])    # 水平拼接
        surr_img = cv2.vconcat([surr_img_top, surr_img_btm])
        surr_img = cv2.cvtColor(surr_img, cv2.COLOR_RGB2BGR)
        plt.imshow(surr_img)
    else:
        img[0] = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
        plt.imshow(img[0])
    
    if save_img:
        if img_id:
            plt.savefig(f'work_dirs/bev_pred_{img_id}.png')
        else:
            plt.savefig('bev_pred.png')
    else:
        plt.show()
    # plt.show(block=False)
    # plt.pause(0.1)
    plt.close("all")
    

def draw_instance_on_bev(img, gt_insts, pred_insts, save_img=True, img_id=None, transpose=False):
    
    color = ['g', 'b', 'r']
    # if img_id == 0:
    plt.figure()
    
    '''
    car_img = get_car_icon()
    car_img = np.rot90(car_img.astype(np.uint8), k=-θ/(np.pi/2))
    if θ/(np.pi/2) == 0:
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
    else:
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
    '''
    plt.subplot(1,3,1)
    # show all pred pts [tensor,tensor,tensor]
    # torch.Size([4, 20, 2])
    for cls, vec in enumerate(pred_insts):
        num_instance = vec.size(0)
        for idx in range(num_instance):
            pts = vec[idx].numpy()
            if transpose:
                # 将x, y 互换显示
                x_values = pts[:,1] 
                y_values = pts[:,0] 
            else:
                x_values = pts[:,0] 
                y_values = pts[:,1] 
            plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
            plt.scatter(x_values[1:], y_values[1:],c=color[cls])
            plt.plot(x_values, y_values,color=color[cls])
            plt.grid(True)  # 显示网格线

    plt.subplot(1,3,2)
    # show all gt pts [np.array, np.array, np.array]
    for cls, vec in enumerate(gt_insts):
        for pts in vec:
            if transpose:
                # 将x, y 互换显示
                x_values = pts[:,1] 
                y_values = pts[:,0] 
            else:
                x_values = pts[:,0] 
                y_values = pts[:,1] 
            plt.scatter(x_values[0:1], y_values[0:1],c='#00FF00')
            plt.scatter(x_values[1:], y_values[1:],c=color[cls])
            plt.plot(x_values, y_values,color=color[cls])
            plt.grid(True)  # 显示网格线
    
    plt.subplot(1,3,3)
    if not isinstance(img, list):
        img = [img]
    if len(img) == 6:
        surr_img_top = cv2.hconcat(img[0:3])    # 水平拼接
        surr_img_btm = cv2.hconcat(img[3:6])    # 水平拼接
        surr_img = cv2.vconcat([surr_img_top, surr_img_btm])
        surr_img = cv2.cvtColor(surr_img, cv2.COLOR_RGB2BGR)
        plt.imshow(surr_img)
    else:
        img[0] = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
        plt.imshow(img[0])
    
    if save_img:
        if img_id:
            plt.savefig(f'work_dirs/bev_pred_{img_id}.png')
        else:
            plt.savefig('bev_pred.png')
    else:
        plt.show()
    # plt.show(block=False)
    # plt.pause(0.1)
    plt.close("all")

# 

def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

def draw_pts_on_img(img, imgfov_pts_2d, thickness=3, color=[0,255,0]):
    if isinstance(imgfov_pts_2d, torch.Tensor):
        imgfov_pts_2d = imgfov_pts_2d.detach().cpu().numpy()
    
    h,w,c = img.shape
    for i in range(imgfov_pts_2d.shape[0]):
        if imgfov_pts_2d[i, 0] > w or imgfov_pts_2d[i, 0] < 0:
            continue
        if imgfov_pts_2d[i, 1] > h or imgfov_pts_2d[i, 1] < 0:
            continue
        cv2.circle(
            img,
            center=(int(np.round(imgfov_pts_2d[i, 0])),
                    int(np.round(imgfov_pts_2d[i, 1]))),
            radius=thickness,
            color=color,
            thickness=thickness,
        )
    return img


import matplotlib.pyplot as plt
import torch

def visualize_bevbox2d_corners(bevbox2d_corners):
    """
    可视化BEV Box 2D的角点
    
    bevbox2d_corners: Tensor with shape [bs, num_box, 5], XYWHR format, value range: [-1,1]
    """
    # 从XYWHR格式转换成角点坐标格式
    def xywhr_to_corners(xywhr):
        xy = xywhr[..., 0:2]
        w = xywhr[..., 2:3]
        h = xywhr[..., 3:4]
        r = xywhr[..., 4:5]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        corners = torch.zeros_like(xywhr.repeat(1,1,2))[...,:8]
        corners[..., 0:2] = xy + 0.5 * torch.cat([w, h], dim=-1) #* torch.cat([cos_r, -sin_r], dim=-1)
        corners[..., 2:4] = xy + 0.5 * torch.cat([w, h], dim=-1) #* torch.cat([sin_r, cos_r], dim=-1)
        corners[..., 4:6] = xy - 0.5 * torch.cat([w, h], dim=-1) #* torch.cat([cos_r, -sin_r], dim=-1)
        corners[..., 6:8] = xy - 0.5 * torch.cat([w, h], dim=-1) #* torch.cat([sin_r, cos_r], dim=-1)
        return corners

    # 转换成角点坐标格式
    corners = xywhr_to_corners(bevbox2d_corners)    # [1, 26, 8]
    
    corners = corners.cpu()

    # 绘制图像
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(bevbox2d_corners.shape[0]):
        for j in range(bevbox2d_corners.shape[1]):
            # 绘制框的角点
            x = corners[i, j, [0, 2, 4, 6, 0]]
            y = corners[i, j, [1, 3, 5, 7, 1]]
            ax.plot(x, y, color='r')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.savefig('bev_corner.png')
    # plt.show()