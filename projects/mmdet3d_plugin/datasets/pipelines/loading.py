
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import torch

from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile,LoadPointsFromMultiSweeps
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox

@PIPELINES.register_module()
class LoadPointAnnotations(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        point_noise (bool): Whether to use shifted noise. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 point_noise=False,
                 noise_density=1/3,
                 norm_type="truncated_normal",
                 norm_mean_std=[0.5, 0.3],
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.point_noise = point_noise
        self.noise_density = noise_density
        self.norm_type = norm_type
        assert norm_type in ['truncated_normal', 'uniform'], f"`norm_type` is \
            not supported as {norm_type}"
        
        self.mean, self.std_dev = norm_mean_std
        # mean = 0.5  # 截断正态分布的均值
        # std_dev = 0.1  # 截断正态分布的标准差
        
        if point_noise:
            assert noise_density > 0., "`noise_density` should be greater \
                than 0., else set `point_noise` to `False`."
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        point_coord = results['gt_bboxes_3d'].gravity_center

        if self.point_noise:
            # 生成一个 缩小noise_density倍的长方体，用随机数×该长方体边长/2, 生成noise
            dims = results['gt_bboxes_3d'].dims / 2  # [n, 3]  w, l, h
            # 生成 [0,1] 随机数 符合分布规律
            if self.norm_type == 'uniform':
                rand_noise_scale = np.random.rand(dims.size(0), 3)
            elif self.norm_type == 'truncated_normal':
                # 使用nn.init.trunc_normal函数生成符合截断正态分布的随机数矩阵
                rand_noise_scale = torch.nn.init.trunc_normal_(
                    torch.empty(dims.size(0), 3), 
                    mean=self.mean, std=self.std_dev, a=0, b=1).numpy()
            # [0,1] -> [-1,1]
            rand_noise_scale = rand_noise_scale * 2 - 1
            assert (rand_noise_scale.any() > 1 and 
                    rand_noise_scale.any() < -1) == False
            # [-1,1] * dims/2 为offset， * noise_density 为再缩放一次尺度
            noise = dims * rand_noise_scale * self.noise_density
            # 以center为中心点
            point_coord += noise

        labels = results['gt_labels_3d']
        point_ann = {}
        results['point_coord'] = point_coord
        results['labels'] = labels
        # results['point_ann'] = point_ann

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
    

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesV2(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, 
                 color_type='unchanged',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_imgs(self, imgs_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        # print(f"imgs_filename: {imgs_filename}")
        imgs = []
        for img_filename in imgs_filename:
            img_bytes = self.file_client.get(img_filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, channel_order=self.channel_order)
            imgs.append(img)    
        imgs = np.stack(
            imgs, axis=-1)
        
        # print(f"imgs: {imgs.shape}")
        return imgs

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        
        # img is of shape (h, w, c, num_views)
        img = self._load_imgs(filename)
        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str    


from mmdet3d.core.points import LiDARPoints

@PIPELINES.register_module()
class FilterEnvPoints(object):
    """
    Get the original point cloud in the area near gt
    """

    def __init__(self,
                 ):

        # 每个类别的按顺序依次是 hmin,hmid,hmax  rmin,rmid,rmax
        label_h_r = {'car': [1.51,1.74,2.43,4.18,4.59,5.41], 'barrier': [0.83,1.07,1.43, 1.17,2.10,3.03], 'bicycle': [0.6,1.26,1.92, 1.17,1.73,2.57],'bus': [3.08,3.42,4.1,7.44,11.12,16.64],
                     'construction_vehicle': [2.28,3.17,4.95,3.29,6.59,16.49], 'motorcycle': [1.24,1.43,1.81,1.58,2.12,2.93], 'pedestrian': [1.51,1.72,2.35, 0.59,0.76,1.27],
                     'traffic_cone': [0.65,0.93,1.77, 0.31,0.42,0.75], 'trailer': [3.25,4.04,4.83, 10.02,13.58,17.14], 'truck': [2.23,2.99,4.51,4.82,7.13,14.06]}

        gtlabels2names = {0: 'car', 1: 'truck', 2: 'construction_vehicle', 3: 'bus', 4: 'trailer', 5: 'barrier',
                          6: 'motorcycle', 7: 'bicycle',
                          8: 'pedestrian', 9: 'traffic_cone'}
        self.label_h_r = label_h_r
        self.gtlabels2names = gtlabels2names

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        gtlabels2names = self.gtlabels2names
        label_h_r = self.label_h_r
        

        labels = results['labels']
        point_coords = results['point_coord']

        num_gt = point_coords.shape[0]
        all_proposals = []
        all_sampled_points = []

        points = results['points']

        for gt_id in range(num_gt):
            label = gtlabels2names[labels[gt_id]]
            # 确定圆柱体的高和半径，对于不同的类别，
            hmin = label_h_r[label][0]
            hmid = label_h_r[label][1]
            hmax = label_h_r[label][2]
            hmin = torch.tensor([hmin], dtype=torch.float32)
            hmid = torch.tensor([hmid], dtype=torch.float32)
        
        
        for gt_id in range(num_gt):
            label = gtlabels2names[labels[gt_id]]
            # 确定圆柱体的高和半径，对于不同的类别，
            hmin = label_h_r[label][0]
            hmid = label_h_r[label][1]
            hmax = label_h_r[label][2]
            hmin = torch.tensor([hmin], dtype=torch.float32)
            hmid = torch.tensor([hmid], dtype=torch.float32)
            hmax = torch.tensor([hmax], dtype=torch.float32)
            rmin = label_h_r[label][3]
            rmid = label_h_r[label][4]
            rmax = label_h_r[label][5]
            rmin = torch.tensor([rmin], dtype=torch.float32)
            rmid = torch.tensor([rmid], dtype=torch.float32)
            rmax = torch.tensor([rmax], dtype=torch.float32)
            angle = torch.tensor([0.0],dtype=torch.float32)
            # get pseudo box
            # pseudo_bbox_min = torch.cat((point_coords[gt_id], rmin, rmin, hmin, angle), dim=0).unsqueeze(0)
            # pseudo_bbox_mid = torch.cat((point_coords[gt_id], rmid, rmid, hmid, angle), dim=0).unsqueeze(0)
            pseudo_bbox_max = torch.cat((point_coords[gt_id], rmax, rmax, hmax, angle), dim=0).unsqueeze(0)
            # pseudo_bbox = torch.cat((pseudo_bbox_min,pseudo_bbox_mid,pseudo_bbox_max),dim=0)    # box (3, 7)
            pseudo_bbox = pseudo_bbox_max
            # get pts in pseudo box
            point_indices = points_in_rbbox(points.tensor.numpy(), pseudo_bbox.numpy(), origin=(0.5, 0.5, 0.5))
            # sampled_points_min = points[point_indices[:, 0]].tensor
            # sampled_points_mid = points[point_indices[:, 1]].tensor
            # sampled_points_max = points[point_indices[:, 2]].tensor
            
            # sampled_points = torch.stack((sampled_points_min[:,:3],sampled_points_mid[:,:3],sampled_points_max[:,:3]),dim=0)
            
            sampled_points = points[point_indices[:, 0]].tensor

            all_sampled_points.append(sampled_points)
            all_proposals.append(pseudo_bbox)

        # all_sampled_points = torch.stack(all_sampled_points,dim=0)
        all_sampled_points = torch.cat(all_sampled_points,dim=0)
        
        # all_proposals = torch.stack(all_proposals,dim=0)
        all_proposals = torch.cat(all_proposals,dim=0)
        
        results['all_sampled_points'] = all_sampled_points
        results['all_proposals'] = all_proposals
        
        all_sampled_points = LiDARPoints(all_sampled_points, 
                    points_dim=all_sampled_points.shape[-1])
        
        results['points'] = all_sampled_points
        return results