from .dgcnn3d_head import DGCNN3DHead
from .detr3d_head import Detr3DHead
from .point_dgcnn3d_head import PointDGCNN3DHead
# from .point_dgcnn_headV2 import PointDGCNN3DHeadV2
from .h_point_dgcnn_headV2 import HPointDGCNN3DHeadV2
from .point_dgcnn_head_nomatch import PointDGCNN3DHead_Nomatch

from .group_point_dgcnn_head import GropuPointDGCNN3DHead
from .h_group_point_dgcnn_head import HGropuPointDGCNN3DHead

from .mm_point_dgcnn_headV2 import MMPointDGCNN3DHeadV2
from .mm_point_dgcnn_headV3 import MMPointDGCNN3DHeadV3
from .mm_point_dgcnn_headV4 import MMPointDGCNN3DHeadV4

# ws student
from .ws_centerpoint_head import WSCenterHead

# sp centerpoint
from .sp_centerpoint_head import SPCenterHead

__all__ = ['DGCNN3DHead', 'Detr3DHead','PointDGCNN3DHead',
           'PointDGCNN3DHead_Nomatch','GropuPointDGCNN3DHead','HGropuPointDGCNN3DHead',
           'MMPointDGCNN3DHeadV2', 'MMPointDGCNN3DHeadV3', 'MMPointDGCNN3DHeadV4',
           'WSCenterHead', 'SPCenterHead']