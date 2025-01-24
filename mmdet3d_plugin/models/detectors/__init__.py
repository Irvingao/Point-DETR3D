from .obj_dgcnn import ObjDGCNN
from .detr3d import Detr3D
from .point_obj_dgcnn import PointObjDGCNN
from .point_objdgcnnv2 import PointObjDGCNNV2
from .mm_point_objdgcnnv2 import MMPointObjDGCNNV2
from .mm_point_objdgcnnv3 import MMPointObjDGCNNV3
from .mm_point_objdgcnnv4 import MMPointObjDGCNNV4  # 增加load gt_box3d, 对box范围进行投影

# from .mm_point_objdgcnn_depth import MMPointObjDGCNNDepth

from .ws_student_centerpointV0 import WSStudentCenterPointV0
from .ws_student_centerpoint import WSStudentCenterPoint
from .ws_student_centerpointV2 import WSStudentCenterPointV2
from .ws_student_centerpointV3 import WSStudentCenterPointV3

__all__ = [ 'ObjDGCNN', 'Detr3D','PointObjDGCNN','PointObjDGCNNV2', 
           'MMPointObjDGCNNV2', 'MMPointObjDGCNNV3', 'MMPointObjDGCNNV4',
        #    'MMPointObjDGCNNDepth'
            'WSStudentCenterPoint', 'WSStudentCenterPointV2', 'WSStudentCenterPointV3',
            'WSStudentCenterPointV0'
           ]
