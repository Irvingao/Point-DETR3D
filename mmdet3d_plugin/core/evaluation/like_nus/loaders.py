# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import json
from typing import Dict, Tuple

import numpy as np
import tqdm
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes


# 传入自车ego-pose代替nusc
def add_center_dist(e2g_path: str,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    with open(e2g_path) as f:
        data = json.load(f)

    for sample_token in eval_boxes.sample_tokens:
        for box in eval_boxes[sample_token]:
            pose_record = data[sample_token]
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record[0],
                               box.translation[1] - pose_record[1],
                               box.translation[2] - pose_record[2])
            if isinstance(box, DetectionBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes


def filter_eval_boxes(eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float], 
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)
    
    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])
        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])
    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        # print("val:", val)
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
