# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from mmdet.models.builder import BACKBONES as PE

MODELS = Registry('models', parent=MMCV_MODELS)

POINT_ENCODER = MODELS


def build_point_encoder(cfg):
    """Build fusion layer."""
    return POINT_ENCODER.build(cfg)
