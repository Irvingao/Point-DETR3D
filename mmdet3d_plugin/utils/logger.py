# Copyright (c) OpenMMLab. All rights reserved.
import logging
from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO, name='mmdet3d'):
    """Get root logger and add a keyword filter to it.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


import sys, os
from collections import Counter
_log_counter = Counter()

def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"ssod"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back
        


def log_every_n(msg: str, n: int = 50, level: int = logging.DEBUG, backend="auto"):
    """
    Args:
        msg (Any):
        n (int):
        level (int):
        name (str):
    """
    caller_module, key = _find_caller()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        if isinstance(msg, dict) and (wandb is not None) and (wandb.run is not None):
            wandb.log(msg, commit=False)
        else:
            get_root_logger().log(level, msg)
