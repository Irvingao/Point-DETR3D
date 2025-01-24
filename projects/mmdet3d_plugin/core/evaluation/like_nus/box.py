# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2019.

import abc
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np


EvalBoxType = Union['DetectionBox']

class EvalBoxes:
    """ Data class that groups EvalBox instances by sample. """

    def __init__(self):
        """
        Initializes the EvalBoxes for GT or predictions.
        """
        self.boxes = defaultdict(list)

    def __repr__(self):
        return "EvalBoxes with {} boxes across {} samples".format(len(self.all), len(self.sample_tokens))

    def __getitem__(self, item) -> List[EvalBoxType]:
        return self.boxes[item]

    def __eq__(self, other):
        if not set(self.sample_tokens) == set(other.sample_tokens):
            return False
        for token in self.sample_tokens:
            if not len(self[token]) == len(other[token]):
                return False
            for box1, box2 in zip(self[token], other[token]):
                if box1 != box2:
                    return False
        return True

    def __len__(self):
        return len(self.boxes)

    @property
    def all(self) -> List[EvalBoxType]:
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        """ Returns a list of all keys. """
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[EvalBoxType]) -> None:
        """ Adds a list of boxes. """
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        """
        Initialize from serialized content.
        :param content: A dictionary with the serialized content of the box.
        :param box_cls: The class of the boxes, DetectionBox or TrackingBox.
        """
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
        return eb

