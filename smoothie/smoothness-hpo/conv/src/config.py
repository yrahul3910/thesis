from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    n_filters: int = 3
    kernel_size: int = 3
    padding: Literal["valid", "same"] = "valid"
    n_blocks: int = 2
    featurewise_center: bool = False
    samplewise_center: bool = False
    featurewise_std_normalization: bool = False
    samplewise_std_normalization: bool = False
    rotation_range: int = 0
    width_shift_range: float = 0.
    height_shift_range: float = 0.
    zoom_range: float = 0.
    horizontal_flip: bool = False
    vertical_flip: bool = False
    