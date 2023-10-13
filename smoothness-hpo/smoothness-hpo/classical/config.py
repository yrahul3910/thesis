from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    n_units: int = 5
    n_layers: int = 2
    weighted: bool = True
    wfo: bool = True
    ultrasample: bool = True
    smote: bool = True
    smooth: bool = True
    transform: Literal['normalize', 'standardize',
                       'minmax', 'maxabs', 'robust'] = 'normalize'
    