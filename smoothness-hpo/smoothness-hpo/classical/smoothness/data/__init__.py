from .issues import load_issue_lifetime_prediction_data
from .defect import load_defect_prediction_data
from .smooth import remove_labels_legacy

import pyximport

pyximport.install()

from .remove_labels import remove_labels
