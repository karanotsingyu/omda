import textwrap

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .core import Analysis
from .models import Preference, Property
from .methods import normalize_vec, normalize_mat,compute_weight_roc, compute_weight_ewm

__all__ = [
    'Analysis',
    'Preference', 
    'Property',
    'compute_weight_roc',
    'compute_weight_ewm',
    'normalize_vec',
    'normalize_mat'
]

# TODO: [High Priority] Implement class 'DecisionMethod' and remove redundant snippets
# TODO: [High Priority] Implement class 'Results' or simply some functions to remove redundant snippets
# TODO: Optimize the user experience of creating instances of class 'Preference'
# TODO: More detailed results report with more params mentioned with respect to specific methods
# TODO: Write and publish a reference manual
# TODO: Add method: "BWM"
# TODO: Add method: "TOPSIS"
# TODO: Add method: "VIKOR"
# TODO: Smart recognition of decision needs and their transformation to "entities" "properties" "values" "preference" (regarding properties and also prospect/expectation/risk)