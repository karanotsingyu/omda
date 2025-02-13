from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import entropy

# # FUTURE:
# class DecisionMethod(ABC):
#     @abstractmethod
#     def compute_weights(self, options, pref):
#         pass
        
#     @abstractmethod  
#     def compute_scores(self, options, weights):
#         pass
        
#     @abstractmethod
#     def compute_ranks(self, scores):
#         pass

# class ROCMethod(DecisionMethod):
#     def compute_weights(self, options, pref):
#         return compute_weight_roc(options, pref)
        
#     def compute_scores(self, options, weights):
#         return compute_scores(options, weights, suffix='ROC')
        
#     def compute_ranks(self, scores):
#         return compute_rank(scores['TotalScore_ROC'], suffix='ROC')

# class EWMethod(DecisionMethod):
#     def compute_weights(self, options, pref):
#         weights, entropy = compute_weight_ewm(options, pref)
#         self.entropy = entropy
#         return weights
        
#     def compute_scores(self, options, weights):
#         return compute_scores(options, weights, suffix='EWM')
        
#     def compute_ranks(self, scores): 
#         return compute_rank(scores['TotalScore_EWM'], suffix='EWM')

def normalize_vec(vec, low_is_better, method='prop'):
    if not (type(vec) in [np.ndarray, pd.Series] and len(vec.shape) == 1):
        err_msg = "Object 'vec` must be 'pandas.Series' or 1d 'numpy.ndarray'"
        raise ValueError(err_msg)
    if not method in ['prop', 'sub']:
        err_msg = "Value of 'norm_method' must be 'prop' or 'sub'!"
        raise ValueError(err_msg)
    
    # TODO: [High Priority][240820-01] Double check double types of normalization esp. their combinations among roc, ewm, and others -- clarification and examination of the 'norm_method' usage is strictly needed
    if method == 'prop': # proportional
        summed_score = np.sum(vec)
        # Make sure 'summed_score' is not 0
        if summed_score == 0:
            err_msg = "Object 'summed_score' is zero which cannot be denominator!"
            raise ValueError(err_msg)
        # Normalization by summation
        normalized_vec = vec / summed_score
    elif method == 'sub': # subtraction
        min_element, max_element = np.min(vec), np.max(vec)
        difference = np.max(vec) - min_element
        # Make sure 'difference' is not 0
        if difference == 0:
            err_msg = "Object 'difference' is zero which cannot be denominator!"
            raise ValueError(err_msg)
        # Normalization by subtraction
        if not low_is_better:
            normalized_vec = (vec - min_element) / difference
        else:
            normalized_vec = (max_element - vec) / difference
    else:
        err_msg = f"Normalization method '{method}' is not implemented."
        raise NotImplementedError(err_msg)
    
    return normalized_vec

def normalize_mat(data, pref_monotonicity, norm_method='sub'):
    if not isinstance(data, pd.DataFrame):
        err_msg = "\'data\' object must be \'pandas.DataFrame\'"
        raise TypeError(err_msg)
    
    # Get the number of columns
    num_col = data.shape[1]
    
    normalized_data = pd.DataFrame()
    for i_col in range(num_col):
        # Get the current vector
        vec = data.iloc[:,i_col]
        # Get the monotonical preference of the current vector
        monotonicity = pref_monotonicity[vec.name]
        if monotonicity is None:
            err_msg = f"Monotonicity preference for column '{vec.name}' is missing."
            raise ValueError(err_msg)
        # Convert monotonicity to boolean
        if monotonicity in ['low', 'high']:
            low_is_better = (monotonicity == 'low')
        else:
            err_msg = "Object 'monotonicity' must be 'high' or 'low'!"
            raise ValueError(err_msg)
        
        # Normalize the given vector
        normalized_vec = normalize_vec(vec, low_is_better, method=norm_method)
        # Assign the named normalized vector
        new_vec_name = vec.name
        normalized_data[new_vec_name] = normalized_vec
    
    return normalized_data

def compute_weight_roc(options, pref, method='inverse'):
    """
    See (Edwards & Barron, 1994).
    """
    # Initialization
    num_property = len(pref.property_names)
    # Top priority first
    original_array = np.arange(1, num_property+1)
    
    # Compute weight given the selected method
    # The first weight is the most valuable
    if method == "inverse":
        inversed_array = 1 / original_array
        weights = [
            inversed_array[i-1] / np.sum(inversed_array) \
            for i in range(1, num_property+1)
        ]
    elif method == "addition": 
        weights = [
            (num_property + 1 - i) / np.sum(original_array) \
            for i in range(1, num_property+1)
        ]
    elif method == "exponent": 
        z = np.log(100) / np.log(num_property)
        weights = [
            (num_property + 1 - i)**z / np.sum(original_array**z) \
            for i in range(1, num_property+1)
        ]
    elif method == "simple":
        inversed_array = 1 / original_array
        weights = [
            np.sum(inversed_array[i-1:]) / num_property \
            for i in range(1, num_property+1)
        ]
    
    # Order by 'pref.priority_order'
    sorted_property_names = pref.priority_order.sort_values().index
    weights = pd.Series(weights, index=sorted_property_names)
    weights.rename('Weight_ROC', inplace=True)
    
    return weights

def compute_weight_ewm(normalized_data, pref):
    # Compute information entropy given information entropy
    k = 1 / np.log(normalized_data.shape[0])
    entropy_vec = k * entropy(normalized_data, axis=0)
    redundancy_vec = 1 - entropy_vec
    # Compute weights given information utility
    weights = pd.Series(
        normalize_vec(redundancy_vec, low_is_better=False), 
        index=normalized_data.columns,
        name='Weight_Entropy'
    )
    entropy_series = pd.Series(
        entropy_vec, 
        index=normalized_data.columns,
        name='Entropy'
    )
    entropy_series.sort_values(ascending=True, inplace=True)
    
    return weights, entropy_series

def compute_scores(normalized_data, weights, suffix=None):
    if not isinstance(weights, pd.Series):
        err_msg = "Object 'weights' must be 'pd.Series'!"
        raise TypeError(err_msg)
    if not isinstance(normalized_data, pd.DataFrame):
        err_msg = "Object 'normalized_data' must be 'pd.DataFrame'"
        raise TypeError(err_msg)
    
    scores = pd.DataFrame()
    for property_name in normalized_data.columns:
        scores[property_name] = \
            (normalized_data[property_name] * weights.loc[property_name])
    
    try:
        if not suffix.lower() in ['roc', 'ewm']:
            err_msg = "Value of 'suffix' must be 'roc' or 'ewm'"
            raise ValueError(err_msg)
    except AttributeError:
        if not suffix:
            raise TypeError("Object 'suffix' is 'NoneType`")
        else:
            raise AttributeError("Unexpected AttributeError of object 'suffix`")
    
    total_score = scores.sum(axis=1)
    scores[f'TotalScore_{suffix}'] = total_score

    return scores

def compute_rank(scores, suffix=None):
    if not isinstance(scores, pd.Series):
        err_msg = "Object 'scores' must be 'pd.Series'!"
        raise TypeError(err_msg)
    try:
        if not suffix.lower() in ['roc', 'ewm']:
            err_msg = "Value of 'suffix' must be 'roc' or 'ewm'"
            raise ValueError(err_msg)
    except AttributeError:
        if not suffix:
            raise TypeError("Object 'suffix' is 'NoneType`")
        else:
            raise AttributeError("Unexpected AttributeError of object 'suffix`")
    
    rank = scores.rank(ascending=False)
    rank = rank.astype(int)
    rank.sort_values(ascending=True, inplace=True)
    
    if suffix:
        rank.rename(f'Rank_{suffix}', inplace=True)
    else:
        rank.rename(f'Rank', inplace=True)
    
    return rank