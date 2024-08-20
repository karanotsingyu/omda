import textwrap

import numpy as np
import pandas as pd
from scipy.stats import entropy

# TODO: [High Priority][240820-01] Double check double types of normalization esp. their combinations among roc, ewm, and others -- clarification and examination of the 'norm_method' usage is strictly needed
# TODO: [High Priority] Implement class 'DecisionMethod' and remove redundant snippets
# TODO: [High Priority] Implement class 'Results' or simply some functions to remove redundant snippets
# TODO: Optimize the user experience of creating instances of class 'Preference'
# TODO: More detailed results report with more params mentioned with respect to specific methods
# TODO: Write and publish a reference manual
# TODO: Add method: "BWM"
# TODO: Add method: "TOPSIS"
# TODO: Add method: "VIKOR"
# TODO: Smart recognition of decision needs and their transformation to "entities" "properties" "values" "preference" (regarding properties and also prospect/expectation/risk)

class Analysis:
    def __init__(self, options_raw, pref, 
            normalized=False, norm_method='prop'):
        self.options_raw = self._validate_options(options_raw)
        self.pref = self._validate_pref(pref)
        self.normalized = normalized
        self.norm_method = norm_method
        
        self.options_filtered = self._filter_property_by_pref(
            options=self.options_raw, pref=self.pref
        )
        if not normalized: 
            self.options_filtered = normalize_mat(
                data=self.options_filtered,
                pref_monotonicity=pref.monotonicity,
                norm_method=norm_method,
            )
        else:
            raise NotImplementedError
            
        self.results_title = '# Decision Analysis Results'
        self.results = [self.results_title]
    
    def __repr__(self):
        class_name = self.__class__.__name__
        # TODO: define __repr__
        pass
    
    def _filter_property_by_pref(self, options, pref, ordered=False):
        if ordered == True:
            raise NotImplementedError
        filtered_options = self.options_raw[list(pref.property_names)]
        return filtered_options
    
    def _validate_options(self, options):
        if isinstance(options, pd.DataFrame):
            return options
        elif isinstance(options, str):
            try:
                options_df = pd.read_excel(options)
                return options_df
            except Exception as e:
                err_msg = f"Failed to read Excel file from path: {options}. Error: {e}"
                raise ValueError(err_msg)
        else:
            err_msg = "'options' object must be a pandas.DataFrame or a string representing a file path"
            raise TypeError(err_msg)
    
    def _validate_pref(self, pref):
        if isinstance(pref, Preference):
            return pref
        elif isinstance(pref, (pd.DataFrame, str)):
            return Preference(pref)
        elif isinstance(pref, str):
            raise NotImplementedError
            # TODO: Implement load preference from str (???)
            try:
                pref_df = pd.read_excel(pref)
                return pref_df
            except Exception as e:
                raise ValueError(f"Failed to read Excel file from path: {pref}. Error: {e}")
        else:
            err_msg = "Variable `self.pref` must be one of these types: omda.Preference, dictionary, pandas.DataFrame or a string representing a file path"
            raise TypeError(err_msg)
    
    def roc(self, default=True, options=None, pref=None,
            score_precision=1, output_filepath=None):
        if default:
            options = self.options_filtered
            pref = self.pref
        else:
            raise NotImplementedError
        
        # Compute weights by ROC method
        self.weight_roc = compute_weight_roc(options, pref)
        self.score_roc  = compute_scores(options, self.weight_roc, suffix='ROC')
        self.rank_roc   = compute_rank(self.score_roc['TotalScore_ROC'], suffix='ROC')
        self.score_roc.sort_values(by='TotalScore_ROC', ascending=False, inplace=True)
        # Deduce for convenience, 
        #     while strictly distinguished from original values
        self.score_roc_total = self.score_roc['TotalScore_ROC']
        self.score_roc_scaled = self.score_roc * 100
        
        # Generate results and return them
        self.results_roc = ["## Results of ROC"]
        self.results_roc.append("### Object `weight_roc`")
        self.results_roc.append(series_by_property_to_md(self.weight_roc, 6))
        self.results_roc.append("### Object `score_roc`")
        self.results_roc.append(self.score_roc.round(score_precision).to_markdown(floatfmt=f".{score_precision}f"))
        self.results_roc.append("### Object `rank_roc`")
        self.results_roc.append(self.rank_roc.to_markdown())
        
        save_results_as_md(self.results_roc, output_filepath)
        
        return self.results_roc
    
    def ewm(self, default=True, options=None, pref=None,
            score_precision=1, output_filepath=None):
        if default:
            options = self.options_filtered
            pref = self.pref
        else:
            raise NotImplementedError
        
        # Compute weights by entropy method
        self.weight_ewm, self.entropy = compute_weight_ewm(options, pref)
        self.score_ewm  = compute_scores(options, self.weight_ewm, suffix='EWM')
        self.rank_ewm   = compute_rank(self.score_ewm['TotalScore_EWM'], suffix='EWM')
        self.score_ewm.sort_values(by='TotalScore_EWM', ascending=False, inplace=True)
        # Deduce for convenience, 
        #     while strictly distinguished from original values
        self.score_ewm_total = self.score_ewm['TotalScore_EWM']
        self.score_ewm_scaled = self.score_ewm * 100
    
        # Generate results and return them
        self.results_ewm = ["## Results of EWM"]
        self.results_ewm.append("### Object `entropy`")
        self.results_ewm.append(series_by_property_to_md(self.entropy, 6))
        self.results_ewm.append("### Object `weight_ewm`")
        self.results_ewm.append(series_by_property_to_md(self.weight_ewm, 6))
        self.results_ewm.append("### Object `score_ewm`")
        self.results_ewm.append(self.score_ewm.round(score_precision).to_markdown(floatfmt=f".{score_precision}f"))
        self.results_ewm.append("### Object `rank_ewm`")
        self.results_ewm.append(self.rank_ewm.to_markdown())
        
        save_results_as_md(self.results_ewm, output_filepath)
    
        return self.results_ewm    
    
    def combine_results_different_methods(self, score_precision=2):
        self.results_comparison = ['## Results Comparison']
        
        self.weights_comparison = pd.concat([self.weight_roc, self.weight_ewm], axis=1)
        self.total_scores_comparison = pd.concat([self.score_roc_total, self.score_ewm_total], axis=1)
        self.ranks_comparison = pd.concat([self.rank_roc, self.rank_ewm], axis=1)
        
        self.weights_comparison.reset_index(inplace=True)
        self.weights_comparison.rename(columns={'index': 'PropertyName'}, inplace=True)
        
        self.results_comparison += \
            [
                "### Object `weights_comparison`", 
                self.weights_comparison.to_markdown(index=False, floatfmt=f".{score_precision}f"), 
                "### Object `total_scores_comparison`", 
                (self.total_scores_comparison*100).to_markdown(floatfmt=f".{score_precision}f"), 
                "### Object `ranks_comparison`", 
                self.ranks_comparison.to_markdown(),
            ]
        
        return self.results_comparison
    
    def run(self, output_filepath=None):
        
        self.results = [self.results_title]
        
        self.results += self.roc()
        self.results += self.ewm()
        self.results += self.combine_results_different_methods()
        
        save_results_as_md(self.results, output_filepath)
        if output_filepath:
            print("All-in-one results generated as a Markdown file.")
        else:
            print("All-in-one results generated.")
        
        return self.results

class Preference:
    # Preference 需要处理的难点在于：分析中途修改 preference 的顺序
    """
    A 'Preference' object contains properties and corresponding monotonicity 
        preferences as well as property order, i.e. ordered properties with 
        preffered value monotonical values
    """
    def __init__(self, pref_dir=None, # create from outer files by default
                 property_names=None, monotonicity=None, property_order=None, 
                 property_name_col=None, monotonicity_col=None, property_order_col=None):
        if pref_dir:
            if None in (property_name_col, monotonicity_col, property_order_col):
                raise ValueError("One or more variables are None!") 
                
            # specify attributes
            #     'self.property_names',
            #     'self.monotonicity', and 
            #     'self.priority_order'
            self._init_from_file(
                pref_dir, 
                property_name_col=property_name_col, 
                monotonicity_col=monotonicity_col,
                property_order_col=property_order_col
            )
        else:
            # if (properties is None) or (order is None):
            #     err_msg = "Both properties and order must be provided if pref_dir is not specified."
            #     raise ValueError(err_msg)
            # # TODO: finish '_validate_properties'
            # self.property_names = self._validate_properties(properties)
            # self.priority_order = self._validate_pref_order(priority_order)
            
            raise NotImplementedError
    
    def __str__(self):
        # TODO: rewrite params concerning 'r'/'repr'
        class_name = self.__class__.__name__
        repr_priority_order = list(self.priority_order)
        repr_monotonicity = list(self.monotonicity)
        repr_property_names = [p.property_name for p in list(self.properties)]
        
        repr_str = textwrap.dedent(f"""
            {class_name}(
                property_names={repr_property_names!r},
                property_order={repr_priority_order!r},
                monotonicity={repr_monotonicity!r}
            )
        """)
        return repr_str
    
    def __repr__(self):
        # TODO: rewrite `repr_str` so that 
        #     the output can be executed by Python built-in `eval`
        class_name = self.__class__.__name__
        repr_priority_order = list(self.priority_order)
        repr_monotonicity = list(self.monotonicity)
        repr_property_names = [p.property_name for p in list(self.properties)]
        
        repr_str = textwrap.dedent(f"""
            {class_name}(
                property_names={repr_property_names!r},
                property_order={repr_priority_order!r},
                monotonicity={repr_monotonicity!r}
            )
        """)
        return repr_str
    
    def _init_from_file(self, pref_dir, 
            property_name_col, monotonicity_col, property_order_col):
        pref_df = pd.read_excel(pref_dir)
        pref_df.dropna(how='any', inplace=True)
        
        # Specify the values
        try:
            self.property_names = pref_df[property_name_col]
            self.monotonicity = pref_df[monotonicity_col]
            self.priority_order = pref_df[property_order_col].astype(int)
        except KeyError:
            raise KeyError("Please ensure column names matched")
        # Specify the indexes
        self.monotonicity.index = self.property_names
        self.priority_order.index = self.property_names
        
        # Instantiate a list of 'omda.Property'
        self.properties = []
        for i in range(len(pref_df)):
            self.properties.append(
                Property(
                    property_name=self.property_names.iloc[i],
                    monotonicity=self.monotonicity.iloc[i],
                )
            )
        self.properties = pd.Series(self.properties, index=self.property_names)
        # Sort 'self.properties' by 'priority_order'
        self.properties = self.properties.loc[self.priority_order.sort_values().index]
    
    def _validate_properties(self, properties):
        pass
        
        raise NotImplementedError
    
    def _validate_pref_order(self, priority_order):
        pass
        
        raise NotImplementedError
        
        # # check if order is a list
        # if not isinstance(order, list):
        #     raise TypeError("priority_order must be a list")
        #
        # # check if all elements in order are strings
        # if not all(isinstance(item, str) for item in order):
        #     raise ValueError("All items in priority_order must be strings")

class Options:
    
    # TODO: consider Options(pd.DataFrame) ???
    
    def __init__(self):
        
        raise NotImplementedError

class Property:
    
    def __init__(self, property_name, monotonicity):
        self.property_name = property_name
        self.monotonicity = monotonicity
    
    def __repr__(self):
        class_name = self.__class__.__name__
        
        repr_str = f"{class_name}({self.property_name!r}, {self.monotonicity!r})"
        return repr_str

class DecisionMethod:
    
    def __init__(self):
        
        raise NotImplementedError

class Results:
    
    def __init__(self):
        
        raise NotImplementedError

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

def series_by_property_to_md(s, precision):
    if not isinstance(s, pd.Series):
        raise TypeError
    return s.reset_index().rename(columns={'index': 'PropertyName'}).to_markdown(index=False, floatfmt=f".{precision}f")
    
def save_results_as_md(results, output_filepath):
    if not isinstance(results, list):
        err_msg = f"Object of 'results' should be a 'list', but it is a '{results.__class__.__name__}'"
        raise TypeError(err_msg)
    elif not all(isinstance(item, str) for item in results):
        err_msg = f"Make sure every element in 'results' is 'str' type"
        raise TypeError(err_msg)
        
    if output_filepath:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write('\n\n'.join(results))
