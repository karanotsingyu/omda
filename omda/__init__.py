import numpy as np
import pandas as pd

class Analysis:
    # TODO: 一键生成结果和报告
    
    def __init__(self, options, pref, normalized=False):
        self.options = self._validate_options(options)
        self.pref = self._validate_pref(pref)
        self.normalized = normalized
    
    def _validate_options(self, options):
        if isinstance(options, pd.DataFrame):
            return options
        elif isinstance(options, str):
            try:
                options_df = pd.read_excel(options)
                return options_df
            except Exception as e:
                raise ValueError(f"Failed to read Excel file from path: {options}. Error: {e}")
        else:
            raise TypeError("options must be a pandas.DataFrame or a string representing a file path")
    
    def _validate_pref(self, pref):
        if isinstance(pref, Preference):
            return pref
        elif isinstance(pref, (pd.DataFrame, str)):
            return Preference(pref)
        # elif isinstance(pref, str):
        #     try:
        #         pref_df = pd.read_excel(pref)
        #         return pref_df
        #     except Exception as e:
        #         raise ValueError(f"Failed to read Excel file from path: {pref}. Error: {e}")
        else:
            raise TypeError("pref must be one of these types: omda.Preference, dictionary, pandas.DataFrame or a string representing a file path")
        
    # if not normalized:
    #     # TODO: normalization by dimension
    #     raise NotImplementedError
    #     self.options = normalize_mat(
    #         data, method=norm_method,
    #         low_is_better=low_is_better
    #     )
    
    def roc(self):
        pass
    
    def ewm(self):
        pass

class Preference:
    # Preference 需要处理的难点在于：分析中途修改 preference 的顺序
    """
    Preference include properties and their preffered values as well as order,
    i.e. ordered properties with preffered value directions
    """
    def __init__(self, properties=None, pref_val=None, pref_order=None, 
                 pref_dir=None, 
                 property_col=None, pref_val_col=None, pref_order_col=None):
        if pref_dir:
            self._load_from_file(
                pref_dir, 
                property_col=property_col, 
                pref_val_col=pref_val_col,
                pref_order_col=pref_order_col
            )
        else:
            if properties is None or order is None:
                raise ValueError("Both properties and order must be provided if pref_dir is not specified.")
            # TODO: [240611-03] finish _validate_properties
            self.properties = self._validate_properties(properties)
            # TODO: alert: 区分 pref_order 和 ordered_properties
            self.pref_order = self._validate_pref_order(pref_order)
    
    def _load_from_file(self, pref_dir, 
                        property_col, pref_val_col, pref_order_col):
        pref_df = pd.read_excel(pref_dir)
        self.properties = pref_df[property_col]
        self.pref_val = pref_df[pref_val_col]
        self.pref_order = pref_df[pref_order_col]
    
    def _validate_properties(self, properties):
        # TODO: [240611-03] finish _validate_properties
        pass
    
    def _validate_pref_order(self, pref_order):
        # check if order is a list
        if not isinstance(order, list):
            raise TypeError("pref_order must be a list")
        
        # check if all elements in order are strings
        if not all(isinstance(item, str) for item in order):
            raise ValueError("All items in pref_order must be strings")

class Options:
    pass

class Property:
    def __init__(self, property_name, pref_val):
        self.property_name = property_name
        self.pref_val = pref_val