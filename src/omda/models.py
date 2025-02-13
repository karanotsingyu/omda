import textwrap
import pandas as pd

class Preference:
    # Preference 需要处理的难点在于：分析中途修改 preference 的顺序
    """
    A 'Preference' object contains properties and corresponding monotonicity 
        preferences as well as property order, i.e. ordered properties with 
        preffered value monotonical values
    """
    def __init__(self, preference_path=None, # create from outer files by default
                 property_names=None, monotonicity=None, property_order=None, 
                 property_name_col=None, monotonicity_col=None, property_order_col=None):
        if preference_path:
            if None in (property_name_col, monotonicity_col, property_order_col):
                raise ValueError("One or more variables are None!") 
                
            # specify attributes
            #     'self.property_names',
            #     'self.monotonicity', and 
            #     'self.priority_order'
            self._init_from_file(
                preference_path, 
                property_name_col=property_name_col, 
                monotonicity_col=monotonicity_col,
                property_order_col=property_order_col
            )
        else:
            # if (properties is None) or (order is None):
            #     err_msg = "Both properties and order must be provided if preference_path is not specified."
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
    
    def _init_from_file(self, preference_path, 
            property_name_col, monotonicity_col, property_order_col):
        pref_df = pd.read_excel(preference_path)
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