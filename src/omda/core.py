import pandas as pd

from .models import *
from .methods import *
from .utils import *

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
        
        # # FUTURE:
        # self.methods = {
        #     'roc': ROCMethod(),
        #     'ewm': EWMethod()
        # }
            
        self.results_title = '# Decision Analysis Results'
        self.results = [self.results_title]
    
    def _filter_property_by_pref(self, options, pref, ordered=False):
        if ordered == True:
            raise NotImplementedError
        filtered_options = self.options_raw[list(pref.property_names)]
        return filtered_options
    
    def _validate_options(self, options):
        if isinstance(options, pd.DataFrame):
            return self._check_empty_rows(options)
        elif isinstance(options, str):
            try:
                options_df = pd.read_excel(options)
                return self._check_empty_rows(options_df)
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
        else:
            err_msg = "Variable `self.pref` must be one of these types: omda.Preference, dictionary, pandas.DataFrame or a string representing a file path"
            raise TypeError(err_msg)
        
    def _check_empty_rows(self, df):
        """
        Check if there are empty rows in the DataFrame
        
        Args:
            df (pd.DataFrame): The DataFrame to check
        
        Returns:
            pd.DataFrame: The original DataFrame if there are no empty rows
            
        Raises:
            ValueError: If there are empty rows, raise an error and point out the location of the empty rows
        """
        empty_rows = df.isna().all(axis=1)
        if empty_rows.any():
            empty_row_indices = [i + 1 for i in range(len(df)) if empty_rows[i]]
            err_msg = f"Empty rows found at indices: {empty_row_indices}. Please delete them."
            raise ValueError(err_msg)
        return df
        
    def roc(self, default=True, options=None, pref=None,
            score_precision=4, output_filepath=None):
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
            score_precision=4, output_filepath=None):
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
    
    # # FUTURE:
    # def run_method(self, method_name):
    #     method = self.methods[method_name]
    #     weights = method.compute_weights(self.options_filtered, self.pref)
    #     scores = method.compute_scores(self.options_filtered, weights) 
    #     ranks = method.compute_ranks(scores)
        
    #     return weights, scores, ranks

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

class Results:
    
    def __init__(self):
        
        raise NotImplementedError