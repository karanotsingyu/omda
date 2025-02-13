import pandas as pd

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