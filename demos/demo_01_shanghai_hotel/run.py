import pandas as pd

import omda as da

OPTIONS_DIR = 'options_v1.xlsx'
PREF_DIR = 'pref_v2.xlsx'

options_1 = pd.read_excel(OPTIONS_DIR, index_col=0)
pref_1_df = pd.read_excel(PREF_DIR).dropna(how='all')
pref_1 = da.Preference(
    PREF_DIR,
    property_name_col="PropertyName",
    monotonicity_col="PrefMono",
    property_order_col="PrefOrder",
)
print("Preference instantiated:\n", pref_1)

round_1 = da.Analysis(
    options_1, 
    pref=pref_1,
    norm_method='sub',
    )
round_1.run('results_all_in_one.md')