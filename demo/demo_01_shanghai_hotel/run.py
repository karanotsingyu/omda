from pathlib import Path

import pandas as pd

import omda as da

DATA_DIR = Path(__file__).parent
OPTIONS_PATH = DATA_DIR / 'options_v1.xlsx'
PREFERENCE_PATH = DATA_DIR / 'pref_v2.xlsx'

options_1 = pd.read_excel(OPTIONS_PATH, index_col=0)
pref_1_df = pd.read_excel(PREFERENCE_PATH).dropna(how='all')
pref_1 = da.Preference(
    PREFERENCE_PATH,
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

# for testing
assert round(float(round_1.weight_roc['Risk']), 3) == 0.48
assert round(float(round_1.weight_roc['Review']), 3) == 0.24
assert round(float(round_1.weight_roc['Price']), 3) == 0.16
assert round_1.rank_roc['一只柠萌_淮海路'] == 1
assert round_1.rank_roc['华敏青旅_外滩'] == 2
assert round_1.rank_roc['如眠客_静安寺店'] == 3
assert round(float(round_1.weight_ewm['Risk']), 6) == 0.494525
assert round(float(round_1.weight_ewm['Review']), 6) == 0.106088
assert round(float(round_1.weight_ewm['Price']), 6) == 0.200234
assert round_1.rank_ewm['一只柠萌_淮海路'] == 1
assert round_1.rank_ewm['华敏青旅_外滩'] == 2
assert round_1.rank_ewm['如眠客_静安寺店'] == 3