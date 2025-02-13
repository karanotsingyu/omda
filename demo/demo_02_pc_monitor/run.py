from pathlib import Path

import pandas as pd

import omda as da

DATA_DIR = Path(__file__).parent

# round 1

OPTIONS_PATH_1 = DATA_DIR / 'options_v1.xlsx'
PREFERENCE_PATH_1 = DATA_DIR / 'pref_v1.xlsx'

options_1 = pd.read_excel(OPTIONS_PATH_1, index_col=0)
pref_1 = da.Preference(
    PREFERENCE_PATH_1,
    property_name_col="Property",
    monotonicity_col="Monotonicity",
    property_order_col="Order",
)
print("Preference 1 instantiated:\n", pref_1)

round_1 = da.Analysis(options_1, pref_1, norm_method='sub')
round_1.run('r01_results.md')

print("Round 1 analysis done.\n\n")

# round 2

OPTIONS_PATH_2 = DATA_DIR / 'options_v2.xlsx'
PREFERENCE_PATH_2 = DATA_DIR / 'pref_v2.xlsx'

options_2 = pd.read_excel(OPTIONS_PATH_2, index_col=0)
pref_2 = da.Preference(
    PREFERENCE_PATH_2,
    property_name_col="Property",
    monotonicity_col="Monotonicity",
    property_order_col="Order",
)
print("Preference 2 instantiated:\n", pref_2)
round_2 = da.Analysis(options_2, pref_2, norm_method='sub')
round_2.run('r02_results.md')

print("Round 2 analysis done.\n\n")