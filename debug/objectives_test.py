#%%
import pandas as pd
import numpy as np
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.objectives import (
    compute_reliability,
    compute_max_annual_accumulated_degree_days,
    compute_mean_thermal_bank_usage_ratio
)


dates = pd.date_range('2018-01-01', '2019-12-31', freq='D')
# Year 2018: mostly 22°C (2°C above threshold=20), Year 2019: mostly 25°C (5°C above threshold)
temps = [22] * 365 + [25] * 365  # 365 days each year

df_temp = pd.DataFrame({
    'date': dates,
    'Tavg_L': temps
})
df_temp.set_index('date', inplace=True)
result2 = compute_max_annual_accumulated_degree_days(df_temp, col='Tavg_L', threshold=20)





dates_bank = [
    pd.Timestamp('2018-09-01'),
    pd.Timestamp('2019-09-01'),
    pd.Timestamp('2020-09-01')
]
bank_amounts = [1200, 800, 1000]  # Remaining bank amounts (out of 1620 total)

df_bank = pd.DataFrame({
    'date': dates_bank,
    'remained_bank_amounts': bank_amounts
})
df_bank.set_index('date', inplace=True)

result3 = compute_mean_thermal_bank_usage_ratio(df_bank, col='remained_bank_amounts', bank_size=1620)


temp_data = [22, 23, 25, 26, 24, 23, 22, 27, 28, 25,  # First 10 days: 4/10 <= 24
            20, 21, 22, 23, 24, 25, 26, 23, 22, 21]  # Next 10 days: 8/10 <= 24
temp_data = np.zeros(20) + 25
result1 = compute_reliability(temp_data, threshold=24)
