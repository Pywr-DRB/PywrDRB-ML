import pandas as pd
import pathnavigator

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()



db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)
db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)

df = pd.DataFrame()
df["Salt front (RM)"] = db_SalinityLSTM[["saltfront"]]
df["Water Tmax (degC)"] = db_TempLSTM[["QbcTmax_T_L"]]
df["Water Tavg (degC)"] = db_TempLSTM[["QbcTavg_T_L"]]
df["Q_Lordville"] = db_TempLSTM[["QbcTmax_Q_L"]]
df["Q_Trenton"] = db_SalinityLSTM[["Q_Trenton_bc"]]
df["Q_Schuylkill"] = db_SalinityLSTM[["Q_Schuylkill_bc"]]


df = df["1979":"2023"]
#df = df[df["Water Tavg (degC)"] >= 20]
df = df[df["Salt front (RM)"] >= 80]
df = df[df.index.month.isin([6, 7, 8])]
#%%

import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot for all columns in df
sns.pairplot(df)
plt.suptitle("Pairplot of all variables", y=1.02)
plt.show()

#%%
df_clean = df.dropna(subset=["Salt front (RM)", "Water Tmax (degC)"])
# Jointplot between saltfront and Tmax
g = sns.jointplot(
    data=df_clean, 
    x="Salt front (RM)", y="Water Tmax (degC)", kind="hex", color="#4CB391")
g.ax_joint.axhline(24, color="r", linestyle="-")
g.ax_joint.axvline(82.9, color="b", linestyle="--")
g.ax_joint.axvline(87, color="b", linestyle="-.")
g.ax_joint.axvline(92.5, color="b", linestyle=":")
plt.show()

#%%
df_clean = df.dropna(subset=["Salt front (RM)", "Water Tavg (degC)"])
g = sns.jointplot(
    data=df_clean, 
    x="Salt front (RM)", y="Water Tavg (degC)", kind="hex", color="#4CB391")
# Add vertical line to the joint (main) axis
g.ax_joint.axvline(82.9, color="k", linestyle="--")
plt.show()


#%%
fig, ax = plt.subplots()
dff = df["1999/8/18":"1999/09/30"]
dff["Salt front (RM)"].plot(ax=ax)
ax.axhline(82.9, color="b", linestyle="--")
ax.axhline(87, color="b", linestyle="-.")
ax.axhline(92.5, color="b", linestyle=":")
ax.set_ylabel("Salt front (RM)")
plt.show()

fig, ax = plt.subplots()
dff = df["2001/12/18":"2002/11/25"]
dff["Salt front (RM)"].plot(ax=ax)
ax.axhline(82.9, color="b", linestyle="--")
ax.axhline(87, color="b", linestyle="-.")
ax.axhline(92.5, color="b", linestyle=":")
ax.set_ylabel("Salt front (RM)")
plt.show()


#%%

sns.pairplot(data=penguins, hue="species")


import numpy as np
import seaborn as sns
sns.set_theme(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x=x, y=y, kind="hex", color="#4CB391")