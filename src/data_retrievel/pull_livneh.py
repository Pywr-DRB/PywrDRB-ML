import os
import requests
from tqdm import tqdm

# Directory to save files
output_dir = "D:\Data\livneh"
os.makedirs(output_dir, exist_ok=True)

# URL template
var = "tmin" #"tmax" #"prec"     [K, mm]
base_url = f"https://downloads.psl.noaa.gov/Datasets/livneh/metvars/{var}.{{year}}.nc"

var = "qnet" # Solar radiation (W m-2)
base_url = f"https://downloads.psl.noaa.gov/Datasets/livneh/fluxvars/{var}.{{year}}.nc"

# Year range
start_year = 1915
end_year = 2011

# Download files
for year in tqdm(range(start_year, end_year + 1), desc="Downloading files"):
    url = base_url.format(year=year)
    output_path = os.path.join(output_dir, f"{var}.{year}.nc")

    if os.path.exists(output_path):
        print(f"File for {year} already exists. Skipping...")
        continue

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for bad status codes

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


#%%
import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import xarray as xr
import pandas as pd
from tqdm import tqdm

# Year range
start_year = 1915
end_year = 2011

# Lordville lat/lon
lat = 41.86727778
lon = -75.21375
lon = lon % 360 # convert a longitude from the -180 to 180 system to 0 to 360

data_vars_shrt_all = ["tmin", "tmax", "prec", "qnet"]

### lat lon bounds
lon_min = lon
lat_min = lat
lon_max = lon
lat_max = lat

df_combined = []
## Loop through variables and pull data + place in dict
for var in data_vars_shrt_all:
    dfs = []
    for year in tqdm(range(start_year, end_year + 1), desc=var):
        ## Pulling data
        path = f"D:\Data\livneh\{var}.{year}.nc"
        ds = xr.open_dataset(path)

        #ds.info

        ## Subset to timeframe and bbox of interest:
        ds_subset = ds.sel(lon = lon,
                           lat = lat,
                           method='nearest')
        data_var_name = list(ds_subset.data_vars.keys())[0]
        df = ds_subset[data_var_name].to_dataframe().drop(columns=['lat','lon'])
        df.index.name = "date"
        # Append to dict of xr.datasets
        dfs.append(df)
    df_combined.append(pd.concat(dfs, axis=0))
df_combined = pd.concat(df_combined, axis=1)

# Unit conversion
#  tmmx: K to C
#  tmmn: K to C
#  pr: mm to m
#  srad: W m-2

df_combined["prec"] *= 0.001 # mm to m
df_combined["tmin"] -= 273.15  # Kelvin to Celsius
df_combined["tmax"] -= 273.15  # Kelvin to Celsius

df_combined = df_combined.rename(columns={"tmin": "tmmn", "tmax": "tmmx", "prec": "pr", "qnet": "srad"})
df_combined.to_csv(pn.data.raw.get() / "livneh_lordville.csv")







