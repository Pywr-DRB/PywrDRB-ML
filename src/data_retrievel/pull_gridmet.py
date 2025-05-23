import time
import xarray as xr
import pandas as pd

# for pulling gridmet from Lordville site

# Lordville lat/lon
lat = 41.86727778
lon = -75.21375

data_vars_shrt_all = ['tmmx', 'tmmn', 'pr', 'srad', 'vs','rmax','rmin','sph']

# commenting out to just pull all dates
# start_date = '1979-01-01'
# end_date = '2025-02-03'

### lat lon bounds
lon_min = lon
lat_min = lat
lon_max = lon
lat_max = lat

## if only 1 var (short) is provided, change to list
if not isinstance(data_vars_shrt_all, list):
    data_vars_shrt_all = [data_vars_shrt_all]

## Initiate list of empty datasets
xarray_dict = dict()

## Loop through variables and pull data + place in dict
for var in data_vars_shrt_all:
    ## Pulling data
    # source: http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_MET_aggregated_catalog.html
    url = f'http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_{var}_1979_CurrentYear_CONUS.nc'
    # call data from url
    start = time.perf_counter()
    ds = xr.open_dataset(url + '#fillmismatch')

    ## Subset to timeframe and bbox of interest:
    ds_subset = ds.sel(lon = lon,
                       lat = lat,
                       method='nearest')
    end = time.perf_counter()
    print(f'finish {var} in {round(end - start, 2)} seconds')

    # Append to dict of xr.datasets
    xarray_dict[var] = ds_subset

# print(xarray_dict)

# Initialize an empty list to hold DataFrames
dataframes = []

# looping through all variables for holding in one dataframe
for key, dataset in xarray_dict.items():
    # Extract the data variable
    data_var_name = list(dataset.data_vars.keys())[0]  # Get the first data variable name
    print(data_var_name)
    df_temp = dataset[data_var_name].to_dataframe().reset_index()  # Convert to DataFrame and reset index
    df_temp.rename(columns={data_var_name: key}, inplace=True)  # Rename the column to the variable name
    dataframes.append(df_temp)


merged_df = dataframes[0].drop(columns=['lat','lon'])  # Start with the first DataFrame
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df.drop(columns=['lat','lon']), on='day', how='outer')  # Merge on 'day'

merged_df.rename(columns={'day':'date'}, inplace=True)

# unit conversions - not entirely necessary because these are data driven models,
#  but will keep consistent with what we've done in the past
#  to see the units for each variable, view with xarray_dict[<var_name>].variables
# below are the current units and desired units
#  tmmx: K to C
#  tmmn: K to C
#  pr: mm to m
#  srad: W m-2
#  vs: m/s
#  rmax: % to fraction
#  rmin: % to fraction
#  sph: kg/kg

k_to_c = -273.15
mm_to_m = 0.001
percent_to_frac = 0.01

# Perform the conversions
merged_df['tmmx'] = merged_df['tmmx'] + k_to_c  # Convert tmmx from K to C
merged_df['tmmn'] = merged_df['tmmn'] + k_to_c  # Convert tmmn from K to C
merged_df['pr'] = merged_df['pr'] * mm_to_m     # Convert pr from mm to m
merged_df['rmax'] = merged_df['rmax'] * percent_to_frac  # Convert rmax from % to fraction
merged_df['rmin'] = merged_df['rmin'] * percent_to_frac  # Convert rmin from % to fraction

merged_df.to_csv(r"C:\Users\CL\Documents\GitHub\PywrDRB-ML\data\raw\gridmet_lordville.csv", index=False)

### output folder
#output_path = '2_data_prep/out/'
#merged_df.to_csv("2_data_prep/out/gridmet_lordville.csv", index=False)

