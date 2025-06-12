import pandas as pd
import pathnavigator
if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

class USGSNWIS():
    def __init__(self):
        """
        USGS NWIS class for accessing USGS National Water Information System data.
        
        URL Genration Tool: waterservices.usgs.gov/test-tools/
        USGS code: https://help.waterdata.usgs.gov/codes-and-parameters/parameters
        USGS code (Physical): https://help.waterdata.usgs.gov/code/parameter_cd_query?fmt=rdb&group_cd=PHY&inline=true
        """
        pass

    def form_url(self, site, start_date, end_date, parameter_code="all", service="dv", format="rdb"):
        """
        Form a URL for the USGS NWIS service.
        
        Parameters
        ==========
        site : str
            Site number or site name.
        start_date : str
            Start date in the format YYYY-MM-DD.
        end_date : str
            End date in the format YYYY-MM-DD.
        parameter_code : str
            Parameter code for the data.
            USGS code: https://help.waterdata.usgs.gov/codes-and-parameters/parameters
        service : str, optional
            - "iv": instantaneous values.
            - "dv": daily values (default).
            
        format : str, optional
            Format of the output data. 
            - "json": JSON format.
            - "rdb": tab-separated values (default).
            - "excel": Excel format.
            
        Returns
        =======
        str
            Formed URL for the USGS NWIS service.
        """
        base_url = "https://waterservices.usgs.gov/nwis/"
        url = f"{base_url}{service}?format={format}&sites={site}&startDT={start_date}&endDT={end_date}&parameterCd={parameter_code}"
        return url
    
    def _raw_text2frame(self, raw_text):
        from io import StringIO
        import pandas as pd
        lines = [
            line for line in raw_text.splitlines()
            if not line.startswith('#') 
        ]
        del lines[1]
        data_io = StringIO('\n'.join(lines))
        df = pd.read_csv(data_io, sep='\t')
        return df
    
    def download(self, url, to_frame=False):
        """
        Download data from the given URL.
        
        Parameters
        ==========
        url : str
            URL to download data from.
            
        Returns
        =======
        str
            Path to the downloaded file.
        """
        import requests
        
        response = requests.get(url)
        if response.status_code == 200:
            if to_frame:
                return self._raw_text2frame(response.text)
            else:
                return response
        else:
            raise Exception(f"Failed to download data. Status code: {response.status_code}")

#%% Cannonsville
nwis = USGSNWIS()
url = nwis.form_url(site="01425000", start_date="1963-10-01", end_date="2024-12-31")#, parameter_code="00060")#"00060")
raw = nwis.download(url, to_frame=False).text
data = nwis.download(url, to_frame=True)
data["105329_00060_00003"] *= 0.64632 # cfs to mgd
data.index = pd.to_datetime(data["datetime"])
data = data[['105330_00010_00001', '105331_00010_00002', '105332_00010_00003', '105329_00060_00003']]
cols = ['tmmx_water_cannonsville', 'tmmn_water_cannonsville', 'tavg_water_cannonsville', 'discharge_cannonsville']
data.columns = cols
new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
data = data.reindex(new_index) # Make sure dates are consecutive
data.index.name = "date"
for col in cols:
    col_src = col + "_src"
    data[col_src] = "obs"
    data.loc[data[col].isna(), col_src] = "linear_interpolation"

# Add dwelling and other filling     
data = data.interpolate(method='linear')
data.to_csv(pn.data.raw.get() / "nwis_Cannonsville_degC_mgd.csv")

#%% Lordville
nwis = USGSNWIS()
url = nwis.form_url(site="01427207", start_date="1963-10-01", end_date="2024-12-31")#, parameter_code="00010")#"00060")
raw = nwis.download(url, to_frame=False).text
data = nwis.download(url, to_frame=True)
data["105344_00060_00003"] *= 0.64632 # cfs to mgd
data.index = pd.to_datetime(data["datetime"])
data = data[['105345_00010_00001', '105346_00010_00002', '105347_00010_00003', '105344_00060_00003']]
cols = ['tmmx_water_lordville', 'tmmn_water_lordville', 'tavg_water_lordville', 'discharge_lordville']
data.columns = cols
new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
data = data.reindex(new_index) # Make sure dates are consecutive
data.index.name = "date"
for col in cols:
    col_src = col + "_src"
    data[col_src] = "obs"
    data.loc[data[col].isna(), col_src] = "linear_interpolation"

# Add dwelling and other filling         
data = data.interpolate(method='linear')
data.to_csv(pn.data.raw.get() / "nwis_Lordville_degC_mgd.csv")

#%%
df_raw = pd.read_csv(pn.data.raw.get() / "sf_data.csv", index_col=0, parse_dates=True)

df = pd.DataFrame(index=df_raw.index)
df.index.name = "date"
df["saltfront"] = df_raw["7dmaRM"]
# LSTM can acommondate nan
# df = df.interpolate(method='time')
# df = df.interpolate(method='time').ffill().bfill()
df["saltfront_src"] = "other"
df.loc[df_raw["Flag"] == "A", "saltfront_src"] = "obs"
df.loc[df_raw["Flag"] == "P", "saltfront_src"] = "obs"

nwis = USGSNWIS()
url = nwis.form_url(site="01463500", start_date="1963-10-01", end_date="2024-12-31", parameter_code="00060", service="dv", format="rdb")
raw_text = nwis.download(url=url).text
dff = nwis.download(url=url, to_frame=True)
dff.index = pd.to_datetime(dff["datetime"])
df["01463500"] = dff["97504_00060_00003"] * 0.64632 # cfs to mgd

nwis = USGSNWIS()
url = nwis.form_url(site="01474500", start_date="1963-10-01", end_date="2024-12-31", parameter_code="00060", service="dv", format="rdb")
raw_text = nwis.download(url=url).text
dff = nwis.download(url=url, to_frame=True)
dff.index = pd.to_datetime(dff["datetime"])
df["01474500"] = dff["118500_00060_00003"] * 0.64632 # cfs to mgd

df.to_csv(pn.data.raw.get() / "salt_front_data.csv")