from copy import deepcopy
import pathnavigator
import clt
from tqdm import tqdm
import numpy as np
import pandas as pd

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()

from src.torch_bmi import bmi_lstm
from src.prep_data import data_prep

config_template = clt.io.read_yaml(pn.models.config_template.get("config_template.yml"))
db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)
db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)

def make_lstm_model(
        model_id="my_model",
        subfolder=None,
        yml_subsubfolder=None,
        **kwargs
        ):
    
    model_config = deepcopy(config_template)
    
    if subfolder is not None:
        data_file = f"models/{subfolder}/{model_id}/data_{model_id}.npz"
        train_dir = f"models/{subfolder}/{model_id}/"
        pn.mkdir(train_dir)
    else:
        data_file = f"models/{model_id}/data_{model_id}.npz"
        train_dir = f"models/{model_id}/"
        pn.mkdir(train_dir)
    
    model_config["model_id"] = model_id
    model_config["data_file"] = data_file
    model_config["train_dir"] = train_dir
    
    for key, value in kwargs.items():
        if key in model_config:
            model_config[key] = value
        else:
            print(f"Warning: {key} not found in model_config template.")
            model_config[key] = value
    if subfolder is not None and yml_subsubfolder is not None:
        yml_filename = pn.models.get() / f"{subfolder}/{yml_subsubfolder}/{model_id}.yml"
    elif subfolder is not None:
        yml_filename = pn.models.get() / f"{subfolder}/{model_id}.yml"
    else:
        yml_filename = pn.models.get() / f"{model_id}.yml"
    clt.io.to_yaml(data=model_config, file_path=yml_filename)
    return yml_filename
    
def loop_to_train_lstm_models(model_ids, subfolder=None, disable=False, overwrite=False):
    
    for model_id in tqdm(model_ids, disable=disable):
        if subfolder is not None:
            config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
            if not overwrite and (pn.models.get() / f"{subfolder}/{model_id}/{model_id}_c.npy").exists():
                print(f"Model {model_id} already trained. Skipping.")
                continue
        else:
            config_file = pn.models.get(f"{model_id}.yml")
            if not overwrite and (pn.models.get() / f"{model_id}/{model_id}_c.npy").exists():
                print(f"Model {model_id} already trained. Skipping.")
                continue
        
        _ = data_prep(config_file, root_dir)
        lstm = bmi_lstm()
        lstm.initialize(config_file=config_file, train=True, root_dir=pn.get())
        lstm.train_model()

def loop_to_simple_run_lstm_models(model_ids, subfolder=None, mode="TempLSTM", disable=False, overwrite=False):
    """
    Parameters
    ----------
    model_ids : list
        List of model ids to run.
    subfolder : str, optional
        The default is None.
        Subfolder to save the model in. If None, the model is saved in the root folder.
    mode : str, optional
        The default is "TempLSTM".
        Mode to run the model on. Can be "TempLSTM" or "SalinityLSTM".
    disable : bool, optional
        The default is False.
        If True, disable the progress bar.
    overwrite : bool, optional
        The default is False.
        If True, overwrite the existing model.
    """
    if mode == "TempLSTM":
        global db_TempLSTM
        database = db_TempLSTM.copy()
    elif mode == "SalinityLSTM":
        global db_SalinityLSTM
        database = db_SalinityLSTM.copy()
        
    lstms = {}
    for model_id in tqdm(model_ids, disable=disable):
        if subfolder is not None:
            config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
            output_path = pn.models.get() / f"{subfolder}/simple_run_{model_id}.csv"
        else:
            config_file = pn.models.get(f"{model_id}.yml")
            output_path = pn.models.get() / f"simple_run_{model_id}.csv"
        if not overwrite and output_path.exists():
            print(f"simple_run_{model_id}.csv already run. Skipping.")
            continue
        lstm = bmi_lstm()
        lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
        sim = lstm.simple_run()
        # Save the simulation results to a CSV file
        sim.index.name = "date"
        
        model_config = lstm.cfg_bmi
        target = model_config["y_vars"][0]
        target_src = model_config["y_vars_src"][0]
        
        database.loc[database[target_src] != "obs", target] = np.nan
        sim["obs"] = database.loc[sim.index, target]
        
        sim.to_csv(output_path)
        # Save the model to a dictionary
        lstms[model_id] = lstm
    return lstms

def loop_to_eval_lstm_models(model_ids, subfolder=None, mode="TempLSTM", period="all", only_months=None, xbound=None):
    """
    
    Parameters
    ----------
    model_ids : list
        List of model ids to run.
    subfolder : str, optional
        The default is None.
        Subfolder to save the model in. If None, the model is saved in the root folder.
    mode : str, optional
        The default is "TempLSTM".
        Mode to run the model on. Can be "TempLSTM" or "SalinityLSTM".
    period : str, optional
        The default is "all".
        Period to evaluate the model on. Can be "train", "val", "test" or "all". If a tuple is provided, it is used as the start and end date.
    only_months : list, optional
        The default is None.
        List of months to evaluate the model on. If None, all months are used.
    xbound : list, optional
        The default is None.
        Observations within the xbound will be used to evaluate the model. If None, all observations are used.
        xbound = [min, max)
    """
    if mode == "TempLSTM":
        global db_TempLSTM
        database = db_TempLSTM.copy()
    elif mode == "SalinityLSTM":
        global db_SalinityLSTM
        database = db_SalinityLSTM.copy()
    
    df_metric = []
    for model_id in model_ids:     
        if subfolder is not None:
            config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
            simple_run = pd.read_csv(pn.models.get(f"{subfolder}/simple_run_{model_id}.csv"), parse_dates=True, index_col=[0])
            output_path = pn.models.get() / f"{subfolder}/metrics_{period}.csv"
        else:
            config_file = pn.models.get(f"{model_id}.yml")
            simple_run = pd.read_csv(pn.models.get(f"simple_run_{model_id}.csv"), parse_dates=True, index_col=[0])
            output_path = pn.models.get() / f"metrics_{period}.csv"
        model_config = clt.io.read_yaml(config_file)
        
        if isinstance(period, tuple):
            sim = simple_run[period[0]:period[1]]["mu_ft"]
        elif period == "all":
            sim = simple_run["mu_ft"]
        elif period == "train":
            sim = simple_run[model_config['start_date_train']:model_config['end_date_train']]["mu_ft"]
        elif period == "val":
            sim = simple_run[model_config['start_date_val']:model_config['end_date_val']]["mu_ft"]
        elif period == "test":
            sim = simple_run[model_config['start_date_test']:model_config['end_date_test']]["mu_ft"]
        
        if only_months is not None:
            sim = sim[sim.index.month.isin(only_months)]
        
        target = model_config["y_vars"][0]
        target_src = model_config["y_vars_src"][0]
        
        database.loc[database[target_src] != "obs", target] = np.nan
        obs = database.loc[sim.index, target].values
        if xbound is not None:
            if xbound[0] is not None:
                obs[obs < xbound[0]] = np.nan
            if xbound[1] is not None:
                obs[obs >= xbound[1]] = np.nan
        sim, obs = clt.dropna_any(sim, obs)
        df_metric.append(clt.metrics.error_metrics(sim=sim, obs=obs))
        
    df_metric = pd.DataFrame(df_metric, index=model_ids)
    df_metric.index.name = "model_id"
    df_metric.to_csv(output_path)
    return df_metric

def return_sim_obs_pair(model_ids, subfolder=None, period="all", only_months=None, mode="TempLSTM"):
    """
    
    Parameters
    ----------
    lstms : dict
        Dictionary of lstm models.
    period : str, optional
        The default is "all".
        Period to evaluate the model on. Can be "train", "val", "test" or "all". If a tuple is provided, it is used as the start and end date.
    only_months : list, optional
        The default is None.
        List of months to evaluate the model on. If None, all months are used.
    mode : str, optional
        The default is "TempLSTM".
        Mode to evaluate the model on. Can be "TempLSTM" or "SalinityLSTM".
    disable : bool, optional
        The default is False.
        If True, disable the progress bar.
    """
    if mode == "TempLSTM":
        global db_TempLSTM
        database = db_TempLSTM.copy()
    elif mode == "SalinityLSTM":
        global db_SalinityLSTM
        database = db_SalinityLSTM.copy()
    
    pairs = {}
    for model_id in model_ids:     
        if subfolder is not None:
            config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
            simple_run = pd.read_csv(pn.models.get(f"{subfolder}/simple_run_{model_id}.csv"), parse_dates=True, index_col=[0])
        else:
            config_file = pn.models.get(f"{model_id}.yml")
            simple_run = pd.read_csv(pn.models.get(f"simple_run_{model_id}.csv"), parse_dates=True, index_col=[0])
        model_config = clt.io.read_yaml(config_file)
        
        if isinstance(period, tuple):
            sim = simple_run[period[0]:period[1]]["mu_ft"]
        elif period == "all":
            sim = simple_run["mu_ft"]
        elif period == "train":
            sim = simple_run[model_config['start_date_train']:model_config['end_date_train']]["mu_ft"]
        elif period == "val":
            sim = simple_run[model_config['start_date_val']:model_config['end_date_val']]["mu_ft"]
        elif period == "test":
            sim = simple_run[model_config['start_date_test']:model_config['end_date_test']]["mu_ft"]
        
        if only_months is not None:
            sim = sim[sim.index.month.isin(only_months)]
        
        target = model_config["y_vars"][0]
        target_src = model_config["y_vars_src"][0]
        
        database.loc[database[target_src] != "obs", target] = np.nan
        obs = database.loc[sim.index, target]
        
        pairs[model_id] = (sim, obs)

    return pairs

def get_rf_model():
    try:
        rf_model = clt.io.read_joblib(pn.models.get("rf_model.gz"))
    except:
        from sklearn.ensemble import RandomForestRegressor
        global db_TempLSTM
        database = db_TempLSTM.copy()
        df = database[["QobsTavg_T_L", 'QobsTmax_T_L']].dropna()
        X = df["QobsTavg_T_L"].values
        y = df['QobsTmax_T_L'].values
        X, y = clt.dropna_any(X, y)
        X = X.reshape(-1, 1)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        clt.io.to_joblib(rf_model, pn.models.get()/"rf_model.gz")
    return rf_model
    
def return_T_L_lstm(lstm1, lstm2, map_to_Tmax=True):
    T_C = lstm1.simple_run()["mu_ft"].to_frame("T_C")
    T_i = lstm2.simple_run()["mu_ft"].to_frame("T_i")
    
    global db_TempLSTM
    database = db_TempLSTM.copy()
    
    prefix = lstm1.cfg_bmi["y_vars"][0].split("_")[0]
    Q_C = database.loc[T_C.index, prefix + "_Q_C"].to_frame("Q_C")
    Q_i = database.loc[T_i.index, prefix + "_Q_i"].to_frame("Q_i")
    Q_L = database.loc[T_C.index, prefix + "_Q_L"].to_frame("Q_L")
    
    Tavg = (T_C["T_C"]*Q_C["Q_C"] + T_i["T_i"]*Q_i["Q_i"])/Q_L["Q_L"]
    Tavg = Tavg.to_frame("Tavg")
    
    if map_to_Tmax:
        rf_model = get_rf_model()
        Tmax = rf_model.predict(Tavg.values.reshape(-1, 1))
        Tavg["T_L"] = Tmax
        Tavg.loc[pd.isna(Tavg["Tavg"]), "T_L"] = np.nan
    df = pd.concat([T_C, T_i, Tavg], axis=1)
    return df

def return_T_L(T_C, T_i, prefix="QbcTavg", map_to_Tmax=True, src_col="tavg_water_src"):
    T_C = T_C.to_frame("T_C")
    T_i = T_i.to_frame("T_i")
    
    global db_TempLSTM
    database = db_TempLSTM.copy()
    
    Q_C = database.loc[T_C.index, prefix + "_Q_C"].to_frame("Q_C")
    Q_i = database.loc[T_i.index, prefix + "_Q_i"].to_frame("Q_i")
    Q_L = database.loc[T_C.index, prefix + "_Q_L"].to_frame("Q_L")
    
    Tavg = (T_C["T_C"]*Q_C["Q_C"] + T_i["T_i"]*Q_i["Q_i"])/Q_L["Q_L"]
    Tavg = Tavg.to_frame("Tavg")
    
    T_L_src = database.loc[Tavg.index, [src_col]]
    
    if map_to_Tmax:
        rf_model = get_rf_model()
        Tmax = rf_model.predict(Tavg.values.reshape(-1, 1))
        Tavg["T_L"] = Tmax
        Tavg.loc[T_L_src[src_col] != "obs", "T_L"] = np.nan
    df = pd.concat([T_C, T_i, Tavg], axis=1)
    return df

def return_sim_obs_pair_for_T_L(lstm1, lstm2):
    global db_TempLSTM
    database = db_TempLSTM.copy()
    
    df_sim = return_T_L(lstm1, lstm2, map_to_Tmax=True)
    prefix = lstm1.cfg_bmi["y_vars"][0].split("_")[0]
    
    database = database.loc[df_sim.index, :]
    df_obs = [
        database[prefix + "_T_C"].to_frame("T_C"),
        database[prefix + "_T_i"].to_frame("T_i"),
        database[prefix + "_T_L"].to_frame("Tavg"),
        database[prefix.replace("avg", "max") + "_T_L"].to_frame("T_L"),
        ]
    df_obs = pd.concat(df_obs, axis=1)
    return df_obs, df_sim
    
def eval_TempLSTM(lstm1, lstm2, period="all", only_months=None, disable=False):
    """
    
    Parameters
    ----------
    lstms : dict
        Dictionary of lstm models.
    period : str, optional
        The default is "all".
        Period to evaluate the model on. Can be "train", "val", "test" or "all". If a tuple is provided, it is used as the start and end date.
    only_months : list, optional
        The default is None.
        List of months to evaluate the model on. If None, all months are used.
    mode : str, optional
        The default is "TempLSTM".
        Mode to evaluate the model on. Can be "TempLSTM" or "SalinityLSTM".
    disable : bool, optional
        The default is False.
        If True, disable the progress bar.
    """
    global db_TempLSTM
    database = db_TempLSTM.copy()
    
    df_sim = return_T_L(lstm1, lstm2, map_to_Tmax=True)
    prefix = lstm1.cfg_bmi["y_vars"][0].split("_")[0]
    
    database = database.loc[df_sim.index, :]
    df_obs = [
        database[prefix + "_T_C"].to_frame("T_C"),
        database[prefix + "_T_i"].to_frame("T_i"),
        database[prefix + "_T_L"].to_frame("Tavg"),
        database[prefix.replace("avg", "max") + "_T_L"].to_frame("T_L"),
        ]
    df_obs = pd.concat(df_obs, axis=1)
    
    model_config1 = lstm1.cfg_bmi
    model_config2 = lstm2.cfg_bmi
        
    df_metric = []
    for col in df_sim:
        if col == "T_C":
            model_config = model_config1
        else:
            model_config = model_config2
        
        obs = df_obs[col]
        sim = df_sim[col]
        target = model_config["y_vars"][0]
        target_src = model_config["y_vars_src"][0]
            
        if isinstance(period, tuple):
            obs = obs[period[0]:period[1]]
            sim = sim[period[0]:period[1]]
        elif period == "all":
            obs = obs
            sim = sim            
        elif period == "train":
            obs = obs[model_config['start_date_train']:model_config['end_date_train']]
            sim = sim[model_config['start_date_train']:model_config['end_date_train']]
        elif period == "val":
            obs = obs[model_config['start_date_val']:model_config['end_date_val']]
            sim = sim[model_config['start_date_val']:model_config['end_date_val']]
        elif period == "test":
            obs = obs[model_config['start_date_test']:model_config['end_date_test']]
            sim = sim[model_config['start_date_test']:model_config['end_date_test']]
        
        obs.loc[database[target_src] != "obs"] = np.nan
        
        if only_months is not None:
            obs = obs[obs.index.month.isin(only_months)]
            sim = sim[sim.index.month.isin(only_months)]
        
        sim, obs = clt.dropna_any(sim, obs)
        df_metric.append(clt.metrics.error_metrics(sim=sim, obs=obs))
        
    df_metric = pd.DataFrame(df_metric, index=df_sim.columns)
    return df_metric