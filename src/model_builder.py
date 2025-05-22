from copy import deepcopy
import pathnavigator
import clt
from tqdm import tqdm
import numpy as np
import pandas as pd

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-LSTMs"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-LSTMs")

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
    if subfolder is not None:
        yml_filename = pn.models.get() / f"{subfolder}/{model_id}.yml"
    else:
        yml_filename = pn.models.get() / f"{model_id}.yml"
    clt.io.to_yaml(data=model_config, file_path=yml_filename)
    return yml_filename
    
def loop_to_train_lstm_models(model_ids, subfolder=None, disable=False):
    
    for model_id in tqdm(model_ids, disable=disable):
        if subfolder is not None:
            config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
        else:
            config_file = pn.models.get(f"{model_id}.yml")
        _ = data_prep(config_file, root_dir)
        lstm = bmi_lstm()
        lstm.initialize(config_file=config_file, train=True, root_dir=pn.get())
        lstm.train_model()

def loop_to_simple_run_lstm_models(model_ids, subfolder=None, disable=False):
    lstms = {}
    for model_id in tqdm(model_ids, disable=disable):
        if subfolder is not None:
            config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
        else:
            config_file = pn.models.get(f"{model_id}.yml")
        lstm = bmi_lstm()
        lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
        lstm.simple_run()
        lstms[model_id] = lstm
    return lstms

def loop_to_eval_lstm_models(lstms, period="all", only_months=None, mode="TempLSTM", disable=False):
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
    
    df_metric = []
    model_ids = []
    for model_id, lstm in tqdm(lstms.items(), disable=disable):        

        model_config = lstm.cfg_bmi
        
        if isinstance(period, tuple):
            sim = lstm.simple_run()[period[0]:period[1]]["mu_ft"]
        elif period == "all":
            sim = lstm.simple_run()["mu_ft"]
        elif period == "train":
            sim = lstm.simple_run()[model_config['start_date_train']:model_config['end_date_train']]["mu_ft"]
        elif period == "val":
            sim = lstm.simple_run()[model_config['start_date_val']:model_config['end_date_val']]["mu_ft"]
        elif period == "test":
            sim = lstm.simple_run()[model_config['start_date_test']:model_config['end_date_test']]["mu_ft"]
        
        if only_months is not None:
            sim = sim[sim.index.month.isin(only_months)]
        
        target = model_config["y_vars"][0]
        target_src = model_config["y_vars_src"][0]
        
        database.loc[database[target_src] != "obs", target] = np.nan
        obs = database.loc[sim.index, target]
        
        sim, obs = clt.dropna_any(sim, obs)
        df_metric.append(clt.metrics.error_metrics(sim=sim, obs=obs))
        model_ids.append(model_id)
        
    df_metric = pd.DataFrame(df_metric, index=model_ids)
    return df_metric

def return_sim_obs_pair(lstms, period="all", only_months=None, mode="TempLSTM", disable=False):
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
    for model_id, lstm in tqdm(lstms.items(), disable=disable):        

        model_config = lstm.cfg_bmi
        
        if isinstance(period, tuple):
            sim = lstm.simple_run()[period[0]:period[1]]["mu_ft"]
        elif period == "all":
            sim = lstm.simple_run()["mu_ft"]
        elif period == "train":
            sim = lstm.simple_run()[model_config['start_date_train']:model_config['end_date_train']]["mu_ft"]
        elif period == "val":
            sim = lstm.simple_run()[model_config['start_date_val']:model_config['end_date_val']]["mu_ft"]
        elif period == "test":
            sim = lstm.simple_run()[model_config['start_date_test']:model_config['end_date_test']]["mu_ft"]
        
        if only_months is not None:
            sim = sim[sim.index.month.isin(only_months)]
        
        target = model_config["y_vars"][0]
        target_src = model_config["y_vars_src"][0]
        
        database.loc[database[target_src] != "obs", target] = np.nan
        obs = database.loc[sim.index, target]
        
        pairs[model_id] = (sim, obs)

    return pairs