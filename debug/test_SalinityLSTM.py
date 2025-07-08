import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.mkdir("outputs/coupled_pywrdrb_with_control")
pn.sc.add("wd", pn.get("outputs/coupled_pywrdrb_with_control"), overwrite=True)

from src.lstm_model import SalinityLSTMModel


ml_model = SalinityLSTMModel(
    model_salinity=pn.models.get() / r"SalinityLSTM_comparison\SalinityLSTM_1d_7d_avg.yml",
    start_date='1979-01-01', end_date='2023-12-31',
    Q_Trenton_lstm_var_name="Q_Trenton_bc",
    Q_Schuylkill_lstm_var_name="Q_Schuylkill_bc",
    debug=True,
    disable_tqdm=True
    )
ml_model.load_data()
ml_model.forecast(t=ml_model.t)
ml_model.update(t=ml_model.t)
ml_model.update_until(date=ml_model.end_date)
