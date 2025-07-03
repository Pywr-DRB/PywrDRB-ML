import pathnavigator
import joblib

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

rf_model = joblib.load(pn.models.get("rf_model.gz"))


rf_model.predict([[20], [15]])
