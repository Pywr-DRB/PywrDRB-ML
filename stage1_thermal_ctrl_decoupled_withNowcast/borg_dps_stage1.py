#%%
import sys
import warnings
import pathnavigator

# Suppress sklearn version warnings (as we pickle a newer version of sklearn than hopper)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()

# Add the root directory to Python path so we can import stage1_thermal_ctrl_decoupled_withNowcast module
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


##### Load sys args ####################################################################
# job ID
job_id = "00000"
if len(sys.argv) > 1:
    job_id = sys.argv[1]  # Capture the job ID from the command line

# Policy type
policy_type = None
if len(sys.argv) > 2 and sys.argv[2] != "None":
    policy_type = sys.argv[2]
    
# With storage dynamics
storage_dynamics = False
if len(sys.argv) > 4 and sys.argv[4] != "None":
    storage_dynamics = bool(sys.argv[4])

policy = None
if policy_type == "piecewise":
    from stage1_thermal_ctrl_decoupled_withNowcast.lstm_thermal_ctrl_piecewise import eval_func, n_dim, n_steps
    from src.policies import GeneralizedPiecewiseLinearPolicy
    policy = GeneralizedPiecewiseLinearPolicy(n_dim=n_dim, n_steps=n_steps)
elif policy_type == "gaussian_rbf":
    if storage_dynamics:
        from stage1_thermal_ctrl_decoupled_withNowcast.lstm_thermal_ctrl_gaussian_rbf import eval_func, n_dim, n_basis
    else:
        from stage1_thermal_ctrl_decoupled_withNowcast.lstm_thermal_ctrl_gaussian_rbf_noStorageDynamics import eval_func, n_dim, n_basis
    from src.policies import GaussianRBFPolicy
    policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
elif policy_type == "regression":
    from stage1_thermal_ctrl_decoupled_withNowcast.lstm_thermal_ctrl_regression import eval_func, n_dim, degree
    from src.policies import RegressionPolicy
    policy = RegressionPolicy(n_dim=n_dim, degree=degree)
elif policy_type == "cubic_rbf":
    from stage1_thermal_ctrl_decoupled_withNowcast.lstm_thermal_ctrl_cubic_rbf import eval_func, n_dim, n_basis
    from src.policies import CubicRBFPolicy
    policy = CubicRBFPolicy(n_dim=n_dim, n_basis=n_basis)

# Random seed for Borg
borg_seed = None
if len(sys.argv) > 3 and sys.argv[3] != "None":
    borg_seed = int(sys.argv[3])  # Capture the seed from the command line


##### Set for parallel borg ############################################################
from stage1_thermal_ctrl_decoupled_withNowcast.borg import *

Configuration.startMPI()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#%% Load values from the policy
bounds = policy.bounds
param_names = policy.param_names
obj_names = ["-Jrel", "Jadd", "Jtubr"]
constr_names = []

use_par_names = True
use_obj_names = True
use_constr_names = False

##### Borg settings ####################################################################
nvars = len(bounds)
nobjs = len(obj_names)
nconstrs = 0
nfe = 50_000 # 100_000
runtime_freq = 500
epsilon = 0.01
islands = 1  # 1 = MW, >1 = MM

borg_settings = {
    "numberOfVariables": nvars,
    "numberOfObjectives": nobjs,
    "numberOfConstraints": nconstrs,
    "function": eval_func,
    "epsilons": [epsilon] * nobjs,
    "bounds": bounds,
    "directions": None,  # default is to minimize all objectives. keep this unchanged.
    "seed": borg_seed
}
borg = Borg(**borg_settings)

##### Parallel borg - solvempi #########################################################
exp_folder = f"stage1_nowcast_{policy.name}_{job_id}"
pn.mkdir(f"outputs/{exp_folder}")
pn.outputs.mkdir(f"{exp_folder}/runtimes")
#pn.outputs.mkdir(f"{exp_folder}/checkpoints")

# Runtime
if islands == 1: # Master slave version
    runtime_filename = pn.outputs.get(f"{exp_folder}/runtimes") / f"{job_id}_nfe{nfe}_seed{borg_seed}.runtime"
else:
    # For MMBorg, the filename should include one %d which gets replaced by the island index
    runtime_filename = pn.outputs.get(f"{exp_folder}/runtimes") / f"{job_id}_nfe{nfe}_seed{borg_seed}_%d.runtime"

# Checkpoint (Still have errors)
# newCheckpointFileBase_filename = pn.outputs.get(f"{exp_folder}/checkpoints") / f"{job_id}_nfe{nfe}_seed{borg_seed}"

# Load previous checkpoint (the file must already exist)
# oldCheckpointFile_filename = pn.outputs.dps_borg.get() / "dps_id{job_id}_nfe{nfe}_seed{borg_seed}.checkpoint"

solvempi_settings = {
    "islands": islands,
    "maxTime": None,
    "maxEvaluations": nfe,  # Total NFE is islands * maxEvaluations if island > 1
    "initialization": None,
    "runtime": runtime_filename,
    "allEvaluations": None,
    "frequency": runtime_freq,
    #"newCheckpointFileBase": newCheckpointFileBase_filename, # Output checkpoint
    #"oldCheckpointFile": oldCheckpointFile_filename, # Load checkpoint if uncommented
}
result = borg.solveMPI(**solvempi_settings)

##### Save results #####################################################################
if result is not None:
    # The result will only be returned from one node
    with open(pn.outputs.get(exp_folder) / f"{job_id}_nfe{nfe}_seed{borg_seed}.csv", "w") as file:
        # You may add header here
        headers = []
        if use_par_names:
            headers += param_names
        else:
            headers += [f"var{i+1}" for i in range(nvars)]
        if use_obj_names:
            headers += obj_names
        else:
            headers += [f"obj{i+1}" for i in range(nobjs)]
        if use_constr_names:
            headers += constr_names
        else:
            headers += [f"constr{i+1}" for i in range(nconstrs)]

        file.write(",".join(headers) + "\n")
        result.display(out=file, separator=",")

    # for MOEAFramework-5.0
    with open(pn.outputs.get(exp_folder) / f"{job_id}_nfe{nfe}_seed{borg_seed}.set", "w") as file:
        # You may add header here
        file.write("# Version=5\n")
        file.write(f"# NumberOfVariables={nvars}\n")
        file.write(f"# NumberOfObjectives={nobjs}\n")
        file.write(f"# NumberOfConstraints={nconstrs}\n")
        for i, bound in enumerate(borg_settings["bounds"]):
            file.write(f"# Variable.{i+1}.Definition=RealVariable({bound[0]},{bound[1]})\n")
        if borg_settings.get("directions") is None:
            for i in range(nobjs):
                file.write(f"# Objective.{i+1}.Definition=Minimize\n")
        else:
            for i, direction in enumerate(borg_settings["directions"]):
                if direction == "min":
                    file.write(f"# Objective.{i+1}.Definition=Minimize\n")
                elif direction == "max":
                    file.write(f"# Objective.{i+1}.Definition=Maximize\n")
        file.write(f"//NFE={nfe}\n") # if using check point or multi island, the NFE may not be correct.
        result.display(out=file, separator=" ")
        file.write("#\n")

    # Write the dictionary to a file in a readable format
    with open(pn.outputs.get(exp_folder) / f"{job_id}_nfe{nfe}_seed{borg_seed}.info", 'w') as file:
        file.write("\nBorg settings\n")
        file.write("=================\n")
        for key, value in borg_settings.items():
            file.write(f"{key}: {value}\n")
        file.write("\nBorg solveMPI settings\n")
        file.write("=================\n")
        for key, value in solvempi_settings.items():
            file.write(f"{key}: {value}\n")

    if islands == 1:
        print(f"Master: Completed {job_id}_nfe{nfe}_seed{borg_seed}")
    elif islands > 1:
        print(f"Multi-master controller: Completed {job_id}_nfe{nfe}_seed{borg_seed}")

##### End MPI #########################################################################
Configuration.stopMPI()