#%%
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.policies import GeneralizedPiecewiseLinearPolicy, RegressionPolicy, GaussianRBFPolicy, CubicRBFPolicy

#%% GeneralizedPiecewiseLinearPolicy example
policy = GeneralizedPiecewiseLinearPolicy(n_dim=2, n_steps=3)
params = policy.gen_params(n=1, seed=42)[0]
policy.set_params(*params)
policy.plot_response_surface()
policy.plot_response_contour()
policy.plot_response_line()

#%% RegressionPolicy example
policy = RegressionPolicy(n_dim=2, degree=2)
params = policy.gen_params(n=1, seed=3)[0]
policy.set_params(*params)
policy.plot_response_surface()
policy.plot_response_contour()
policy.plot_response_line()

#%% GaussianRBFPolicy example
policy = GaussianRBFPolicy(n_dim=2, n_basis=2)
params = policy.gen_params(n=1, seed=42)[0]
policy.set_params(*params)
policy.plot_response_surface()
policy.plot_response_contour()
policy.plot_response_line()

#%% CubicRBFPolicy example
policy = CubicRBFPolicy(n_dim=2, n_basis=2)
params = policy.gen_params(n=1, seed=4)[0]
policy.set_params(*params)
policy.plot_response_surface()
policy.plot_response_contour()
policy.plot_response_line()

