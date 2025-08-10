import argparse

from matplotlib import pyplot as plt
import torch

from flowander.odes import ConditionalVectorFieldODE, LearnedVectorFieldODE
from flowander.sdes import ConditionalVectorFieldSDE, ScoreFromVectorField, LangevinFlowSDE
from flowander.plot_utils import imshow_density
from flowander.distributions import Density
from flowander.simulation import EulerSimulator, EulerMaruyamaSimulator, record_every
from flowander.config import WEIGHTS_DIR, FIGS_DIR
from example_2d_configs import config


parser = argparse.ArgumentParser(description="Show conditional vector field")
parser.add_argument('--use-sde', action='store_true', help='Use SDE instead of ODE')
parser.add_argument('--use-mlp', action='store_true', help='Use MLP instead of conditional path')
args = parser.parse_args()


# simulation params
num_samples = 1000
num_timesteps = 1000
num_marginals = 3
sigma = 0.5


########################
# Setup path and plot  #
########################
# Constants for the duration of our use of Gaussian conditional probability paths, to avoid polluting the namespace...
# model_name = "vf_gcpp_gaussian2GM"
# model_name = "vf_gcpp_gaussian2Moons"
# model_name = "vf_gcpp_gaussian2Circles"
# model_name = "vf_gcpp_gaussian2Check"
# model_name = "vf_linear_mlp_gaussian2Checker"
model_name = "vf_linear_multi_mlp_gaussian2Checker"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p_simple, p_data, path, flow_model = config(model_name, device)

# Setup figure
fig, axes = plt.subplots(1,3, figsize=(36, 12))
scale = 15.0
legend_size = 12
markerscale = 1.0
bins = 200 # density histogram bins
x_bounds = [-scale,scale]
y_bounds = [-scale,scale]

# Sample conditioning variable z
torch.cuda.manual_seed(1)
z = path.sample_conditioning_variable(1) # (1,2)

###############################################
# Graph samples from conditional vector field #
###############################################
ax = axes[1]

ax.set_xlim(*x_bounds)
ax.set_ylim(*y_bounds)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Samples from Conditional vector field', fontsize=12)
ax.scatter(z[:,0].cpu(), z[:,1].cpu(), marker='*', color='red', s=200, label='z',zorder=20) # Plot z

# Plot source and target
bins = 200
imshow_density(p_simple, x_bounds, y_bounds, bins, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
if isinstance(p_data, Density):
    imshow_density(p_data, x_bounds, y_bounds, bins, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

if args.use_mlp:
    flow_model.load_state_dict(torch.load(WEIGHTS_DIR / f"{model_name}.pt"))
    flow_model.eval()

# Construct integrator and plot trajectories
if args.use_sde:
    if args.use_mlp:
        score_model = ScoreFromVectorField(flow_model, path.alpha, path.beta)
        sde = LangevinFlowSDE(flow_model, score_model, sigma)
    else:
        sde = ConditionalVectorFieldSDE(path, z, sigma)
    simulator = EulerMaruyamaSimulator(sde)
else:
    if args.use_mlp:
        ode = LearnedVectorFieldODE(flow_model)
    else:
        ode = ConditionalVectorFieldODE(path, z)
    simulator = EulerSimulator(ode)

x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)

# Extract every n-th integration step to plot
every_n = record_every(num_timesteps, num_timesteps // num_marginals)
xts_every_n = xts[:,every_n,:] # (bs, nts // n, dim)
ts_every_n = ts[0,every_n] # (nts // n,)
for plot_idx in range(xts_every_n.shape[1]):
    tt = ts_every_n[plot_idx].item()
    ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), marker='.', alpha=0.5, label=f't={tt:.2f}')
ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)


###################################################
# Graph Ground-Truth Conditional Probability Path #
###################################################
ax = axes[0]

ax.set_xlim(*x_bounds)
ax.set_ylim(*y_bounds)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Ground-Truth Conditional Probability Path', fontsize=12)
ax.scatter(z[:,0].cpu(), z[:,1].cpu(), marker='*', color='red', s=200, label='z',zorder=20) # Plot z


for plot_idx in range(xts_every_n.shape[1]):
    tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
    zz = z.expand(num_samples, 2)
    marginal_samples = path.sample_conditional_path(zz, tt)
    ax.scatter(marginal_samples[:,0].detach().cpu(), marginal_samples[:,1].detach().cpu(), marker='.', alpha=0.5, label=f't={tt[0,0].item():.2f}')

# Plot source and target
imshow_density(density=p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=bins, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
if isinstance(p_data, Density):
    imshow_density(density=p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=bins, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)


##################################################
# Graph Trajectories of Conditional vector field #
##################################################
ax = axes[2]

ax.set_xlim(*x_bounds)
ax.set_ylim(*y_bounds)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Trajectories of Conditional vector field', fontsize=12)
ax.scatter(z[:,0].cpu(), z[:,1].cpu(), marker='*', color='red', s=200, label='z',zorder=20) # Plot z


# Plot source and target
imshow_density(density=p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=bins, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
if isinstance(p_data, Density):
    imshow_density(density=p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=bins, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

nb_traj = 100
for traj_idx in range(nb_traj):
    ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')
ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

path_fig = FIGS_DIR / model_name
path_fig.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGS_DIR / model_name / f"conditional_path_samples_{model_name}.png")
plt.show()