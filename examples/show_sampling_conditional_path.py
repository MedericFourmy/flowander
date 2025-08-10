
"""
Demonstrate that the conditional probability samples can be obtained
either by calling path.sample_conditional_path or by sampling from p_simple and 
simulating using the ConditionalVectorFieldODE.
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from flowander.odes import ConditionalVectorFieldODE
from flowander.plot_utils import hist2d_samples
from flowander.simulation import EulerSimulator, record_every
from flowander.config import FIGS_DIR

from example_2d_configs import config

num_samples = 2000
num_timesteps = 52
num_marginals = 5
assert num_timesteps % (num_marginals - 1) == 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "vf_gcpp_gaussian2GM"
# model_name = "vf_gcpp_gaussian2Moons"
# model_name = "vf_gcpp_gaussian2Circles"
# model_name = "vf_gcpp_gaussian2Check"
# model_name = "vf_linear_mlp_gaussian2Checker"
# model_name = "vf_linear_multi_mlp_gaussian2Checker"

p_simple, p_data, path, _ = config(model_name, device)

# sample 1 data point
z = path.p_data.sample(1) # (1,2)

##############
# Setup plots #
##############

fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 3))
axes = axes.reshape(2, num_marginals)
# scale = 6.0
scale = 10.0
legend_size = 12
markerscale = 1.0
fontsize_ylabel = 10
x_bounds = [-scale,scale]
y_bounds = [-scale,scale]


#####################################################################
# Graph conditional probability paths using sample_conditional_path #
#####################################################################
ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
for idx, t in tqdm(enumerate(ts), desc="Conditional Proba, sample_conditional_path"):
    ax = axes[0, idx]
    zz = z.expand(num_samples, -1)
    tt = t.view(1,1).expand(num_samples,1)
    xts = path.sample_conditional_path(zz, tt)
    percentile = min(99 + 2 * torch.sin(t).item(), 100)
    hist2d_samples(samples=xts.cpu(), ax=ax, bins=300, scale=scale, percentile=percentile, alpha=1.0)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'$t={t.item():.2f}$', fontsize=15)
axes[0, 0].set_ylabel("Conditional (from Ground-Truth)", fontsize=fontsize_ylabel)

# Plot z
axes[0,-1].scatter(z[:,0].cpu(), z[:,1].cpu(), marker='*', color='red', s=200, label='z',zorder=20)
axes[0,-1].legend()

######################################################################
# Graph conditional probability paths using conditional_vector_field #
######################################################################
ode = ConditionalVectorFieldODE(path, z)
simulator = EulerSimulator(ode)
ts = torch.linspace(0,1,num_timesteps).to(device)
record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
xts = xts[:,record_every_idxs,:]
for idx in tqdm(range(xts.shape[1]), desc="Conditional Proba, conditional_vector_field"):
    ax = axes[1, idx]
    xx = xts[:,idx,:]
    tt = ts[record_every_idxs[idx]]
    percentile = min(99 + 2 * torch.sin(tt).item(), 100)
    hist2d_samples(samples=xx.cpu(), ax=ax, bins=300, scale=scale, percentile=percentile, alpha=1.0)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'$t={tt.item():.2f}$', fontsize=15)
axes[1, 0].set_ylabel("Conditional (from ODE)", fontsize=fontsize_ylabel)

# Plot z
axes[1,-1].scatter(z[:,0].cpu(), z[:,1].cpu(), marker='*', color='red', s=200, label='z',zorder=20)
axes[1,-1].legend()

path_fig = FIGS_DIR / model_name
path_fig.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGS_DIR / model_name / f"conditional_path_samples_{model_name}.png")
plt.show()