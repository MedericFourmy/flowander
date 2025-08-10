##########################
# Play around With These #
##########################
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torch
from tqdm import tqdm

from flowander.odes import LearnedVectorFieldODE
from flowander.plot_utils import hist2d_samples
from flowander.simulation import EulerSimulator, record_every
from flowander.config import WEIGHTS_DIR, FIGS_DIR

from example_2d_configs import config
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_name = "vf_gcpp_gaussian2GM"
# model_name = "vf_gcpp_gaussian2Moons"
# model_name = "vf_gcpp_gaussian2Circles"
# model_name = "vf_gcpp_gaussian2Check"
# model_name = "vf_linear_mlp_gaussian2Checker"
model_name = "vf_linear_multi_mlp_gaussian2Checker"
# model_name = "vf_linear_mlp_circles2Checker"
# model_name = "vf_linear_multi_mlp_circles2Checker"
# kwargs = {"bs_multisample"}

print(model_name)
path_fig = FIGS_DIR / model_name
path_fig.mkdir(parents=True, exist_ok=True)


p_simple, p_data, path, flow_model = config(model_name, device)
model_checkpoint = WEIGHTS_DIR / f"{model_name}.pt"
if model_checkpoint.exists():
    flow_model.load_state_dict(torch.load(model_checkpoint))
flow_model.eval()

parser = argparse.ArgumentParser(description="Show marginal GT vs simulated paths")
parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples")
parser.add_argument("--num_marginals", type=int, default=5, help="Number of marginals to plot")
parser.add_argument("--nfe", type=int, default=500, help="Number of function evaluations (steps)")
parser.add_argument("--dt_animation", type=int, default=4, help="Time taken by the animation")

args = parser.parse_args()

num_samples = args.num_samples
num_marginals = args.num_marginals
nfe = args.nfe

#############
# Simulations
#############
ts_sim = torch.linspace(0,1,nfe).to(device)
record_every_idxs = record_every(len(ts_sim), len(ts_sim) // (num_marginals - 1))

# Record all Groun-Truth Marginals
xt_gt_lst = []
for t in tqdm(ts_sim):
    tt = t.view(1,1).expand(num_samples,1)
    xts = path.sample_marginal_path(tt)
    xt_gt_lst.append(xts)
xts_gt = torch.stack(xt_gt_lst, dim=1)

# Record all generated marginals
ode = LearnedVectorFieldODE(flow_model)
simulator = EulerSimulator(ode)
x0 = path.p_simple.sample(num_samples)
tts_sim = ts_sim.view(1,-1,1).expand(num_samples,-1,1)
xts_gen = simulator.simulate_with_trajectory(x0, tts_sim)

###############
# Setup Plots #
###############

fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
axes = axes.reshape(2, num_marginals)
scale = 6.0

################################
# Plot Ground-Truth Marginals #
################################
for i, idx in enumerate(record_every_idxs):
    ax = axes[0, i]
    hist2d_samples(samples=xts_gt[:,idx,:].cpu(), ax=ax, bins=200, scale=scale, percentile=99, alpha=1.0)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'$t={t.item():.2f}$', fontsize=15)
axes[0, 0].set_ylabel("Ground Truth", fontsize=20)


###########################################
# Graph Marginals of Learned Vector Field #
###########################################
for idx, idx_rec in enumerate(record_every_idxs):
    ax = axes[1, idx]
    xx = xts_gen[:,idx_rec,:]
    hist2d_samples(samples=xx.cpu(), ax=ax, bins=200, scale=scale, percentile=99, alpha=1.0)
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    tt = ts_sim[record_every_idxs[idx]]
    ax.set_title(f'$t={tt.item():.2f}$', fontsize=15)
axes[1, 0].set_ylabel("Learned", fontsize=20)
fig.savefig(FIGS_DIR / model_name / f"marginals_{model_name}_steps={nfe}.png")


################
# Make animation
################


# Create a new figure for the animation
fig_anim, ax_anim = plt.subplots(ncols=2, figsize=(12, 6))
xs = [xts_gt, xts_gen]
titles = ["GT", "ODE"]

def animate(i):
    for k in range(2):
        ax = ax_anim[k]
        ax.clear()
        xx = xs[k][:, i, :]
        hist2d_samples(samples=xx.cpu(), ax=ax, bins=200, scale=scale, percentile=99, alpha=1.0)
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_xticks([])
        ax.set_yticks([])
        tt = ts_sim[i]
        ax.set_title(f'{titles[k]}, $t={tt.item():.2f}$', fontsize=15)

interval = int(1000*args.dt_animation / args.nfe) 
ani = animation.FuncAnimation(
    fig_anim, animate, frames=xts_gen.shape[1], interval=interval, repeat=False
)

# Save the animation as an mp4 file
ani.save(FIGS_DIR / model_name / f"marginals_{model_name}_steps={nfe}.mp4", writer='ffmpeg')

plt.show()