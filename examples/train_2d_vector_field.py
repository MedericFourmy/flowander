import torch
import numpy as np
import matplotlib.pyplot as plt

from flowander.training import ConditionalFlowMatchingTrainer
from flowander.config import WEIGHTS_DIR, FIGS_DIR
from example_2d_configs import config

#qwdwqdqwdqw
# batch_size = 1000
# num_epochs = 5000
# linear flows
batch_size = 2000
num_epochs = 10000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_name = "vf_gcpp_gaussian2GM"
# model_name = "vf_gcpp_gaussian2Moons"
# model_name = "vf_gcpp_gaussian2Circles"
# model_name = "vf_gcpp_gaussian2Check"
# model_name = "vf_linear_mlp_gaussian2Checker"
model_name = "vf_linear_multi_mlp_gaussian2Checker"
# model_name = "vf_linear_mlp_circles2Checker"
# model_name = "vf_linear_multi_mlp_circles2Checker"

p_simple, p_data, path, flow_model = config(model_name, device)

trainer = ConditionalFlowMatchingTrainer(path, flow_model)
losses = trainer.train(num_epochs=num_epochs, device=device, lr=1e-3, batch_size=batch_size)

plt.figure()
plt.plot(losses)
plt.title(f"Training loss {model_name}")
path_fig = FIGS_DIR / model_name
path_fig.mkdir(parents=True, exist_ok=True)
fig_path = FIGS_DIR / model_name / f"training_loss_{model_name}.png"
print("Saving", fig_path)
plt.savefig(fig_path)
torch.save(flow_model.state_dict(), WEIGHTS_DIR / f"{model_name}.pt")
