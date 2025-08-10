import time
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import ot
from flowander.joint_sampler import JointSampler

# N bakeries and N coffee places: find optimal assignment that minimizes 2D

# 1) generate some data
N = 5
D = 2
seed = 1
# rng = np.random.default_rng(seed)
rng = np.random.RandomState(seed)  # legacy but needed by ot
mu1 = np.array([0, 0])
C1 = np.array([[1, 0], [0, 1]])
mu2 = np.array([0, 0])
# mu2 = np.array([4, 4])
C2 = np.array([[1, -0.8], [-0.8, 1]])
x1 = ot.datasets.make_2D_samples_gauss(N, mu1, C1, random_state=rng)
x2 = ot.datasets.make_2D_samples_gauss(N, mu2, C2, random_state=rng)


C = cdist(x1, x2, metric="euclidean")
num_threads = 1
# method = "exact_emd"
# method = "linear_sum_assignment"
method = "condot"
# method = "partial"
jsampler = JointSampler(method, num_threads=num_threads)

t1 = time.time()
rows, cols = jsampler.sample_plan(C)
print("sample_map", 1e3*(time.time() - t1))


fig, axes = plt.subplots(1)
axes.plot(x1[:,0], x1[:,1], "rx")
axes.plot(x2[:,0], x2[:,1], "bx")
for i, j in zip(rows, cols):
    axes.plot([x1[i, 0], x2[j, 0]], [x1[i, 1], x2[j, 1]], 'g-')

plt.show()