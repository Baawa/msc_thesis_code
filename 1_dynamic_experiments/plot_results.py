import os
import numpy as np
import matplotlib.pyplot as plt

# How to use. `cd` to folder with results and run the script

icp = np.genfromtxt("results_icp_performance.txt", delimiter="\t")
icp_resampling = np.genfromtxt("results_icp_with_resampling_performance.txt", delimiter="\t")
mcp = np.genfromtxt("results_mcp_performance.txt", delimiter="\t")
mcp_nd = np.genfromtxt("results_node_degree_mcp_performance.txt", delimiter="\t")
ndw = np.genfromtxt("results_node_degree_weighted_cp_performance.txt", delimiter="\t")
ew = np.genfromtxt("results_embedding_weighted_cp_performance.txt", delimiter="\t")

time_steps = icp[0,1:].astype(int)

def plot_dyn(index, title):
  for i in range(0, len(time_steps)):
    x = [0,1,2,3,4]
    y = [icp[index][i+1], mcp[index][i+1], mcp_nd[index][i+1], ndw[index][i+1], ew[index][i+1]]
    plt.title(time_steps[i])
    plt.ylabel(title)
    plt.bar(x, y, tick_label=["ICP", "CCCP", "NCCP", "NWCP", "EWCP"], color=["b", "r", "g", "m", "y"])

    output_dir = f"plots/{title}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{time_steps[i]}.png")
    plt.close()
    

plot_dyn(1, "coverage")
plot_dyn(3, "avg prediction set size")
plot_dyn(5, "Singleton predictions")
plot_dyn(7, "Empty predictions")