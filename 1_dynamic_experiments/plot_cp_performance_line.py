import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['lines.markersize'] = 14
matplotlib.rcParams['figure.figsize'] = [30, 12]
font = {'size': 20}
matplotlib.rc('font', **font)
alpha = 1

data_icp = np.genfromtxt("results_icp_performance.txt", delimiter="\t")
data_icp_resampling = np.genfromtxt("results_icp_with_resampling_performance.txt", delimiter="\t")
data_mcp = np.genfromtxt("results_mcp_performance.txt", delimiter="\t")
data_mcp_nd = np.genfromtxt("results_node_degree_mcp_performance.txt", delimiter="\t")
data_ndw = np.genfromtxt("results_node_degree_weighted_cp_performance.txt", delimiter="\t")
data_ew = np.genfromtxt("results_embedding_weighted_cp_performance.txt", delimiter="\t")

time_steps = data_icp[0,1:].astype(int)

def plot(index, title, ylabel, ymin=0, ymax=1):
  # conventional
  icp = data_icp[index, 1:]
  icp_resampling = data_icp_resampling[index, 1:]
  mcp = data_mcp[index, 1:]

  # novel
  mcp_nd = data_mcp_nd[index, 1:]
  ndw = data_ndw[index, 1:]
  ew = data_ew[index, 1:]

  X = np.arange(len(time_steps))

  markers = itertools.cycle(['o', 'v', '^', 'X', 's', 'p', 'D'])
  colors = itertools.cycle(['#e6194B', '#f58231', '#ffe119', '#3cb44b', '#43D8CF', '#4363d8'])

  plt.plot(X, icp, label='ICP', marker=next(markers), color=next(colors), ls="-")
  plt.plot(X, icp_resampling, label='ICP-r', marker=next(markers), color=next(colors), ls="-")
  plt.plot(X, mcp, label='CCCP', marker=next(markers), color=next(colors), ls="-")

  plt.plot(X, mcp_nd, label='NCCP', marker=next(markers), color=next(colors), ls="-")
  plt.plot(X, ndw, label='NWCP', marker=next(markers), color=next(colors), ls="-")
  plt.plot(X, ew, label='EWCP', marker=next(markers), color=next(colors), ls="-")

  plt.xticks(X, time_steps)
  plt.ylim(ymin=ymin, ymax=ymax)
  plt.legend(ncol=3)
  plt.title(title)
  plt.xlabel("Time step")
  plt.ylabel(ylabel)
  plt.grid(True)

  plt.tight_layout()
  output_dir = "plots"
  os.makedirs(output_dir, exist_ok=True)
  plt.savefig(f"{output_dir}/{title}.png")
  plt.close()
  # plt.show()

plot(1, "Coverage - OGB Arxiv - Train every time step", "Coverage", ymin=0.85, ymax=1)
plot(3, "Average Prediction Set Size - OGB Arxiv - Train every time step", "Average prediction set size", ymin=1, ymax=7)
plot(5, "Singleton Predictions - OGB Arxiv - Train every time step", "Fraction of singleton predictions", ymin=0, ymax=0.6)
plot(6, "Empty Predictions - OGB Arxiv - Train every time step", "Fraction of empty predictions", ymin=0, ymax=0.04)