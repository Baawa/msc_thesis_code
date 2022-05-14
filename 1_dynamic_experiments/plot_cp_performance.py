import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['lines.markersize'] = 14
matplotlib.rcParams['figure.figsize'] = [40, 12]
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

def plot(index, title, ymin=0, ymax=1):
  # conventional
  icp = data_icp[index, 1:]
  icp_resampling = data_icp_resampling[index, 1:]
  mcp = data_mcp[index, 1:]

  # novel
  mcp_nd = data_mcp_nd[index, 1:]
  ndw = data_ndw[index, 1:]
  ew = data_ew[index, 1:]

  X = np.arange(len(time_steps))
  bar_width = 0.14

  hatches = itertools.cycle(['/', '+', '//', '-', 'x', '\\', '*', 'o', 'O', '.'])
  colors = itertools.cycle(['#e6194B', '#f58231', '#ffe119', '#3cb44b', '#43D8CF', '#4363d8'])

  plt.bar(X, icp, width=bar_width, label='ICP', hatch=next(
      hatches), edgecolor='black', color=next(colors))
  plt.bar(X + 1 * bar_width, icp_resampling, width=bar_width, label='ICP-r',
          hatch=next(hatches), edgecolor='black', color=next(colors))
  plt.bar(X + 2 * bar_width, mcp, width=bar_width, label='CCCP',
          hatch=next(hatches), edgecolor='black', color=next(colors))

  plt.bar(X + 3 * bar_width, mcp_nd, width=bar_width, label='NCCP',
          hatch=next(hatches), edgecolor='black', color=next(colors))
  plt.bar(X + 4 * bar_width, ndw, width=bar_width, label='NWCP',
          hatch=next(hatches), edgecolor='black', color=next(colors))
  plt.bar(X + 5 * bar_width, ew, width=bar_width, label='EWCP',
          hatch=next(hatches), edgecolor='black', color=next(colors))

  plt.xticks([i + (bar_width*3) for i in X], time_steps)
  # plt.yscale('log')
  # plt.ylim(ymax=10000)
  plt.ylim(ymin=ymin, ymax=ymax)
  plt.legend(ncol=3)
  plt.title(title)
  plt.xlabel("Time step")
  plt.ylabel("Performance")
  # plt.axisbelow(True)
  plt.grid(True)

  # plt.savefig(f"plots/{title}.png", bbox_inches='tight')
  plt.savefig(f"plots/{title}.png")
  plt.close()
  # plt.show()

plot(1, "Coverage - OGB Arxiv - Train every time step", ymin=0.8, ymax=1)
plot(3, "Average Prediction Set Size - OGB Arxiv - Train every time step", ymin=0, ymax=10)
plot(5, "Singleton Predictions - OGB Arxiv - Train every time step", ymin=0, ymax=1)
plot(6, "Empty Predictions - OGB Arxiv - Train every time step", ymin=0, ymax=0.1)