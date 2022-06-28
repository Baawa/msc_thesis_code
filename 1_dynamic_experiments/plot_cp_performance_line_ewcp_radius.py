import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['lines.markersize'] = 14
matplotlib.rcParams['figure.figsize'] = [30, 12]
font = {'size': 32}
matplotlib.rc('font', **font)
alpha = 1

data_ew300 = np.genfromtxt("results_300_embedding_weighted_cp_performance.txt", delimiter="\t")
data_ew1000 = np.genfromtxt("results_1000_embedding_weighted_cp_performance.txt", delimiter="\t")
data_ew300r = np.genfromtxt("results_300_embedding_weighted_cp_with_resampling_performance.txt", delimiter="\t")
data_ew1000r = np.genfromtxt("results_1000_embedding_weighted_cp_with_resampling_performance.txt", delimiter="\t")

time_steps = data_ew300[0,1:].astype(int)

def plot(index, title, ylabel, ymin=0, ymax=1, plot_confidence_level=False):
  # 300
  ew300 = data_ew300[index, 1:]
  ew300r = data_ew300r[index, 1:]

  # 1000
  ew1000 = data_ew1000[index, 1:]
  ew1000r = data_ew1000r[index, 1:]

  X = np.arange(len(time_steps))

  red = '#e6194B'
  markers = itertools.cycle(['o', 'v', '^', 'X', 's', 'p', 'D'])
  colors = itertools.cycle(['#f58231', '#ffe119', '#3cb44b', '#4363d8'])

  plt.plot(X, ew300, label='Radius: 300', marker=next(markers), color=next(colors), ls="-")
  plt.plot(X, ew300r, label='Radius: 300 with resampling', marker=next(markers), color=next(colors), ls="-")

  plt.plot(X, ew1000, label='Radius: 1000', marker=next(markers), color=next(colors), ls="-")
  plt.plot(X, ew1000r, label='Radius: 1000 with resampling', marker=next(markers), color=next(colors), ls="-")
  
  if plot_confidence_level == True:
    plt.plot(X, np.ones(X.shape) * 0.95, label='Confidence level', color=red, alpha=0.8, ls="--", linewidth=6)

  #plt.xticks(np.arange(0, len(time_steps), step=5))
  plt.xticks(X, time_steps)
  plt.ylim(ymin=ymin, ymax=ymax)
  plt.legend(ncol=2)
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

plot(1, "Coverage - OGB Arxiv - EWCP radius comparison", "Coverage", ymin=0.7, ymax=1, plot_confidence_level=True)
plot(3, "Average Prediction Set Size - OGB Arxiv - EWCP radius comparison", "Average prediction set size", ymin=2, ymax=30)
plot(5, "Singleton Predictions - OGB Arxiv - EWCP radius comparison", "Fraction of singleton predictions", ymin=0, ymax=0.4)
plot(6, "Empty Predictions - OGB Arxiv - EWCP radius comparison", "Fraction of empty predictions", ymin=0, ymax=0.05)