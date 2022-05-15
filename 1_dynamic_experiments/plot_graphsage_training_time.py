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

data = np.genfromtxt("graphsage_training_time_time.txt", delimiter="\t")

time_steps = data[0].astype(int)

def plot(title, ymin=0, ymax=1):
  training_times = data[1]
  
  X = np.arange(len(time_steps))

  total_training_times = np.cumsum(training_times)

  plt.plot(X, training_times, label='Added training time at each time step', marker="v", color="#e6194B", ls="-")
  plt.plot(X, total_training_times, label='Total training time', marker="o", color="#4363d8", ls="-")
  plt.fill_between(X, total_training_times, np.zeros(len(time_steps)), alpha=0.1, color="#4363d8")
  
  plt.xticks(X, time_steps)
  plt.ylim(ymin=ymin, ymax=ymax)
  plt.legend(ncol=3)
  plt.title(title)
  plt.xlabel("Time step")
  plt.ylabel("Seconds")
  plt.grid(True)

  plt.tight_layout()
  output_dir = "plots"
  os.makedirs(output_dir, exist_ok=True)
  plt.savefig(f"{output_dir}/{title}.png")
  plt.close()
  # plt.show()

plot("GraphSAGE training time - OGB Arxiv", ymin=1, ymax=200)