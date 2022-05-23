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

data = np.genfromtxt("results_graphsage_model_performance.txt", delimiter="\t")

time_steps = data[0,1:].astype(int)

def plot(title, ymin=0, ymax=1):
  accuracies = data[1, 1:]
  macro_f1 = data[3, 1:]
  
  X = np.arange(len(time_steps))

  plt.plot(X, accuracies, label='Accuracy', marker="v", color="#4363d8", ls="-")
  plt.plot(X, macro_f1, label='Macro F1', marker="o", color="#e6194B", ls="-")
  
  plt.xticks(X, time_steps)
  # plt.yscale('log')
  # plt.ylim(ymax=10000)
  plt.ylim(ymin=ymin, ymax=ymax)
  plt.legend(ncol=3)
  plt.title(title)
  plt.xlabel("Time step")
  plt.ylabel("Score")
  # plt.axisbelow(True)
  plt.grid(True)

  # plt.savefig(f"plots/{title}.png", bbox_inches='tight')
  plt.tight_layout()
  output_dir = "plots"
  os.makedirs(output_dir, exist_ok=True)
  plt.savefig(f"{output_dir}/{title}.png")
  plt.close()
  # plt.show()

plot("GraphSAGE accuracy - Bitcoin Elliptic - Train once", ymin=0, ymax=1.1)