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

# conventional
icp = 0.00188
icp_retrain = 0.00311

icp_resampling = 0.00311
icp_resampling_retrain = 0.00312

mcp = 0.00167
mcp_retrain = 0.00211

# novel
mcp_nd = 0.00171
mcp_nd_retrain = 0.00237

ndw = 0.00255
ndw_retrain = 0.00415

ew = 0.00327
ew_retrain = 0.00496

models = ["ICP", "ICP-r", "CCCP", "NCCP", "NWCP", "EWCP"]

def plot(title, ymin=0, ymax=1):
  X = np.arange(len(models))
  bar_width = 0.33

  hatch1 = "/"
  hatch2 = "+"
  red = "#e6194B"
  blue = "#4363d8"
  hatches = itertools.cycle(['/', '+', '//', '-', 'x', '\\', '*', 'o', '.'])

  plt.bar(0, icp * 1000, width=bar_width, label='ICP', edgecolor='black', color=blue, hatch=next(hatches))
  plt.bar(0 + bar_width, icp_retrain * 1000, width=bar_width, label='ICP with retraining', edgecolor='black', color=red, hatch=next(hatches))

  plt.bar(1, icp_resampling * 1000, width=bar_width, label='ICP-r', edgecolor='black', color=blue, hatch=next(hatches))
  plt.bar(1 + bar_width, icp_resampling_retrain * 1000, width=bar_width, label='ICP-r with retraining', edgecolor='black', color=red, hatch=next(hatches))
  
  plt.bar(2, mcp * 1000, width=bar_width, label='CCCP', edgecolor='black', color=blue, hatch=next(hatches))
  plt.bar(2 + bar_width, mcp_retrain * 1000, width=bar_width, label='CCCP with retraining', edgecolor='black', color=red, hatch=next(hatches))

  plt.bar(3, mcp_nd * 1000, width=bar_width, label='NCCP', edgecolor='black', color=blue, hatch=next(hatches))
  plt.bar(3 + bar_width, mcp_nd_retrain * 1000, width=bar_width, label='NCCP with retraining', edgecolor='black', color=red, hatch=next(hatches))

  plt.bar(4, ndw * 1000, width=bar_width, label='NWCP', edgecolor='black', color=blue, hatch=next(hatches))
  plt.bar(4 + bar_width, ndw_retrain * 1000, width=bar_width, label='NWCP with retraining', edgecolor='black', color=red, hatch=next(hatches))

  plt.bar(5, ew * 1000, width=bar_width, label='EWCP', edgecolor='black', color=blue, hatch=next(hatches))
  plt.bar(5 + bar_width, ew_retrain * 1000, width=bar_width, label='EWCP with retraining', edgecolor='black', color=red, hatch=next(hatches))

  plt.xticks(X + bar_width/2, models)
  plt.ylim(ymin=ymin, ymax=ymax)
  plt.legend(ncol=3)
  plt.title(title)
  plt.xlabel("Model")
  plt.ylabel("milliseconds")
  plt.grid(True)

  # plt.savefig(f"plots/{title}.png", bbox_inches='tight')
  plt.tight_layout()
  output_dir = "plots"
  os.makedirs(output_dir, exist_ok=True)
  plt.savefig(f"{output_dir}/{title}.png")
  plt.close()
  # plt.show()

plot("Average prediction latency per sample across time steps - OGB Arxiv", ymin=0, ymax=6)