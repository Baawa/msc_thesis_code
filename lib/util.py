import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx

def print_percentage(label, frac):
  print(label + ": {:2.2f}%".format(frac * 100))

def plot(title, x, y, x_label="", y_label="", save_file=False):
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.plot(x, y, "+-")
  if save_file:
    plt.savefig("plots/" + title + ".png")
  plt.show()

def plot_multilines(title, x, y, labels, x_label="", y_label="", save_file=False):
  """
  x: x-values (n_values)
  y: y-values (n_categories, n_values)
  """
  line_colors = ["b", "r", "g", "c", "m", "y"]
  
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  
  for i, yi in enumerate(y):
    plt.plot(x, yi, "{}+-".format(line_colors[i]), label=labels[i])
  
  plt.legend()
  
  if save_file:
    plt.savefig("plots/" + title + ".png")
  
  plt.show()

def describe_graph(data, log_scale = False, calc_diameter=False, save_fig=False):
  graph = to_networkx(data)

  is_strongly_connected = nx.is_strongly_connected(graph)
  print("graph is strongly connected component: {}".format(is_strongly_connected))
  print("graph is weakly connected component: {}".format(nx.is_weakly_connected(graph)))
  print("graph has # weakly connected components: {}".format(nx.number_weakly_connected_components(graph)))
  print("graph has # isolated nodes: {}".format(nx.algorithms.number_of_isolates(graph)))
  if is_strongly_connected and calc_diameter:
    print("graph has diameter: {}".format(nx.algorithms.distance_measures.diameter(graph)))

  degree_histogram = nx.degree_histogram(graph)
  plt.plot(range(0,len(degree_histogram)), degree_histogram)
  plt.xlabel("Node degree")
  plt.ylabel("Num of nodes")
  if log_scale:
    plt.xscale("log")
    plt.yscale("log")
  plt.title("Degree distribution")
  if save_fig:
    plt.savefig("plots/" + "degree_distribution.png")
  plt.show()