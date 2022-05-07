import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from lib.logger import Logger
from lib import evaluation
from lib.graphsage import GraphSAGE
from lib.data import split, split_dataset
from lib.util import plot

# special setting for plotting on ubuntu
os.system('Xvfb :1 -screen 0 1600x1200x16  &')    # create virtual display with size 1600x1200 and 16 bit color. Color can be changed to 24 or 8
os.environ['DISPLAY']=':1.0'    # tell X clients to use our virtual DISPLAY :1.0.

import matplotlib.pyplot as plt
from tabulate import tabulate
import time
from cp_evaluators import ICPEvaluator, ICPWithResamplingEvaluator, create_icp, create_mcp, MCPEvaluator, NodeDegreeMCPEvaluator, create_node_degree_mcp, NodeDegreeWeightedCPEvaluator, create_node_degree_weighted_cp, EmbeddingWeightedCPEvaluator, create_embedding_weighted_cp
from graph import Graph


NUM_EXPERIMENTS = 2
CONFIDENCE_LEVEL = 0.95
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# arxiv specific
NODE_YEARS = [2010, 2015, 2020]
OUTPUT_FOLDER = "output/arxiv/run_train_once_no_resampling/"
BINS = torch.tensor([0,1,5,10,20,100]) # boundaries[i-1] < input[x] <= boundaries[i]
MODEL_ARGS = {
    "num_layers": 3,
    "hidden_dim": 256,
    "lr": 0.01,  # learning rate
    "epochs": 100,
}

def split_graph(data):
  graphs = [Graph]

  for year in NODE_YEARS:
    indices = torch.nonzero(torch.where(data.node_year[:, 0] <= year, 1, 0))[
        :, 0].tolist()

    year_data = split(data, indices)

    train_data, calibration_indices, test_indices = split_dataset(
        year_data, test_frac=0.2, calibration_frac=0.2)
    graphs.append(Graph(year, year_data, train_data, calibration_indices, test_indices))

  return graphs


def train_model(graph: Graph, num_features, num_classes):
  model = GraphSAGE(num_features, MODEL_ARGS["hidden_dim"],
                    num_classes, MODEL_ARGS["num_layers"]).to(DEVICE)

  # reset the parameters to initial random value
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_ARGS["lr"])

  loss_fn = torch.nn.NLLLoss()

  for epoch in range(1, 1 + MODEL_ARGS["epochs"]):
      loss = model.train_model(graph.train_data, optimizer, loss_fn)

  return model


def save_results(output_dir, file, str):
  try:
    os.makedirs(output_dir, exist_ok=True)
  finally:
    f = open(output_dir + file, "w")
    f.write(str)
    f.close()

def plot_class_distribution(ext, y, num_classes, output_dir):
  plt.title("Class distribution {}".format(ext))
  plt.hist(y, num_classes)
  plt.xlabel("Class")
  plt.ylabel("num of nodes")
  plt.savefig(output_dir + "class-dist-{}.png".format(ext))
  plt.close()

def save_times(prefix, times, output_dir):
  time_avg = np.mean(times)
  time_std = np.std(times)
  save_results(output_dir, "{}_time.txt".format(prefix), tabulate([[time_avg], [time_std]], headers=["avg","std"]))

def plot(title, x_label, y_label, x, y, output_dir):
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.plot(x, y, "+-")
  plt.savefig(output_dir + title + ".png")
  plt.close()

def run_train_once_no_resampling():
  logger = Logger(OUTPUT_FOLDER)

  logger.log("STARTED: run_train_once_no_resampling")

  # download dataset using ogb pytorch geometric loader.
  dataset = PygNodePropPredDataset(name="ogbn-arxiv")

  data = dataset[0]  # pyg graph object

  logger.log("dataset loaded")

  plot_class_distribution("full graph", data.y.reshape(-1).detach().numpy(), dataset.num_classes)


  logger.log('Device: {}'.format(DEVICE))
  
  graphsage_training_times = []

  accuracy_scores = []
  macro_f1_scores = []

  icp_evaluator = ICPEvaluator("arxiv_icp",NODE_YEARS,OUTPUT_FOLDER,CONFIDENCE_LEVEL)
  icp_with_resampling_evaluator = ICPWithResamplingEvaluator("arxiv_icp_with_resampling",NODE_YEARS,OUTPUT_FOLDER,CONFIDENCE_LEVEL)
  mcp_evaluator = MCPEvaluator("arxiv_mcp",NODE_YEARS,OUTPUT_FOLDER,CONFIDENCE_LEVEL)
  nd_mcp_evaluator = NodeDegreeMCPEvaluator("arxiv_node_degree_mcp",NODE_YEARS,OUTPUT_FOLDER,CONFIDENCE_LEVEL)
  nd_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("arxiv_node_degree_weighted_cp",NODE_YEARS,OUTPUT_FOLDER,CONFIDENCE_LEVEL)
  embedding_weighted_cp_evaluator = NodeDegreeWeightedCPEvaluator("arxiv_embedding_weighted_cp",NODE_YEARS,OUTPUT_FOLDER,CONFIDENCE_LEVEL)

  for experiment_num in range(NUM_EXPERIMENTS):
    logger.log("Experiment {} started".format(experiment_num))

    # split graph
    graphs = split_graph(data)

    for graph in graphs:
      plot_class_distribution(graph.timestep, graph.data.y.reshape(-1).detach().numpy(), dataset.num_classes)
    for graph in graphs:
      graph.train_data = graph.train_data.to(DEVICE)
      graph.data = graph.data.to(DEVICE)

    # train on first snapshot
    first_snapshot = graphs[0]
    start_time = time.time()
    model = train_model(first_snapshot, data.num_features, dataset.num_classes)
    graphsage_training_time = time.time() - start_time
    graphsage_training_times.append(graphsage_training_time)

    y_hat = model.predict(first_snapshot.data)
    y_hat = y_hat[first_snapshot.test_indices]
    y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1)

    y_true = first_snapshot.data.y[first_snapshot.test_indices
                                      ].reshape(-1)

    acc, macro_f1 = evaluation.get_multiclass_classification_performance(
        y_hat.detach().cpu(), y_true.detach().cpu())

    logger.log(f"Finished training GraphSAGE model with accuracy {100 * acc:.2f}% and macro avg f1 score: {100 * macro_f1:.2f}%")

    # capture model performance
    _accuracy_scores = []
    _macro_f1_scores = []

    for graph in graphs:
      graph_data = graph.data
      test_indices = graph.test_indices

      y_hat = model.predict(graph_data)
      y_hat = y_hat[test_indices]
      y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1).detach().cpu()

      y_true = graph_data.y[test_indices].reshape(-1).detach().cpu()

      acc, macro_f1 = evaluation.get_multiclass_classification_performance(y_hat, y_true)
      _accuracy_scores.append(acc)
      _macro_f1_scores.append(macro_f1)
        
    accuracy_scores.append(_accuracy_scores)
    macro_f1_scores.append(_macro_f1_scores)

    # ICP
    logger.log("running ICP")

    y_hat = model.predict(first_snapshot.data)
    y_hat = y_hat[first_snapshot.calibration_indices]

    y_true = data.y[first_snapshot.calibration_indices]
    y_true = y_true.reshape(-1).detach()

    icp = create_icp(y_hat, y_true, dataset.num_classes)

    icp_evaluator.capture(model, icp, graphs)
    
    # ICP with resampling
    logger.log("running ICP with resampling")
    
    icp_with_resampling_evaluator.capture(model, graphs, dataset.num_classes, CONFIDENCE_LEVEL)

    # MCP
    logger.log("running MCP")
    
    mcp = create_mcp(model, first_snapshot.data, first_snapshot.calibration_indices)

    mcp_evaluator.capture(model, mcp, graphs)

    # Node degree MCP
    logger.log("running node degree MCP")

    nd_mcp = create_node_degree_mcp(model, first_snapshot.data, first_snapshot.calibration_indices, BINS)

    nd_mcp_evaluator.capture(model, nd_mcp, graphs, BINS)

    # Node degree weighted CP
    logger.log("running node degree weighted CP")

    nd_weighted_cp = create_node_degree_weighted_cp(model, first_snapshot.data, first_snapshot.calibration_indices)

    nd_weighted_cp_evaluator.capture(model, nd_weighted_cp, graphs)
    
    # Embedding weighted CP
    logger.log("running embedding weighted CP")

    embedding_weighted_cp = create_embedding_weighted_cp(model, first_snapshot.data, first_snapshot.calibration_indices)

    embedding_weighted_cp_evaluator.capture(model, embedding_weighted_cp, graphs)

  # save graphsage training time
  save_times("graphsage_training", graphsage_training_times, OUTPUT_FOLDER)

  # plot model performance
  plot("graphsage_performance", "Timestep", "Accuracy", NODE_YEARS, np.mean(accuracy_scores, axis=0), OUTPUT_FOLDER)

  # print cp performance
  icp_evaluator.save_results()
  icp_with_resampling_evaluator.save_results()
  mcp_evaluator.save_results()
  nd_mcp_evaluator.save_results()
  nd_weighted_cp_evaluator.save_results()
  embedding_weighted_cp_evaluator.save_results()





# run tests
run_train_once_no_resampling()