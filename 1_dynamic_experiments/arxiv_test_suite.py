
import os
import sys
module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from lib.logger import Logger
from lib.conformal_predictor import InductiveConformalClassifier, get_nonconformity_measure_for_classification
from lib import evaluation
from lib.graphsage import GraphSAGE
from lib.data import split, split_dataset
from lib.util import plot
import matplotlib.pyplot as plt
from tabulate import tabulate


NUM_EXPERIMENTS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NODE_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
CONFIDENCE_LEVEL = 0.95

MODEL_ARGS = {
    "num_layers": 3,
    "hidden_dim": 256,
    "lr": 0.01,  # learning rate
    "epochs": 100,
}

def split_graph(data):
  graphs = []

  for year in NODE_YEARS:
    print("year: {}".format(year))
    indices = torch.nonzero(torch.where(data.node_year[:, 0] <= year, 1, 0))[
        :, 0].tolist()

    year_data = split(data, indices)

    train_data, calibration_indices, test_indices = split_dataset(
        year_data, test_frac=0.2, calibration_frac=0.2)
    graphs.append({
        "year": year,
        "data": year_data,
        "train_data": train_data,
        "calibration_indices": calibration_indices,
        "test_indices": test_indices,
    })

  return graphs


def train_model(graph, num_classes):
  train_data = graph["train_data"]
  data = graph["data"]
  model = GraphSAGE(data.x.shape[1], MODEL_ARGS["hidden_dim"],
                    num_classes, MODEL_ARGS["num_layers"]).to(DEVICE)

  # reset the parameters to initial random value
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_ARGS["lr"])

  loss_fn = torch.nn.NLLLoss()

  for epoch in range(1, 1 + MODEL_ARGS["epochs"]):
      loss = model.train_model(train_data, optimizer, loss_fn)

  return model

def get_confidence_intervals(cp, y_hat, confidence_level=0.95):
  confidence_intervals = []
  for yi in y_hat:
    alphas = get_nonconformity_measure_for_classification(yi)
    ci = cp.predict(alphas, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def run_train_once_no_resampling():
  logger = Logger("output")

  logger.log("STARTED: run_train_once_no_resampling")

  # download dataset using ogb pytorch geometric loader.
  dataset = PygNodePropPredDataset(name="ogbn-arxiv")

  data = dataset[0]  # pyg graph object

  logger.log("dataset loaded")

  plt.title("Class distribution (full graph)")
  plt.hist(data.y.reshape(-1).detach().numpy(), dataset.num_classes)
  plt.xlabel("Class")
  plt.ylabel("# nodes")
  plt.savefig("output/arxiv-class-dist-full.png")
  plt.close()

  graphs = split_graph(data)

  for graph in graphs:
    plt.title("Class distribution")
    plt.hist(graph["data"].y.reshape(-1).detach().numpy(),
              dataset.num_classes)
    plt.xlabel("Class")
    plt.ylabel("# nodes")
    plt.savefig("output/arxiv-class-dist-{}.png".format(graph["year"]))
    plt.close()

  logger.log('Device: {}'.format(DEVICE))

  for graph in graphs:
    graph["train_data"] = graph["train_data"].to(DEVICE)
    graph["data"] = graph["data"].to(DEVICE)


  accuracy_scores = []
  macro_f1_scores = []

  icp_coverages = []
  icp_avg_prediction_set_sizes = []
  icp_frac_singleton_preds = []
  icp_frac_empty_preds = []

  for experiment_num in range(NUM_EXPERIMENTS):
    logger.log("Experiment {} started".format(experiment_num))

    # train on first snapshot
    first_snapshot = graphs[0]
    model = train_model(first_snapshot)

    y_hat = model.predict(first_snapshot["data"])
    y_hat = y_hat[first_snapshot["test_indices"]]
    y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1)

    y_true = first_snapshot["data"].y[first_snapshot["test_indices"]
                                      ].reshape(-1)

    acc, macro_f1 = evaluation.get_multiclass_classification_performance(
        y_hat.detach().cpu(), y_true.detach().cpu())

    logger.log(f"Finished training GraphSAGE model with accuracy {100 * acc:.2f}% and macro avg f1 score: {100 * macro_f1:.2f}%")

    # capture model performance
    _accuracy_scores = []
    _macro_f1_scores = []

    for graph in graphs:
      graph_data = graph["data"]
      test_indices = graph["test_indices"]

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
    cal_data = first_snapshot["data"]
    calibration_indices = first_snapshot["calibration_indices"]

    y_hat = model.predict(cal_data)
    y_hat = y_hat[calibration_indices]

    y_true = cal_data.y[calibration_indices]
    y_true = y_true.reshape(-1).detach()

    alphas = []
    for y_probas, yt in zip(y_hat,y_true):
      y = yt.item()
      alphas = get_nonconformity_measure_for_classification(y_probas)
      alphas.append(alphas[y])

    alphas = torch.tensor(alphas)

    icp = InductiveConformalClassifier(alphas, dataset.num_classes)

    _icp_coverages = []
    _icp_avg_prediction_set_sizes = []
    _icp_frac_singleton_preds = []
    _icp_frac_empty_preds = []
    for graph in graphs:
      graph_data = graph["data"]
      test_indices = graph["test_indices"]

      y_hat = model.predict(graph_data)
      y_hat = y_hat[test_indices].detach().cpu()

      y_true = graph_data.y[test_indices].reshape(-1).detach().cpu()

      confidence_intervals = get_confidence_intervals(icp, y_hat, CONFIDENCE_LEVEL)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

      _icp_coverages.append(coverage)
      _icp_avg_prediction_set_sizes.append(avg_prediction_set_size)
      _icp_frac_singleton_preds.append(frac_singleton_pred)
      _icp_frac_empty_preds.append(frac_empty_pred)
    
    icp_coverages.append(_icp_coverages)
    icp_avg_prediction_set_sizes.append(_icp_avg_prediction_set_sizes)
    icp_frac_singleton_preds.append(_icp_frac_singleton_preds)
    icp_frac_empty_preds.append(_icp_frac_empty_preds)



  # plot model performance
  plot("arxiv-train_once_no_resampling-graphsage-perf", NODE_YEARS, np.mean(accuracy_scores, axis=1), "Timestep", "Accuracy")

  # print icp performance
  logger.log("ICP performance")
  coverages = ["coverage"].append(np.mean(icp_coverages, axis=1))
  avg_prediction_set_sizes = ["avg prediction set size"].append(np.mean(icp_avg_prediction_set_sizes, axis=1))
  frac_singleton_preds = ["frac singleton pred"].append(np.mean(icp_frac_singleton_preds, axis=1))
  frac_empty_preds = ["frac_empty_pred"].append(np.mean(icp_frac_empty_preds, axis=1))
  scores = [coverages, avg_prediction_set_sizes, frac_singleton_preds, frac_empty_preds]
  logger.log(tabulate(scores, headers=NODE_YEARS))





# run tests
run_train_once_no_resampling()