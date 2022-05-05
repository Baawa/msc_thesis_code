
import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from lib.logger import Logger
from lib.conformal_predictor import InductiveConformalClassifier, get_nonconformity_measure_for_classification, MondrianConformalClassifier
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


NUM_EXPERIMENTS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NODE_YEARS = [2000, 2005, 2010, 2015, 2020]
CONFIDENCE_LEVEL = 0.95
OUTPUT_FOLDER = "output/arxiv/run_train_once_no_resampling/"

MODEL_ARGS = {
    "num_layers": 3,
    "hidden_dim": 256,
    "lr": 0.01,  # learning rate
    "epochs": 100,
}

def split_graph(data):
  graphs = []

  for year in NODE_YEARS:
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


def train_model(graph, num_features, num_classes):
  train_data = graph["train_data"]
  data = graph["data"]
  model = GraphSAGE(num_features, MODEL_ARGS["hidden_dim"],
                    num_classes, MODEL_ARGS["num_layers"]).to(DEVICE)

  # reset the parameters to initial random value
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_ARGS["lr"])

  loss_fn = torch.nn.NLLLoss()

  for epoch in range(1, 1 + MODEL_ARGS["epochs"]):
      loss = model.train_model(train_data, optimizer, loss_fn)

  return model

def get_confidence_intervals_icp(cp, y_hat, confidence_level=0.95):
  confidence_intervals = []
  for yi in y_hat:
    alphas = get_nonconformity_measure_for_classification(yi)
    ci = cp.predict(alphas, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def get_confidence_intervals_mcp(cp, y_hat, confidence_level=0.95):
  confidence_intervals = []
  for yi in y_hat:
    alphas = get_nonconformity_measure_for_classification(yi)
    max_y = yi.argmax(dim=-1, keepdim=True)
    ci = cp.predict(alphas, max_y, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def save_results(file, str):
  output_dir = OUTPUT_FOLDER
  try:
    os.mkdir(output_dir)
  finally:
    f = open(output_dir + file, "w")
    f.write(str)
    f.close()

def plot_class_distribution(ext, y, num_classes):
  plt.title("Class distribution {}".format(ext))
  plt.hist(y, num_classes)
  plt.xlabel("Class")
  plt.ylabel("num of nodes")
  plt.savefig(OUTPUT_FOLDER + "class-dist-{}.png".format(ext))
  plt.close()

def create_icp(model, data, calibration_indices, num_classes):
  y_hat = model.predict(data)
  y_hat = y_hat[calibration_indices]

  y_true = data.y[calibration_indices]
  y_true = y_true.reshape(-1).detach()

  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    a = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(a[y])

  alphas = torch.tensor(alphas)

  icp = InductiveConformalClassifier(alphas, num_classes)

  return icp

def create_mcp(model, data, calibration_indices):
  y_hat = model.predict(data)
  y_hat = y_hat[calibration_indices]

  y_true = data.y[calibration_indices]
  y_true = y_true.reshape(-1).detach()

  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    a = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(a[y])

  alphas = torch.tensor(alphas)
  y = y_true

  mcp = MondrianConformalClassifier(alphas, y)

  return mcp

def save_times(prefix, times):
  time_avg = np.mean(times)
  time_std = np.std(times)
  save_results("{}_time.txt".format(prefix), tabulate([time_avg, time_std], headers=["avg","std"]))

def save_cp_performance(prefix, coverages, avg_prediction_set_sizes, frac_singleton_preds, frac_empty_preds):
  coverage_avg = ["coverage avg"].append(np.mean(coverages, axis=1))
  coverage_std = ["coverage std"].append(np.std(coverages, axis=1))
  
  avg_prediction_set_size_avg = ["avg prediction set size avg"].append(np.mean(avg_prediction_set_sizes, axis=1))
  avg_prediction_set_size_std = ["avg prediction set size std"].append(np.std(avg_prediction_set_sizes, axis=1))
  
  frac_singleton_pred_avg = ["frac singleton pred avg"].append(np.mean(frac_singleton_preds, axis=1))
  frac_singleton_pred_std = ["frac singleton pred std"].append(np.std(frac_singleton_preds, axis=1))
  
  frac_empty_pred_avg = ["frac empty pred avg"].append(np.mean(frac_empty_preds, axis=1))
  frac_empty_pred_std = ["frac empty pred std"].append(np.std(frac_empty_preds, axis=1))
  
  scores = [coverage_avg, coverage_std, avg_prediction_set_size_avg, avg_prediction_set_size_std, frac_singleton_pred_avg, frac_singleton_pred_std, frac_empty_pred_avg, frac_empty_pred_std]
  save_results("{}_performance.txt".format(prefix), tabulate(scores, headers=NODE_YEARS))

def plot(title, x_label, y_label, x, y):
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.plot(x, y, "+-")
  plt.savefig(OUTPUT_FOLDER + title + ".png")
  plt.close()

def run_train_once_no_resampling():
  logger = Logger(OUTPUT_FOLDER)

  logger.log("STARTED: run_train_once_no_resampling")

  # download dataset using ogb pytorch geometric loader.
  dataset = PygNodePropPredDataset(name="ogbn-arxiv")

  data = dataset[0]  # pyg graph object

  logger.log("dataset loaded")

  plot_class_distribution("full graph", data.y.reshape(-1).detach().numpy(), dataset.num_classes)

  graphs = split_graph(data)

  for graph in graphs:
    plot_class_distribution(graph["year"], graph["data"].y.reshape(-1).detach().numpy(), dataset.num_classes)

  logger.log('Device: {}'.format(DEVICE))

  for graph in graphs:
    graph["train_data"] = graph["train_data"].to(DEVICE)
    graph["data"] = graph["data"].to(DEVICE)
  
  graphsage_training_times = []

  accuracy_scores = []
  macro_f1_scores = []

  icp_coverages = []
  icp_avg_prediction_set_sizes = []
  icp_frac_singleton_preds = []
  icp_frac_empty_preds = []
  icp_prediction_times = []
  
  icp_with_resampling_coverages = []
  icp_with_resampling_avg_prediction_set_sizes = []
  icp_with_resampling_frac_singleton_preds = []
  icp_with_resampling_frac_empty_preds = []
  icp_with_resampling_prediction_times = []

  mcp_coverages = []
  mcp_avg_prediction_set_sizes = []
  mcp_frac_singleton_preds = []
  mcp_frac_empty_preds = []
  mcp_prediction_times = []

  for experiment_num in range(NUM_EXPERIMENTS):
    logger.log("Experiment {} started".format(experiment_num))

    # train on first snapshot
    first_snapshot = graphs[0]
    start_time = time.time()
    model = train_model(first_snapshot, data.num_features, dataset.num_classes)
    graphsage_training_time = time.time() - start_time
    graphsage_training_times.append(graphsage_training_time)

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
    icp = create_icp(model, first_snapshot["data"], first_snapshot["calibration_indices"], dataset.num_classes)

    _icp_coverages = []
    _icp_avg_prediction_set_sizes = []
    _icp_frac_singleton_preds = []
    _icp_frac_empty_preds = []

    start_time = time.time()
    for graph in graphs:
      graph_data = graph["data"]
      test_indices = graph["test_indices"]

      y_hat = model.predict(graph_data)
      y_hat = y_hat[test_indices].detach().cpu()

      y_true = graph_data.y[test_indices].reshape(-1).detach().cpu()

      confidence_intervals = get_confidence_intervals_icp(icp, y_hat, CONFIDENCE_LEVEL)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = evaluation.get_coverage_and_efficiency(confidence_intervals, y_true)

      _icp_coverages.append(coverage)
      _icp_avg_prediction_set_sizes.append(avg_prediction_set_size)
      _icp_frac_singleton_preds.append(frac_singleton_pred)
      _icp_frac_empty_preds.append(frac_empty_pred)
    
    icp_prediction_time = time.time() - start_time
    icp_prediction_times.append(icp_prediction_time)
    
    icp_coverages.append(_icp_coverages)
    icp_avg_prediction_set_sizes.append(_icp_avg_prediction_set_sizes)
    icp_frac_singleton_preds.append(_icp_frac_singleton_preds)
    icp_frac_empty_preds.append(_icp_frac_empty_preds)
    
    # ICP with resampling
    _icp_with_resampling_coverages = []
    _icp_with_resampling_avg_prediction_set_sizes = []
    _icp_with_resampling_frac_singleton_preds = []
    _icp_with_resampling_frac_empty_preds = []

    start_time = time.time()
    for graph in graphs:
      graph_data = graph["data"]
      test_indices = graph["test_indices"]

      icp = create_icp(model, graph_data, graph["calibration_indices"], dataset.num_classes)

      y_hat = model.predict(graph_data)
      y_hat = y_hat[test_indices].detach().cpu()

      y_true = graph_data.y[test_indices].reshape(-1).detach().cpu()

      confidence_intervals = get_confidence_intervals_icp(icp, y_hat, CONFIDENCE_LEVEL)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = evaluation.get_coverage_and_efficiency(confidence_intervals, y_true)

      _icp_with_resampling_coverages.append(coverage)
      _icp_with_resampling_avg_prediction_set_sizes.append(avg_prediction_set_size)
      _icp_with_resampling_frac_singleton_preds.append(frac_singleton_pred)
      _icp_with_resampling_frac_empty_preds.append(frac_empty_pred)
    
    icp_with_resampling_prediction_time = time.time() - start_time
    icp_with_resampling_prediction_times.append(icp_with_resampling_prediction_time)
    
    icp_with_resampling_coverages.append(_icp_with_resampling_coverages)
    icp_with_resampling_avg_prediction_set_sizes.append(_icp_with_resampling_avg_prediction_set_sizes)
    icp_with_resampling_frac_singleton_preds.append(_icp_with_resampling_frac_singleton_preds)
    icp_with_resampling_frac_empty_preds.append(_icp_with_resampling_frac_empty_preds)

    # MCP
    mcp = create_mcp(model, first_snapshot["data"], first_snapshot["calibration_indices"])

    _mcp_coverages = []
    _mcp_avg_prediction_set_sizes = []
    _mcp_frac_singleton_preds = []
    _mcp_frac_empty_preds = []

    start_time = time.time()
    for graph in graphs:
      graph_data = graph["data"]
      test_indices = graph["test_indices"]

      y_hat = model.predict(graph_data)
      y_hat = y_hat[test_indices].detach().cpu()

      y_true = graph_data.y[test_indices].reshape(-1).detach().cpu()

      confidence_intervals = get_confidence_intervals_mcp(mcp, y_hat, CONFIDENCE_LEVEL)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = evaluation.get_coverage_and_efficiency(confidence_intervals, y_true)

      _mcp_coverages.append(coverage)
      _mcp_avg_prediction_set_sizes.append(avg_prediction_set_size)
      _mcp_frac_singleton_preds.append(frac_singleton_pred)
      _mcp_frac_empty_preds.append(frac_empty_pred)
    
    mcp_prediction_time = time.time() - start_time
    mcp_prediction_times.append(mcp_prediction_time)
    
    mcp_coverages.append(_mcp_coverages)
    mcp_avg_prediction_set_sizes.append(_mcp_avg_prediction_set_sizes)
    mcp_frac_singleton_preds.append(_mcp_frac_singleton_preds)
    mcp_frac_empty_preds.append(_mcp_frac_empty_preds)


  # save graphsage training time
  save_times("graphsage_training", graphsage_training_times)

  # plot model performance
  logger.log("graphsage_performance: {}".format(np.mean(accuracy_scores, axis=1)))
  plot("graphsage_performance", "Year", "Accuracy", NODE_YEARS, np.mean(accuracy_scores, axis=1))

  # print icp performance
  save_cp_performance("icp", icp_coverages, icp_avg_prediction_set_sizes, icp_frac_singleton_preds, icp_frac_empty_preds)
  save_times("icp_prediction", icp_prediction_times)
  
  # print icp with resampling performance
  save_cp_performance("icp_with_resampling", icp_with_resampling_coverages, icp_with_resampling_avg_prediction_set_sizes, icp_with_resampling_frac_singleton_preds, icp_with_resampling_frac_empty_preds)
  save_times("icp_with_resampling_prediction", icp_with_resampling_prediction_times)

  # print mcp performance
  save_cp_performance("mcp", mcp_coverages, mcp_avg_prediction_set_sizes, mcp_frac_singleton_preds, mcp_frac_empty_preds)
  save_times("mcp_prediction", mcp_prediction_times)





# run tests
run_train_once_no_resampling()