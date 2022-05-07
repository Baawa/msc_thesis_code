import numpy as np
from tabulate import tabulate
import os
from graph import Graph
from lib.conformal_predictor import InductiveConformalClassifier, MondrianConformalClassifier, NodeDegreeMondrianConformalClassifier, NodeDegreeWeightedConformalClassifier, get_nonconformity_measure_for_classification
import time
from lib.evaluation import get_coverage_and_efficiency
import torch

class CPEvaluator(object):
  def __init__(self, title, timesteps, output_dir, confidence_level):
    self.title = title
    self.timesteps = timesteps
    self.output_dir = output_dir
    self.confidence_level = confidence_level
    self.coverages = []
    self.avg_prediction_set_sizes = []
    self.frac_singleton_preds = []
    self.frac_empty_preds = []
    self.prediction_times = []
  
  def save_results(self):
    time_avg = np.mean(self.prediction_times)
    time_std = np.std(self.prediction_times)
    self._save_results("results_{}_time.txt".format(self.title), tabulate([[time_avg], [time_std]], headers=["avg","std"]))

    coverage_avg = ["coverage avg"].append(np.mean(self.coverages, axis=0))
    coverage_std = ["coverage std"].append(np.std(self.coverages, axis=0))
    
    avg_prediction_set_size_avg = ["avg prediction set size avg"].append(np.mean(self.avg_prediction_set_sizes, axis=0))
    avg_prediction_set_size_std = ["avg prediction set size std"].append(np.std(self.avg_prediction_set_sizes, axis=0))
    
    frac_singleton_pred_avg = ["frac singleton pred avg"].append(np.mean(self.frac_singleton_preds, axis=0))
    frac_singleton_pred_std = ["frac singleton pred std"].append(np.std(self.frac_singleton_preds, axis=0))
    
    frac_empty_pred_avg = ["frac empty pred avg"].append(np.mean(self.frac_empty_preds, axis=0))
    frac_empty_pred_std = ["frac empty pred std"].append(np.std(self.frac_empty_preds, axis=0))
    
    scores = [coverage_avg, coverage_std, avg_prediction_set_size_avg, avg_prediction_set_size_std, frac_singleton_pred_avg, frac_singleton_pred_std, frac_empty_pred_avg, frac_empty_pred_std]
    self._save_results("results_{}_performance.txt".format(self.title), tabulate(scores, headers=self.time_steps))
  
  def _save_results(self, file, str):
    try:
      os.makedirs(self.output_dir, exist_ok=True)
    finally:
      f = open(self.output_dir + file, "w")
      f.write(str)
      f.close()

class ICPEvaluator(CPEvaluator):
  def capture(self, model, cp: InductiveConformalClassifier, graphs: "list[Graph]"):
    coverages = []
    avg_prediction_set_sizes = []
    frac_singleton_preds = []
    frac_empty_preds = []

    for graph in graphs:
      start_time = time.time()
      y_hat = model.predict(graph.data)
      y_hat = y_hat[graph.test_indices].detach().cpu()

      y_true = graph.data.y[graph.test_indices].reshape(-1).detach().cpu()

      confidence_intervals = get_confidence_intervals_icp(cp, y_hat, self.confidence_level)
      
      num_predictions = y_hat.shape[0]
      prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
      self.prediction_times.append(prediction_time)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

      coverages.append(coverage)
      avg_prediction_set_sizes.append(avg_prediction_set_size)
      frac_singleton_preds.append(frac_singleton_pred)
      frac_empty_preds.append(frac_empty_pred)
    
    self.coverages.append(coverages)
    self.avg_prediction_set_sizes.append(avg_prediction_set_sizes)
    self.frac_singleton_preds.append(frac_singleton_preds)
    self.frac_empty_preds.append(frac_empty_preds)

class ICPWithResamplingEvaluator(CPEvaluator):
  def capture(self, model, graphs: "list[Graph]", num_classes):
    coverages = []
    avg_prediction_set_sizes = []
    frac_singleton_preds = []
    frac_empty_preds = []

    for graph in graphs:
      test_indices = graph.test_indices
      calibration_indices = graph.calibration_indices

      start_time = time.time()
      
      y_hat = model.predict(graph.data)
      y_hat = y_hat.detach().cpu()
      y_true = graph.data.y.reshape(-1).detach().cpu()

      icp = create_icp(y_hat[calibration_indices], y_true[calibration_indices], num_classes)

      confidence_intervals = get_confidence_intervals_icp(icp, y_hat[test_indices], self.confidence_level)

      num_predictions = y_hat[test_indices].shape[0]
      prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
      self.prediction_times.append(prediction_time)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true[test_indices])

      coverages.append(coverage)
      avg_prediction_set_sizes.append(avg_prediction_set_size)
      frac_singleton_preds.append(frac_singleton_pred)
      frac_empty_preds.append(frac_empty_pred)
    
    self.coverages.append(coverages)
    self.avg_prediction_set_sizes.append(avg_prediction_set_sizes)
    self.frac_singleton_preds.append(frac_singleton_preds)
    self.frac_empty_preds.append(frac_empty_preds)

class MCPEvaluator(CPEvaluator):
  def capture(self, model, mcp: MondrianConformalClassifier, graphs: "list[Graph]"):
    coverages = []
    avg_prediction_set_sizes = []
    frac_singleton_preds = []
    frac_empty_preds = []

    for graph in graphs:
      test_indices = graph.test_indices

      start_time = time.time()
      y_hat = model.predict(graph.data)
      y_hat = y_hat[test_indices].detach().cpu()

      y_true = graph.data.y[test_indices].reshape(-1).detach().cpu()

      confidence_intervals = get_confidence_intervals_mcp(mcp, y_hat, self.confidence_level)

      num_predictions = y_hat.shape[0]
      prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
      self.prediction_times.append(prediction_time)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

      coverages.append(coverage)
      avg_prediction_set_sizes.append(avg_prediction_set_size)
      frac_singleton_preds.append(frac_singleton_pred)
      frac_empty_preds.append(frac_empty_pred)
    
    self.coverages.append(coverages)
    self.avg_prediction_set_sizes.append(avg_prediction_set_sizes)
    self.frac_singleton_preds.append(frac_singleton_preds)
    self.frac_empty_preds.append(frac_empty_preds)

# helpers

def get_elapsed_time_per_unit(start_time, units):
  return (time.time() - start_time) / units

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

def create_icp(y_hat, y_true, num_classes):
  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    a = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(a[y])

  alphas = torch.tensor(alphas)

  icp = InductiveConformalClassifier(alphas.cpu(), num_classes)

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

  mcp = MondrianConformalClassifier(alphas.cpu(), y.cpu())

  return mcp