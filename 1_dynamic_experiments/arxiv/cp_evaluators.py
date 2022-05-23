import numpy as np
from tabulate import tabulate
import os
from graph import Graph
from conformal_predictor import InductiveConformalClassifier, MondrianConformalClassifier, NodeDegreeMondrianConformalClassifier, NodeDegreeWeightedConformalClassifier, get_nonconformity_measure_for_classification, EmbeddingDistanceWeightedConformalClassifier
import time
from evaluation import get_coverage_and_efficiency
import torch

class CPEvaluator(object):
  def __init__(self, title, timesteps, output_dir, confidence_level):
    self.title = title
    self.timesteps = timesteps
    self.output_dir = output_dir
    self.confidence_level = confidence_level
    self.batch_coverages = []
    self.batch_avg_prediction_set_sizes = []
    self.batch_frac_singleton_preds = []
    self.batch_frac_empty_preds = []
    self.coverages = []
    self.avg_prediction_set_sizes = []
    self.frac_singleton_preds = []
    self.frac_empty_preds = []
    self.batch_prediction_times = []
    self.prediction_times = []
    self.batch_units = []
    self.units = []
  
  def new_batch(self):
    self.coverages.append(self.batch_coverages)
    self.avg_prediction_set_sizes.append(self.batch_avg_prediction_set_sizes)
    self.frac_singleton_preds.append(self.batch_frac_singleton_preds)
    self.frac_empty_preds.append(self.batch_frac_empty_preds)
    self.prediction_times.append(self.batch_prediction_times)
    self.units.append(self.batch_units)
    self.batch_coverages = []
    self.batch_avg_prediction_set_sizes = []
    self.batch_frac_singleton_preds = []
    self.batch_frac_empty_preds = []
    self.batch_prediction_times = []
    self.batch_units = []
  
  def save_results(self):
    # time_avg = np.mean(self.prediction_times)
    # time_std = np.std(self.prediction_times)
    prediction_time_avg = ["prediction_time avg"]
    prediction_time_avg.extend(np.mean(np.array(self.prediction_times), axis=0).tolist())
    self._save_results("results_{}_time.txt".format(self.title), tabulate([prediction_time_avg], headers=self.timesteps, tablefmt="tsv"))
    
    unit_avg = ["unit avg"]
    unit_avg.extend(np.mean(np.array(self.units), axis=0).tolist())
    self._save_results("results_{}_units.txt".format(self.title), tabulate([unit_avg], headers=self.timesteps, tablefmt="tsv"))

    coverage_avg = ["coverage avg"]
    coverage_avg.extend(np.mean(np.array(self.coverages), axis=0).tolist())
    coverage_std = ["coverage std"]
    coverage_std.extend(np.std(np.array(self.coverages), axis=0).tolist())
    
    avg_prediction_set_size_avg = ["avg prediction set size avg"]
    avg_prediction_set_size_avg.extend(np.mean(np.array(self.avg_prediction_set_sizes), axis=0).tolist())
    avg_prediction_set_size_std = ["avg prediction set size std"]
    avg_prediction_set_size_std.extend(np.std(np.array(self.avg_prediction_set_sizes), axis=0).tolist())
    
    frac_singleton_pred_avg = ["frac singleton pred avg"]
    frac_singleton_pred_avg.extend(np.mean(np.array(self.frac_singleton_preds), axis=0).tolist())
    frac_singleton_pred_std = ["frac singleton pred std"]
    frac_singleton_pred_std.extend(np.std(np.array(self.frac_singleton_preds), axis=0).tolist())
    
    frac_empty_pred_avg = ["frac empty pred avg"]
    frac_empty_pred_avg.extend(np.mean(np.array(self.frac_empty_preds), axis=0).tolist())
    frac_empty_pred_std = ["frac empty pred std"]
    frac_empty_pred_std.extend(np.std(np.array(self.frac_empty_preds), axis=0).tolist())
    
    scores = [coverage_avg, coverage_std, avg_prediction_set_size_avg, avg_prediction_set_size_std, frac_singleton_pred_avg, frac_singleton_pred_std, frac_empty_pred_avg, frac_empty_pred_std]
    self._save_results("results_{}_performance.txt".format(self.title), tabulate(scores, headers=self.timesteps))
    self._save_results("results_{}_performance.txt".format(self.title), tabulate(scores, headers=self.timesteps, tablefmt="tsv"))
  
  def _save_results(self, file, str):
    try:
      os.makedirs(self.output_dir, exist_ok=True)
    finally:
      f = open(self.output_dir + file, "w")
      f.write(str)
      f.close()

class ICPEvaluator(CPEvaluator):
  def capture(self, model, cp: InductiveConformalClassifier, graph: Graph):
    start_time = time.time()
    y_hat = model.predict(graph.data)
    y_hat = y_hat[graph.test_indices].detach().cpu()

    y_true = graph.data.y[graph.test_indices].reshape(-1).detach().cpu()

    confidence_intervals = get_confidence_intervals_icp(cp, y_hat, self.confidence_level)
    
    num_predictions = y_hat.shape[0]
    prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
    self.batch_prediction_times.append(prediction_time)
    self.units.append(num_predictions)

    coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

    self.batch_coverages.append(coverage)
    self.batch_avg_prediction_set_sizes.append(avg_prediction_set_size)
    self.batch_frac_singleton_preds.append(frac_singleton_pred)
    self.batch_frac_empty_preds.append(frac_empty_pred)

class ICPWithResamplingEvaluator(CPEvaluator):
  def capture(self, model, graph: Graph, num_classes):
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
    self.batch_prediction_times.append(prediction_time)
    self.units.append(num_predictions)

    coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true[test_indices])

    self.batch_coverages.append(coverage)
    self.batch_avg_prediction_set_sizes.append(avg_prediction_set_size)
    self.batch_frac_singleton_preds.append(frac_singleton_pred)
    self.batch_frac_empty_preds.append(frac_empty_pred)

class MCPEvaluator(CPEvaluator):
  def capture(self, model, mcp: MondrianConformalClassifier, graph: Graph):
    test_indices = graph.test_indices

    start_time = time.time()
    y_hat = model.predict(graph.data)
    y_hat = y_hat[test_indices].detach().cpu()

    y_true = graph.data.y[test_indices].reshape(-1).detach().cpu()

    confidence_intervals = get_confidence_intervals_mcp(mcp, y_hat, self.confidence_level)

    num_predictions = y_hat.shape[0]
    prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
    self.batch_prediction_times.append(prediction_time)
    self.units.append(num_predictions)

    coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

    self.batch_coverages.append(coverage)
    self.batch_avg_prediction_set_sizes.append(avg_prediction_set_size)
    self.batch_frac_singleton_preds.append(frac_singleton_pred)
    self.batch_frac_empty_preds.append(frac_empty_pred)

class NodeDegreeMCPEvaluator(CPEvaluator):
  def capture(self, model, cp: NodeDegreeMondrianConformalClassifier, graph: Graph, bins):
    test_indices = graph.test_indices

    start_time = time.time()
    y_hat = model.predict(graph.data)
    y_hat = y_hat[test_indices].detach().cpu()
    y_true = graph.data.y[test_indices].reshape(-1).detach().cpu()

    # get node degrees
    node_ids, some_node_degrees = torch.unique(graph.data.edge_index[1], return_counts=True)
    node_degrees = torch.zeros(graph.data.x.shape[0]).long().cpu()
    node_degrees[node_ids] = some_node_degrees.cpu()

    node_degrees = node_degrees[test_indices]  # only use calibration nodes

    node_degrees_bins = torch.bucketize(node_degrees, bins)

    degrees = torch.unique(node_degrees_bins)

    # capture model prediction time and add it to the final prediction time per bin
    num_predictions = y_hat.shape[0]
    model_prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)

    _coverages = []
    _avg_prediction_set_sizes = []
    _frac_singleton_preds = []
    _frac_empty_preds = []
    _samples_per_bin = []

    for d in degrees:
      start_time = time.time()

      degree_mask = node_degrees_bins == d

      confidence_intervals = get_confidence_intervals_nd_mcp(cp, y_hat[degree_mask], d, self.confidence_level)

      num_predictions = y_hat[degree_mask].shape[0]
      prediction_time = get_elapsed_time_per_unit(start_time, num_predictions) + model_prediction_time
      self.batch_prediction_times.append(prediction_time)
      self.units.append(num_predictions)

      coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true[degree_mask])

      _samples_per_bin.append(y_true[degree_mask].shape[0])
      _coverages.append(coverage)
      _avg_prediction_set_sizes.append(avg_prediction_set_size)
      _frac_singleton_preds.append(frac_singleton_pred)
      _frac_empty_preds.append(frac_empty_pred)

    dataset_size = torch.sum(torch.tensor(_samples_per_bin))
    
    # average 
    self.batch_coverages.append(get_measure_averaged_on_bins(_coverages, _samples_per_bin, dataset_size))
    self.batch_avg_prediction_set_sizes.append(get_measure_averaged_on_bins(_avg_prediction_set_sizes, _samples_per_bin, dataset_size))
    self.batch_frac_singleton_preds.append(get_measure_averaged_on_bins(_frac_singleton_preds, _samples_per_bin, dataset_size))
    self.batch_frac_empty_preds.append(get_measure_averaged_on_bins(_frac_empty_preds, _samples_per_bin, dataset_size))

class NodeDegreeWeightedCPEvaluator(CPEvaluator):
  def capture(self, model, cp: NodeDegreeWeightedConformalClassifier, graph: Graph):
    test_indices = graph.test_indices

    start_time = time.time()

    y_hat = model.predict(graph.data)
    y_hat = y_hat[test_indices].detach().cpu()

    y_true = graph.data.y[test_indices].reshape(-1).detach().cpu()

    # get node degrees
    node_ids, some_node_degrees = torch.unique(graph.data.edge_index[1], return_counts=True)
    node_degrees = torch.zeros(graph.data.x.shape[0]).long().cpu()
    node_degrees[node_ids] = some_node_degrees.cpu()

    confidence_intervals = get_confidence_intervals_node_degree_weighted(cp, y_hat, node_degrees.cpu(), self.confidence_level)

    num_predictions = y_hat.shape[0]
    prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
    self.batch_prediction_times.append(prediction_time)
    self.units.append(num_predictions)

    coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

    self.batch_coverages.append(coverage)
    self.batch_avg_prediction_set_sizes.append(avg_prediction_set_size)
    self.batch_frac_singleton_preds.append(frac_singleton_pred)
    self.batch_frac_empty_preds.append(frac_empty_pred)

class EmbeddingWeightedCPEvaluator(CPEvaluator):
  def capture(self, model, cp: EmbeddingDistanceWeightedConformalClassifier, graph: Graph):
    test_indices = graph.test_indices

    start_time = time.time()

    model.set_return_embeds(True)
    y_hat, embeddings = model.predict(graph.data)
    model.set_return_embeds(False)
    
    y_hat = y_hat[test_indices].detach().cpu()
    y_true = graph.data.y[test_indices].reshape(-1).detach().cpu()
    embeddings = embeddings[test_indices].cpu()

    confidence_intervals = get_confidence_intervals_embedding_weighted(cp, y_hat, embeddings, self.confidence_level)

    num_predictions = y_hat.shape[0]
    prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
    self.batch_prediction_times.append(prediction_time)
    self.units.append(num_predictions)

    coverage, avg_prediction_set_size, frac_singleton_pred, frac_empty_pred = get_coverage_and_efficiency(confidence_intervals, y_true)

    self.batch_coverages.append(coverage)
    self.batch_avg_prediction_set_sizes.append(avg_prediction_set_size)
    self.batch_frac_singleton_preds.append(frac_singleton_pred)
    self.batch_frac_empty_preds.append(frac_empty_pred)

# helpers

def get_elapsed_time_per_unit(start_time, units):
  return (time.time() - start_time) / units

def get_confidence_intervals_icp(cp, y_hat, confidence_level=0.95):
  confidence_intervals = []
  for y_probas in y_hat:
    alphas = get_nonconformity_measure_for_classification(y_probas)
    ci = cp.predict(alphas, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def get_confidence_intervals_mcp(cp, y_hat, confidence_level=0.95):
  confidence_intervals = []
  for y_probas in y_hat:
    alphas = get_nonconformity_measure_for_classification(y_probas)
    ci = cp.predict(alphas, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def get_confidence_intervals_nd_mcp(cp, y_hat, node_degrees, confidence_level=0.95):
    confidence_intervals = []
    for y_probas in y_hat:
        alphas = get_nonconformity_measure_for_classification(y_probas)
        ci = cp.predict(alphas, confidence_level, node_degrees)
        confidence_intervals.append(ci)

    return confidence_intervals

def get_confidence_intervals_node_degree_weighted(cp, y_hat, degrees, confidence_level=0.95):
  confidence_intervals = []
  for y_probas, d in zip(y_hat,degrees):
    alphas = get_nonconformity_measure_for_classification(y_probas)
    ci = cp.predict(alphas, d, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def get_confidence_intervals_embedding_weighted(cp: EmbeddingDistanceWeightedConformalClassifier, y_hat, embeddings, confidence_level=0.95):
  confidence_intervals = []
  for y_probas, emb in zip(y_hat,embeddings):
    alphas = get_nonconformity_measure_for_classification(y_probas, version="v2")
    ci = cp.predict(alphas, emb, confidence_level)
    confidence_intervals.append(ci)
  
  return confidence_intervals

def get_measure_averaged_on_bins(measure, samples_per_bin, dataset_size):
  return torch.sum(torch.tensor(measure) * torch.tensor(samples_per_bin) / dataset_size)

def create_icp(y_hat, y_true, num_classes):
  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    _alphas = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(_alphas[y])

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
    _alphas = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(_alphas[y])

  alphas = torch.tensor(alphas)
  y = y_true

  mcp = MondrianConformalClassifier(alphas.cpu(), y.cpu())

  return mcp

def create_node_degree_mcp(model, data, calibration_indices, bins):
  y_hat = model.predict(data)
  y_hat = y_hat[calibration_indices]

  y_true = data.y[calibration_indices]
  y_true = y_true.reshape(-1).detach()

  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    _alphas = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(_alphas[y])

  alphas = torch.tensor(alphas)
  y = y_true

  # get node degrees
  node_ids, node_degrees_per_id = torch.unique(data.edge_index[1], return_counts=True) # note: nodes with degree 0, won't be represented
  
  node_degrees = torch.zeros(data.x.shape[0]).long().cpu()
  node_degrees[node_ids] = node_degrees_per_id.cpu()

  node_degrees = node_degrees[calibration_indices] # only use calibration nodes

  node_degrees_bins = torch.bucketize(node_degrees, bins)

  cp = NodeDegreeMondrianConformalClassifier(alphas.cpu(), y.cpu(), node_degrees_bins.cpu())

  return cp

def create_node_degree_weighted_cp(model, data, calibration_indices):
  y_hat = model.predict(data)
  y_hat = y_hat[calibration_indices]

  y_true = data.y[calibration_indices]
  y_true = y_true.reshape(-1).detach()

  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    _alphas = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(_alphas[y])

  alphas = torch.tensor(alphas)
  y = y_true

  # get node degrees
  node_ids, node_degrees_per_id = torch.unique(data.edge_index[1], return_counts=True) # note: nodes with degree 0, won't be represented
  node_degrees = torch.zeros(data.x.shape[0]).long().cpu()
  node_degrees[node_ids] = node_degrees_per_id.cpu()

  node_degrees = node_degrees[calibration_indices] # only use calibration nodes

  cp = NodeDegreeWeightedConformalClassifier(alphas.cpu(), y.cpu(), node_degrees.cpu())
  return cp

def create_embedding_weighted_cp(model, data, calibration_indices):
  model.set_return_embeds(True)
  y_hat, embeddings = model.predict(data)
  model.set_return_embeds(False)
  
  y_hat = y_hat[calibration_indices]
  embeddings = embeddings[calibration_indices]
  
  y_true = data.y[calibration_indices]
  y_true = y_true.reshape(-1).detach()

  alphas = []
  for y_probas, yt in zip(y_hat,y_true):
    y = yt.item()
    _alphas = get_nonconformity_measure_for_classification(y_probas)
    alphas.append(_alphas[y])

  alphas = torch.tensor(alphas)
  y = y_true

  cp = EmbeddingDistanceWeightedConformalClassifier(alphas.cpu(), y.cpu(), embeddings.cpu())
  return cp