import numpy as np
from evaluation import get_multiclass_classification_performance
from graph import Graph
from tabulate import tabulate
import time
import os

class ModelEvaluator(object):
  def __init__(self, title, timesteps, output_dir):
    self.title = title
    self.timesteps = timesteps
    self.output_dir = output_dir
    self.batch_accuracy_scores = []
    self.batch_macro_f1_scores = []
    self.accuracy_scores = []
    self.macro_f1_scores = []
    self.prediction_times = []
  
  def new_batch(self):
    self.accuracy_scores.append(self.batch_accuracy_scores)
    self.macro_f1_scores.append(self.batch_macro_f1_scores)
    self.batch_accuracy_scores = []
    self.batch_macro_f1_scores = []
  
  def capture(self, model, graph: Graph):
    start_time = time.time()

    print(f"[modelevaluator] data is cuda: {graph.data.is_cuda}")
    y_hat = model.predict(graph.data.cuda())
    
    num_predictions = y_hat.shape[0]
    prediction_time = get_elapsed_time_per_unit(start_time, num_predictions)
    self.prediction_times.append(prediction_time)
    
    y_hat = y_hat[graph.test_indices]
    y_hat = y_hat.argmax(dim=-1, keepdim=True).reshape(-1).detach().cpu()

    y_true = graph.data.y[graph.test_indices].reshape(-1).detach().cpu()

    acc, macro_f1 = get_multiclass_classification_performance(y_hat, y_true)
    self.batch_accuracy_scores.append(acc)
    self.batch_macro_f1_scores.append(macro_f1)

  def save_results(self):
    time_avg = np.mean(self.prediction_times)
    time_std = np.std(self.prediction_times)
    self._save_results("results_{}_time.txt".format(self.title), tabulate([[time_avg, time_std]], headers=["avg","std"], tablefmt="tsv"))

    accuracy_avg = ["accuracy avg"]
    accuracy_avg.extend(np.mean(np.array(self.accuracy_scores), axis=0).tolist())
    accuracy_std = ["accuracy std"]
    accuracy_std.extend(np.std(np.array(self.accuracy_scores), axis=0).tolist())
    
    macro_f1_avg = ["macro_f1 avg"]
    macro_f1_avg.extend(np.mean(np.array(self.macro_f1_scores), axis=0).tolist())
    macro_f1_std = ["macro_f1 std"]
    macro_f1_std.extend(np.std(np.array(self.macro_f1_scores), axis=0).tolist())
    
    scores = [accuracy_avg, accuracy_std, macro_f1_avg, macro_f1_std]
    self._save_results("results_{}_performance.txt".format(self.title), tabulate(scores, headers=self.timesteps, tablefmt="tsv"))
  
  def _save_results(self, file, str):
    try:
      os.makedirs(self.output_dir, exist_ok=True)
    finally:
      f = open(self.output_dir + file, "w")
      f.write(str)
      f.close()


def get_elapsed_time_per_unit(start_time, units):
  return (time.time() - start_time) / units