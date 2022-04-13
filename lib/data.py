import torch
from torch_geometric.data import Data
import numpy as np

def split_dataset(data, test_frac = 0.2, calibration_frac = 0.1):
  """
  Splits dataset into training, calibration, and test set.
  Training set is disjoint from the calibration and test set. This is how they do it in the GraphSAGE experiments: https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/minibatch.py#L76
  returns Data, list[int], list[int]
  """

  # split into training and test
  test_cal_indices = np.random.choice(data.num_nodes, int(data.num_nodes * (test_frac+calibration_frac)), replace=False)
  
  test_size = int(test_frac * data.num_nodes)
  test_indices = test_cal_indices[:test_size]
  calibration_indices = test_cal_indices[test_size:]

  train_indices = np.arange(data.num_nodes, dtype=np.int64)
  train_indices = np.delete(train_indices, test_cal_indices)
  train = split(data, train_indices)

  return train, calibration_indices.tolist(), test_indices.tolist()

def split(data: Data, indices):
    x = data.x[indices]

    edge_index_0 = np.isin(data.edge_index[0].tolist(), indices)
    edge_index_1 = np.isin(data.edge_index[1].tolist(), indices)

    indices_to_keep = np.logical_and(edge_index_0, edge_index_1)

    edge_index = data.edge_index[:,indices_to_keep]
    
    indices_dict = dict(zip(indices, np.arange(len(indices))))
    edge_index[0] = edge_index[0].apply_(lambda x: indices_dict[x])
    edge_index[1] = edge_index[1].apply_(lambda x: indices_dict[x])
    
    y = data.y[indices]

    return Data(x=x, edge_index=edge_index, y=y)