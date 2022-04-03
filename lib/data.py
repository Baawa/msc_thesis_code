import torch
from torch_geometric.data import Data
import numpy as np

def split_dataset(data, test_frac = 0.2, calibration_frac = 0.1):
  """
  Splits dataset into training, calibration, and test set.
  Training set is disjoint from the calibration and test set. This is how they do it in the GraphSAGE experiments: https://github.com/williamleif/GraphSAGE/blob/a0fdef95dca7b456dab01cb35034717c8b6dd017/graphsage/minibatch.py#L76
  returns Data, list[int], list[int]
  """
  def split(data, indices):
    x = data.x[indices]
    edge_index = None
    
    indices_dict = dict(zip(indices, np.arange(len(indices))))
    
    for edge in torch.t(data.edge_index):
      if edge[0].item() not in indices or edge[1].item() not in indices:
        continue

      new_edge = torch.LongTensor([[indices_dict[edge[0].item()]],[indices_dict[edge[1].item()]]])

      if edge_index is None:
        edge_index = new_edge
      else:
        edge_index = torch.cat((edge_index, new_edge), 1)
    # END: for
    
    y = data.y[indices]

    return Data(x=x, edge_index=edge_index, y=y)


  # split into training and test
  test_cal_indices = np.random.choice(data.num_nodes, int(data.num_nodes * (test_frac+calibration_frac)), replace=False)
  
  test_size = int(test_frac * data.num_nodes)
  test_indices = test_cal_indices[:test_size]
  calibration_indices = test_cal_indices[test_size:]

  train_indices = np.arange(data.num_nodes, dtype=np.int64)
  train_indices = np.delete(train_indices, test_cal_indices)
  train = split(data, train_indices)

  return train, calibration_indices.tolist(), test_indices.tolist()
