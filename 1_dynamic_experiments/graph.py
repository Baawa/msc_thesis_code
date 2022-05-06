class Graph(object):
  def __init__(self, timestep, data, train_data, calibration_indices, test_indices):
    self.timestep = timestep
    self.data = data
    self.train_data = train_data
    self.calibration_indices = calibration_indices
    self.test_indices = test_indices