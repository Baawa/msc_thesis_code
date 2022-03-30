import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGE(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sampling_size):
    """
      sampling_size: number of neighbors to sample at each layer (list[int]).
    """
    super(GraphSAGE, self).__init__()

    self.num_layers = num_layers

    self.convs = torch.nn.ModuleList()

    # input layer
    self.convs.append(SAGEConv(in_channels=input_dim, out_channels=hidden_dim, normalize=True))

    # hidden layers
    for _ in range(0, num_layers-1):
      self.convs.append(SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, normalize=True))

    # output layer
    self.convs.append(SAGEConv(in_channels=hidden_dim, out_channels=output_dim))

    self.softmax = torch.nn.Softmax(dim=1)

    if len(sampling_size) != num_layers:
      raise ValueError("Invalid sampling_size. Expected list to be as long as num_layers")

    self.sampling_size = sampling_size

  def reset_parameters(self):
    for conv in self.convs:
        conv.reset_parameters()

  def forward(self, x, adj_t):
    embedding = x
    for i in range(self.num_layers):
      embedding = self.convs[i](x=embedding, edge_index=adj_t)
      embedding = torch.nn.functional.relu(embedding)
    
    # output layer
    embedding = self.convs[-1](x=embedding, edge_index=adj_t)

    node_class = self.softmax(embedding)

    return node_class

  def train_model(self, data, optimizer, loss_fn, batch_size):
    self.train()

    loader = NeighborLoader(data, batch_size=batch_size, num_neighbors=self.sampling_size)

    for batch in loader:
        optimizer.zero_grad()
        
        out = self(batch.x, batch.edge_index)

        y_true = batch.y.reshape(-1)

        loss = loss_fn(out, y_true)

        loss.backward()
        optimizer.step()

    return loss.item()
  
  def predict(self, data):
    loader = NeighborLoader(data, num_neighbors=self.sampling_size)

    y_hat = []
    for batch in loader:
      out = self(batch.x, batch.edge_index)
      y_hat.extend(out.tolist())
    
    return torch.tensor(y_hat)
