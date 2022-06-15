import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

class GraphSAGEWithSampling(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sampling_size, batch_size):
    """
      sampling_size: number of neighbors to sample at each layer (list[int]).
    """
    super(GraphSAGEWithSampling, self).__init__()

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
    self.batch_size = batch_size

    self.return_embeds = False

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

    if self.return_embeds:
      return node_class, embedding

    return node_class

  def train_model(self, data, optimizer, loss_fn):
    self.train()

    loader = NeighborLoader(data, batch_size=self.batch_size, num_neighbors=self.sampling_size)

    for batch in loader:
        optimizer.zero_grad()
        
        out = self(batch.x, batch.edge_index)

        y_true = batch.y.reshape(-1)

        loss = loss_fn(out, y_true)

        loss.backward()
        optimizer.step()
  
  def predict(self, data):
    self.eval()
    
    loader = NeighborLoader(data, num_neighbors=self.sampling_size, batch_size=self.batch_size)

    y_hat = []
    for batch in loader:
      out = self(batch.x, batch.edge_index)
      y_hat.extend(out.tolist())
    
    return torch.tensor(y_hat)
  
  def set_return_embeds(self, return_embeds: bool):
    self.return_embeds = return_embeds

class GraphSAGE(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
    """
      sampling_size: number of neighbors to sample at each layer (list[int]).
    """
    super(GraphSAGE, self).__init__()

    self.num_layers = num_layers

    self.convs = torch.nn.ModuleList()
    self.bns = torch.nn.ModuleList()

    # input layer
    self.convs.append(SAGEConv(in_channels=input_dim, out_channels=hidden_dim, aggr="mean"))
    self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    # hidden layers
    for _ in range(0, num_layers-1):
      self.convs.append(SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr="mean"))
      self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    # output layer
    self.convs.append(SAGEConv(in_channels=hidden_dim, out_channels=output_dim, aggr="mean"))

    self.softmax = torch.nn.LogSoftmax(dim=1)

    self.dropout = 0.1

    self.return_embeds = False

  def reset_parameters(self):
    for conv in self.convs:
        conv.reset_parameters()

  def forward(self, x, adj_t):
    embedding = x
    for i, conv in enumerate(self.convs[:-1]):
      embedding = conv(x=embedding, edge_index=adj_t)
      embedding = self.bns[i](embedding)
      embedding = torch.nn.functional.relu(embedding)
      embedding = torch.nn.functional.dropout(embedding, p=self.dropout, training=self.training)
    
    # output layer
    class_embedding = self.convs[-1](x=embedding, edge_index=adj_t)

    node_class = self.softmax(class_embedding)

    if self.return_embeds:
      return node_class, embedding

    return node_class

  def train_model(self, data, optimizer, loss_fn):
    self.train()

    optimizer.zero_grad()
    
    out = self(data.x, data.edge_index)

    y_true = data.y.reshape(-1).detach()

    loss = loss_fn(out, y_true)

    loss.backward()
    optimizer.step()
  
  def predict(self, data):
    self.eval()
    
    if self.return_embeds:
        y_hat, embeddings = self(data.x, data.edge_index)
        return y_hat.clone().detach(), embeddings
        
    y_hat = self(data.x, data.edge_index)
    
    return y_hat.clone().detach()

  def set_return_embeds(self, return_embeds: bool):
    self.return_embeds = return_embeds
