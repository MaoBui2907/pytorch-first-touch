import torch
import torch.nn as nn

class NeuronNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim):
    super(NeuronNetwork, self).__init__()
    # ! Hidden layer 1
    self.layer1 = nn.Linear(input_dim, hidden_dim)

    # ! Hidden layer 2
    self.layer2 = nn.Linear(hidden_dim, hidden_dim)

    # ! Hidden layer3
    self.layer3 = nn.Linear(hidden_dim, hidden_dim)

    # ! Output layer
    self.out = nn.Linear(hidden_dim, output_dim)

    # ! Activate function
    self.activate = nn.ReLU()

  def forward(self, input):
    # ! Layer 1
    output = self.layer1(input)
    output = self.activate(output)

    # ! Layer 2
    output = self.layer2(output)
    output = self.activate(output)

    # ! Layer 3
    output = self.layer3(output)
    output = self.activate(output)

    # ! Output layer
    output = self.out(output)

    return output