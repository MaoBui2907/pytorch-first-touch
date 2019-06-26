import torch
import torch.nn as nn
# import warnings


class LinearRegression(nn.Module):
    """
    Class Linear regression kết thừa từ class nn.Module
    """

    def __init__(self, x_dim=1, y_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(x_dim, y_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


