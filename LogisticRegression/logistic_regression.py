import torch
import torch.nn as nn
# import warnings


class LogisticRegression(nn.Module):
    """
    Class Logistic regression kết thừa từ class nn.Module
    """

    def __init__(self, x_dim, y_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(x_dim, y_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


