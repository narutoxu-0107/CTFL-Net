import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        target = target.type(torch.float).view(-1,1)
        try:
            loss = F.mse_loss(x, target)
        except:
            loss = F.mse_loss(x.logits, target)
        return loss