import torch
from torch import nn


class TokenNLLLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()

        self.nll_loss = nn.NLLLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, outputs, targets):
        batch_size, sequence_length = targets.size()
        outputs_flat = outputs.view(batch_size * sequence_length, -1)
        targets_flat = targets.view(-1)

        loss = self.nll_loss(outputs_flat, targets_flat)
        return loss
