import torch

from losses import TokenNLLLoss


def test_loss():
    loss_function = TokenNLLLoss(reduction="mean", ignore_index=-99)
    loss = loss_function(torch.rand(size=(10, 5, 16)), torch.randint(0, 16, size=(10, 5)))
    assert loss is not None
