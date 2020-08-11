import torch


def perplexity_score(loss):
    return torch.exp(loss)
