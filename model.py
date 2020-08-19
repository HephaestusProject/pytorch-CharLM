from collections import namedtuple
from typing import List

import torch
import torch.nn as nn


class CharLM(nn.Module):
    """CNN + highway network + LSTM
    # Input:
        4D tensor with shape [batch_size, in_channel, height, width]
    # Output:
        2D Tensor with shape [batch_size, vocab_size]
    # Arguments:
        char_emb_dim: the size of each character's embedding
        word_emb_dim: the size of each word's embedding
        vocab_size: num of unique words
        num_char: num of characters
        use_gpu: True or False
    """

    def __init__(
        self,
        num_chars: int,
        char_embedding_dim: int,
        char_padding_idx: int,
        hidden_dim: int,
        dropout: float,
    ):
        super(CharLM, self).__init__()

        self.char_embedding = nn.Embedding(
            num_chars, char_embedding_dim, padding_idx=char_padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=char_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=6,
            bias=True,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_dim, num_chars), nn.LogSoftmax(dim=2)
        )

        self._hidden = None

    def forward(self, x):
        batch_size, num_chars = x.size()

        x = self.char_embedding(x)

        x, self._hidden = self.lstm(x, self._hidden)
        return self.output_layer(x)

    def initialize_state(self):
        self._hidden = None

    def detach_state(self):
        if self._hidden is not None:
            self._hidden = [h.detach() for h in self._hidden]


class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.fc1.bias.data.fill_(-2)

    def forward(self, x):
        gate = self.sigmoid(self.fc1(x))
        return gate * self.relu(self.fc2(x)) + (1 - gate) * x
