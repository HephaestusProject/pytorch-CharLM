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
        num_words: int,
        char_embedding_dim: int,
        char_padding_idx: int,
        char_conv_kernel_sizes: List[int],
        char_conv_out_channels: List[int],
        use_batch_norm: bool,
        num_highway_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super(CharLM, self).__init__()

        self.char_embedding = nn.Embedding(
            num_chars, char_embedding_dim, padding_idx=char_padding_idx
        )

        assert len(char_conv_kernel_sizes) == len(char_conv_out_channels)
        self.char_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=char_embedding_dim,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
                for kernel_size, out_channels in zip(char_conv_kernel_sizes, char_conv_out_channels)
            ]
        )
        self.tanh = nn.Tanh()

        highway_input_dim = sum(char_conv_out_channels)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(highway_input_dim, affine=False)
        else:
            self.batch_norm = None

        self.highway_layers = nn.Sequential(
            *[Highway(input_size=highway_input_dim) for _ in range(num_highway_layers)]
        )

        self.lstm = nn.LSTM(
            input_size=highway_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bias=True,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_dim, num_words), nn.LogSoftmax(dim=2)
        )

        self._hidden = None

    def forward(self, x):
        batch_size, num_words, max_word_length = x.size()

        x = x.view(-1, max_word_length).contiguous()
        # [num_seq*seq_len, max_word_len+2]

        x = self.char_embedding(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]

        x = x.transpose(1, 2)
        # x = torch.transpose(torch.unsqueeze(x, 2), 1, 3).contiguous()
        # (num_seq*seq_len, char_embedding_dim, 1, max_word_len+2)

        chosen_list = list()
        for char_conv in self.char_convs:
            feature_map = self.tanh(char_conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen, _ = torch.max(feature_map, dim=2)
            # chosen, _ = torch.max(feature_map, 3)
            # (batch_size, out_channel, 1)
            # chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)

        # (batch_size, total_num_filers)
        x = torch.cat(chosen_list, dim=1)
        # [num_seq*seq_len, total_num_filters]

        if self.batch_norm:
            x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = x.view(batch_size, num_words, -1).contiguous()
        # [num_seq, seq_len, total_num_filters]

        x, self._hidden = self.lstm(x, self._hidden)
        # [num_seq, seq_len, hidden_size]

        x = self.output_layer(x)

        return x

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
