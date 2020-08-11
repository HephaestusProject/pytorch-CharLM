import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, relu, tanh


class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = sigmoid(self.fc1(x))
        return t * relu(self.fc2(x)) + (1 - t) * x


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

    def __init__(self, char_emb_dim, word_emb_dim, vocab_size, num_char):
        super(CharLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size

        self.char_embedding_layer = nn.Embedding(num_char, char_emb_dim)

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        for out_channel, filter_width in filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channel,
                    kernel_size=(char_emb_dim, filter_width),  # (height, width)
                    bias=True,
                )
            )

        self.highway_input_dim = sum([x for x, y in filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        self.lstm = nn.LSTM(
            input_size=self.highway_input_dim,
            hidden_size=self.word_emb_dim,
            num_layers=2,
            bias=True,
            dropout=0.5,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

    def forward(self, x, hidden):
        # Input: [num_seq, seq_len, max_word_len+2]
        # Return: [num_words, len(word_dict)]
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len+2]

        x = self.char_embedding_layer(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]

        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        # [num_seq*seq_len, 1, max_word_len+2, char_emb_dim]

        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway1(x)
        x = self.highway2(x)
        # [num_seq*seq_len, total_num_filters]

        x = x.contiguous().view(lstm_batch_size, lstm_seq_len, -1)
        # [num_seq, seq_len, total_num_filters]

        x, hidden = self.lstm(x, hidden)
        # [seq_len, num_seq, hidden_size]

        x = self.dropout(x)
        # [seq_len, num_seq, hidden_size]

        x = x.contiguous().view(lstm_batch_size * lstm_seq_len, -1)
        # [num_seq*seq_len, hidden_size]

        x = self.linear(x)
        # [num_seq*seq_len, vocab_size]

        return x, hidden

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)

        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)


if __name__ == "__main__":
    model = CharLM(char_emb_dim=10, word_emb_dim=20, vocab_size=30, num_char=30)
    xt = torch.randint(30, (4, 3, 7))
    print(xt)
    xt = model(x=xt, hidden=None)
    print(xt)
