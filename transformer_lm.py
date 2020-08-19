from torch import nn
from transformer import TransformerDecoder, SinusoidalPositionalEncoding
from transformer import mask_from_lengths, mask_from_subsequent_positions


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, n_heads, dropout, pad_id):
        super(TransformerLM, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Embedding(vocab_size, d_model, padding_idx=pad_id),
            SinusoidalPositionalEncoding(encoding_size=d_model, initial_length=512),
            nn.Dropout(dropout),
        )
        self.transformer_decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=False,
        )
        self.output_layer = nn.Sequential(nn.Linear(d_model, vocab_size), nn.LogSoftmax(dim=2))

        self._hidden = None

    def forward_step(self, inputs, input_lengths=None, state=None):
        inputs_subsequent_mask = mask_from_subsequent_positions(size=inputs.size(1)).to(
            inputs.device
        )
        if input_lengths is not None:
            input_lengths_mask = mask_from_lengths(lengths=input_lengths, max_length=inputs.size(1))
        else:
            input_lengths_mask = None

        x = self.input_layer(inputs)
        x, (self._hidden, _) = self.transformer_decoder(
            x,
            inputs_mask=inputs_subsequent_mask,
            inputs_key_padding_mask=input_lengths_mask,
            state=self._hidden,
        )
        outputs = self.output_layer(x)

        return outputs, state

    def forward(self, inputs, input_lengths=None, decoding_sampling_rate=0, state=None):

        # TODO: autoregressive manner
        outputs, state = self.forward_step(inputs=inputs, input_lengths=input_lengths, state=state)
        return outputs, state

    def initialize_state(self):
        self._hidden = None

    def detach_state(self):
        if self._hidden is not None:
            self._hidden = [h.detach() for h in self._hidden]
