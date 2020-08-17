import torch

from model import CharLM


def test_model_size():
    model = CharLM(
        num_chars=50,
        num_words=1000,
        char_embedding_dim=100,
        char_padding_idx=0,
        char_conv_kernel_sizes=[1, 2, 3, 4, 5, 6],
        char_conv_out_channels=[25, 50, 75, 100, 125, 150],
        use_batch_norm=True,
        num_highway_layers=1,
        hidden_dim=300,
        dropout=0.5,
    )
    example_input = torch.randint(high=50, size=(4, 7, 10))
    model.initialize_state()
    example_output = model(example_input)
    assert example_output.size() == (4, 7, 1000)
