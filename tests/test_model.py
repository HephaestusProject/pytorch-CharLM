import torch

from model import CharLM, ConvSize


def test_model_size():
    model = CharLM(
        num_chars=50,
        num_words=1000,
        char_embedding_dim=100,
        char_padding_idx=0,
        char_conv_sizes=(
            ConvSize(kernel_size=1, out_channels=25),
            ConvSize(kernel_size=2, out_channels=50),
            ConvSize(kernel_size=3, out_channels=75),
            ConvSize(kernel_size=4, out_channels=100),
            ConvSize(kernel_size=5, out_channels=125),
            ConvSize(kernel_size=6, out_channels=150),
        ),
        use_batch_norm=True,
        num_highway_layers=1,
        hidden_dim=300,
        dropout=0.5,
    )
    example_input = torch.randint(high=50, size=(4, 7, 10))
    model.initialize_state()
    example_output = model(example_input)
    assert example_output.size() == (4, 7, 1000)
