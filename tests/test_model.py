import torch

from model import CharLM


def test_model_size():
    model = CharLM(
        num_chars=50, num_words=1000, char_embedding_dim=100, word_embedding_dim=200
    )
    example_input = torch.randint(high=50, size=(4, 7, 10))
    example_output, _ = model(example_input, hidden=None)
    assert example_output.size() == (4, 7, 1000)
