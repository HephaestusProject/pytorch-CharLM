"""
Usage:
    main.py test [options]
    main.py test (-h | --help)

Options:
    --test-path <train-path>  Path to train data  [type: path]
    --word-vocabulary-path <word-vocabulary-path>  Load word tokenizer from [type: path]
    --char-vocabulary-path <char-vocabulary-path>  Load char tokenizer from [type: path]
    --checkpoint-path <checkpoint-path>  [type: path]
    
    --max-word-length <max-word-length>  Maximum number of chars in a word [type: int]
    --sequence-length <sequence-length>  Number of timesteps to unroll for [type: int]
    
    -h --help  Show this.
"""
from model import CharLM, ConvSize
from dataset import CHAR_SPECIAL_TOKENS, WORD_SPECIAL_TOKENS, CharCorpusDataset
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer
import torch

import math
from losses import TokenNLLLoss
from metrics import perplexity_score
from tqdm import tqdm


def test(args: dict):

    char_tokenizer = CharTokenizer.load(
        vocabulary_path=args["--char-vocabulary-path"], special_tokens=CHAR_SPECIAL_TOKENS,
    )
    word_tokenizer = WordTokenizer.load(
        vocabulary_path=args["--word-vocabulary-path"], special_tokens=WORD_SPECIAL_TOKENS,
    )

    checkpoint = torch.load(
        args["--checkpoint-path"], map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    hparams = checkpoint["hyper_parameters"]

    model = CharLM(
        num_chars=len(char_tokenizer),
        num_words=len(word_tokenizer),
        char_embedding_dim=hparams["--char-embedding-dim"],
        char_padding_idx=char_tokenizer.special_token_ids["pad_token"],
        char_conv_kernel_sizes=hparams["--char-conv-kernel-sizes"],
        char_conv_out_channels=hparams["--char-conv-out-channels"],
        use_batch_norm=hparams["--use-batch-norm"],
        num_highway_layers=hparams["--num-highway-layers"],
        hidden_dim=hparams["--hidden-dim"],
        dropout=hparams["--dropout"],
    )

    state_dict = {
        key[6:]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model = model.eval()
    model.initialize_state()

    test_dataset = CharCorpusDataset(
        data_path=args["--test-path"],
        char_tokenizer=char_tokenizer,
        word_tokenizer=word_tokenizer,
        add_sentence_end=True,
        max_word_length=args["--max-word-length"],
        sequence_length=args["--sequence-length"],
    )

    loss_function = TokenNLLLoss(reduction="sum", ignore_index=-100)

    total_loss = 0
    total_length = 0
    for inputs, targets in tqdm(test_dataset):
        inputs, targets = torch.utils.data._utils.collate.default_collate([(inputs, targets)])

        model.detach_state()
        outputs = model(inputs["token_ids"])

        loss_sum = loss_function(outputs=outputs, targets=targets["token_ids"])
        num_tokens = len(targets["token_ids"].view(-1))
        total_loss += loss_sum
        total_length += num_tokens

    print("Test Perplexity:", torch.exp(total_loss / total_length))
