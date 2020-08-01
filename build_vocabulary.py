"""
Usage:
    main.py build-vocabulary [options]
    main.py build-vocabulary (-h | --help)

Options:
    --data-path <data-path>  [type: path]
    --word-vocabulary-path <word-vocabulary-path>  [type: path]
    --char-vocabulary-path <char-vocabulary-path>  [type: path]

    -h --help  Show this.
"""

from pathlib import Path

from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.char_tokenizer import CharTokenizer


def build_vocabulary(hparams: dict):
    def generate_sentences(data_path: Path):
        with data_path.open() as data_file:
            for line in data_file:
                yield line

    word_special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "sentence_end_token": "[SENTENCE_END]",
    }

    word_tokenizer = WordTokenizer.build_from_generator(
        sentences=generate_sentences(hparams["--data-path"]), special_tokens=word_special_tokens
    )

    char_special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "word_start_token": "[WORD_START]",
        "word_end_token": "[WORD_END]",
        "sentence_end_token": "[SENTENCE_END]",
    }

    char_tokenizer = CharTokenizer.build_from_generator(
        sentences=generate_sentences(hparams["--data-path"]), special_tokens=char_special_tokens
    )

    word_tokenizer.save(vocabulary_path=hparams["--word-vocabulary-path"])
    char_tokenizer.save(vocabulary_path=hparams["--char-vocabulary-path"])
