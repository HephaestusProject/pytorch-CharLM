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

from dataset import CHAR_SPECIAL_TOKENS, WORD_SPECIAL_TOKENS, CharCorpusDataset
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer


def build_vocabulary(hparams: dict):
    word_tokenizer = WordTokenizer.build_from_generator(
        sentences=generate_sentences(hparams["--data-path"]), special_tokens=WORD_SPECIAL_TOKENS,
    )

    char_tokenizer = CharTokenizer.build_from_generator(
        sentences=generate_sentences(hparams["--data-path"]), special_tokens=CHAR_SPECIAL_TOKENS,
    )

    word_tokenizer.save(vocabulary_path=hparams["--word-vocabulary-path"])
    char_tokenizer.save(vocabulary_path=hparams["--char-vocabulary-path"])


def generate_sentences(data_path: Path):
    with data_path.open() as data_file:
        for line in data_file:
            line = CharCorpusDataset.normalize_line(line)
            yield line
