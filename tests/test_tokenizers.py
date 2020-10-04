from build_vocabulary import CHAR_SPECIAL_TOKENS, WORD_SPECIAL_TOKENS, generate_sentences
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer

from . import SAMPLE_CHAR_VOCABULARY_PATH, SAMPLE_PATH, SAMPLE_WORD_VOCABULARY_PATH


def test_char_tokenizer_tokenize():
    assert CharTokenizer.tokenize("the number") == [
        ["t", "h", "e"],
        ["n", "u", "m", "b", "e", "r"],
    ]


def test_char_tokenizer_load():
    char_tokenizer = CharTokenizer.load(
        vocabulary_path=SAMPLE_CHAR_VOCABULARY_PATH, special_tokens=CHAR_SPECIAL_TOKENS
    )
    assert char_tokenizer.encode_as_ids("the number") == [
        [7, 21, 5],
        [10, 18, 15, 23, 5, 11],
    ]


def test_char_tokenizer_build():
    char_tokenizer = CharTokenizer.build_from_generator(
        sentences=generate_sentences(SAMPLE_PATH), special_tokens=CHAR_SPECIAL_TOKENS
    )

    assert char_tokenizer is not None
    assert len(char_tokenizer) == max(char_tokenizer.id_to_token.keys()) + 1  # indexing from zero


def test_word_tokenizer_tokenize():
    assert WordTokenizer.tokenize("the number") == ["the", "number"]


def test_word_tokenizer_load():
    word_tokenizer = WordTokenizer.load(
        vocabulary_path=SAMPLE_WORD_VOCABULARY_PATH, special_tokens=WORD_SPECIAL_TOKENS
    )
    assert word_tokenizer.encode_as_ids("the number") == [5, 48]


def test_word_tokenizer_build():
    word_tokenizer = WordTokenizer.build_from_generator(
        sentences=generate_sentences(SAMPLE_PATH), special_tokens=WORD_SPECIAL_TOKENS
    )

    assert word_tokenizer is not None
    assert len(word_tokenizer) == max(word_tokenizer.id_to_token.keys()) + 1  # indexing from zero
