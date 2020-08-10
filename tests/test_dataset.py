from dataset import CharCorpusDataset, Word, WORD_START_TOKEN, WORD_END_TOKEN
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from build_vocabulary import generate_sentences, CHAR_SPECIAL_TOKENS, WORD_SPECIAL_TOKENS
from . import SAMPLE_PATH, SAMPLE_CHAR_VOCABULARY_PATH, SAMPLE_WORD_VOCABULARY_PATH


def test_normalize_text():
    raw_line = "<unk> <unk> watching abc 's monday night football can now vote during <unk> for the greatest play in N years from among four or five <unk> <unk> "
    clean_line = "| | watching abc 's monday night football can now vote during | for the greatest play in N years from among four or five | |"
    assert CharCorpusDataset.normalize_line(raw_line) == clean_line


def test_construct_corpus():
    char_tokenizer = CharTokenizer.load(vocabulary_path=SAMPLE_CHAR_VOCABULARY_PATH, special_tokens=CHAR_SPECIAL_TOKENS)
    word_tokenizer = WordTokenizer.load(vocabulary_path=SAMPLE_WORD_VOCABULARY_PATH, special_tokens=WORD_SPECIAL_TOKENS)
    dataset = CharCorpusDataset.construct_corpus(
        data_path=SAMPLE_PATH, char_tokenizer=char_tokenizer, word_tokenizer=word_tokenizer, add_sentence_end=True,
    )
    assert len(dataset.sentences) == 3
    assert dataset.sentences[0].words[0] == Word(
        chars=[WORD_START_TOKEN, "c", "o", "n", "s", "u", "m", "e", "r", "s", WORD_END_TOKEN], word="consumers"
    )


def test_corpus_dataset():
    char_tokenizer = CharTokenizer.load(vocabulary_path=SAMPLE_CHAR_VOCABULARY_PATH, special_tokens=CHAR_SPECIAL_TOKENS)
    word_tokenizer = WordTokenizer.load(vocabulary_path=SAMPLE_WORD_VOCABULARY_PATH, special_tokens=WORD_SPECIAL_TOKENS)
    dataset = CharCorpusDataset(
        data_path=SAMPLE_PATH,
        char_tokenizer=char_tokenizer,
        word_tokenizer=word_tokenizer,
        add_sentence_end=True,
        max_word_length=5,
        sequence_length=10,
    )
    print("dataset[0]", dataset[0])
    assert dataset[0] == (
        {
            "token_ids": [
                [2, 14, 6, 10, 8],
                [2, 15, 9, 22, 3],
                [2, 20, 9, 10, 7],
                [2, 7, 6, 3],
                [2, 15, 6, 16, 5],
                [2, 7, 21, 5, 12],
                [2, 7, 5, 13, 5],
                [2, 9, 3],
                [2, 13, 12, 7, 7],
            ],
            "length": 9,
        },
        {"token_ids": [9, 10, 3, 11, 12, 13, 4, 14, 15], "length": 9},
    )
