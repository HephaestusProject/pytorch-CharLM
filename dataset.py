from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer

UNKNOWN_SYMBOL = "<unk>"

PAD_TOKEN = "_"
UNK_TOKEN = "|"
SENTENCE_END_TOKEN = "+"
WORD_START_TOKEN = "{"
WORD_END_TOKEN = "}"

WORD_SPECIAL_TOKENS = {
    "pad_token": PAD_TOKEN,
    "unk_token": UNK_TOKEN,
    "sentence_end_token": SENTENCE_END_TOKEN,
}

CHAR_SPECIAL_TOKENS = {
    "pad_token": PAD_TOKEN,
    "unk_token": UNK_TOKEN,
    "word_start_token": WORD_START_TOKEN,
    "word_end_token": WORD_END_TOKEN,
    "sentence_end_token": SENTENCE_END_TOKEN,
}


Char = str


@dataclass
class Word:
    chars: List[Char]
    word: str


@dataclass
class Sentence:
    words: List[Word]


@dataclass
class Corpus:
    sentences: List[Sentence]


class CharCorpusDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        char_tokenizer: CharTokenizer,
        word_tokenizer: WordTokenizer,
        add_sentence_end: bool,
        max_word_length: int,
        sequence_length: int,
    ):
        super(CharCorpusDataset, self).__init__()

        corpus = self.construct_corpus(
            data_path=data_path,
            char_tokenizer=char_tokenizer,
            word_tokenizer=word_tokenizer,
            add_sentence_end=add_sentence_end,
        )

        flattened_corpus = []
        for sentence in corpus.sentences:
            flattened_corpus.extend(sentence.words)

        max_word = max(flattened_corpus, key=lambda word: len(word.chars))
        self.max_word_length = min(max_word_length, len(max_word.chars))
        print("max_word_length", self.max_word_length)

        self.flattened_corpus = flattened_corpus
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
        self.sequence_length = sequence_length

    def __getitem__(self, item):
        sequence_pointer = item * self.sequence_length
        input_sequence = self.flattened_corpus[
            sequence_pointer : sequence_pointer + self.sequence_length
        ]
        output_sequence = self.flattened_corpus[
            sequence_pointer + 1 : sequence_pointer + self.sequence_length + 1
        ]

        input_token_ids = []
        for word in input_sequence:
            chars = word.chars[: self.max_word_length]
            while len(chars) < self.max_word_length:
                chars.append(self.char_tokenizer.special_tokens["pad_token"])
            char_ids = self.char_tokenizer.encode_chars_as_ids(chars)
            input_token_ids.append(char_ids)

        target_token_ids = self.word_tokenizer.encode_words_as_ids(
            [word.word for word in output_sequence]
        )

        inputs = {
            "token_ids": torch.tensor(input_token_ids),
            "length": len(input_token_ids),
        }
        targets = {
            "token_ids": torch.tensor(target_token_ids),
            "length": len(target_token_ids),
        }

        return inputs, targets

    def __len__(self):
        input_sequence_size = len(self.flattened_corpus) - 1
        return input_sequence_size // self.sequence_length

    @staticmethod
    def construct_corpus(
        data_path: Path,
        char_tokenizer: CharTokenizer,
        word_tokenizer: WordTokenizer,
        add_sentence_end: bool,
    ) -> Corpus:
        sentences = []
        with data_path.open() as data_file:
            for line in data_file:
                line = CharCorpusDataset.normalize_line(line)
                words = []
                for raw_word in line.split():
                    chars: List[Char] = []
                    for raw_char in raw_word:
                        chars.append(raw_char)
                    if not (raw_word == UNK_TOKEN or raw_word == PAD_TOKEN):
                        chars.insert(0, char_tokenizer.special_tokens["word_start_token"])
                        chars.append(char_tokenizer.special_tokens["word_end_token"])
                    words.append(Word(chars=chars, word=raw_word))
                if add_sentence_end:
                    chars: List[Char] = []
                    chars.append(char_tokenizer.special_tokens["sentence_end_token"])
                    words.append(
                        Word(chars=chars, word=word_tokenizer.special_tokens["sentence_end_token"],)
                    )
                sentences.append(Sentence(words=words))

        corpus = Corpus(sentences=sentences)
        return corpus

    @staticmethod
    def normalize_line(line: str):
        line = line.replace(UNKNOWN_SYMBOL, UNK_TOKEN)
        line = line.replace(WORD_START_TOKEN, "")
        line = line.replace(WORD_END_TOKEN, "")
        line = line.strip()
        return line
