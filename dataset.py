import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from itertools import chain

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
        add_sentence_end: bool,
        max_word_length: int,
        sequence_length: int,
        drop_last: bool = True,
    ):
        super(CharCorpusDataset, self).__init__()

        corpus = self.construct_corpus(
            data_path=data_path, char_tokenizer=char_tokenizer, add_sentence_end=add_sentence_end,
        )

        flattened_corpus = []
        for sentence in corpus.sentences:
            for word in sentence.words:
                flattened_corpus.extend(word.chars)
                flattened_corpus.append(PAD_TOKEN)
            # flattened_corpus.append(SENTENCE_END_TOKEN)

        self.flattened_corpus = flattened_corpus
        self.char_tokenizer = char_tokenizer

        self.sequence_length = sequence_length
        self.drop_last = drop_last

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        sequence_pointer = item * self.sequence_length
        sequence_end_pointer = min(
            sequence_pointer + self.sequence_length, len(self.flattened_corpus) - 1
        )
        input_sequence = self.flattened_corpus[sequence_pointer:sequence_end_pointer]
        output_sequence = self.flattened_corpus[sequence_pointer + 1 : sequence_end_pointer + 1]

        input_token_ids = self.char_tokenizer.encode_chars_as_ids(input_sequence)
        target_token_ids = self.char_tokenizer.encode_chars_as_ids(output_sequence)

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
        if self.drop_last:
            return input_sequence_size // self.sequence_length
        else:
            return math.ceil(input_sequence_size / self.sequence_length)

    @staticmethod
    def construct_corpus(
        data_path: Path, char_tokenizer: CharTokenizer, add_sentence_end: bool,
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
                    words.append(Word(chars=chars, word=raw_word))
                if add_sentence_end:
                    chars: List[Char] = []
                    chars.append(char_tokenizer.special_tokens["sentence_end_token"])
                    words.append(Word(chars=chars, word=SENTENCE_END_TOKEN))
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
