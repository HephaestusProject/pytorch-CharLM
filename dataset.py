from dataclasses import dataclass
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.char_tokenizer import CharTokenizer


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

        self.flattened_corpus = flattened_corpus
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
        self.sequence_length = sequence_length
        self.max_word_length = max_word_length

    def __getitem__(self, item):
        data_sequence = self.flattened_corpus[item * self.sequence_length : (item + 1) * self.sequence_length]
        sequence_char_ids = []
        for word in data_sequence:
            char_ids = self.char_tokenizer.encode_chars_as_ids(word.chars[: self.max_word_length])
            sequence_char_ids.append(char_ids)

        sequence_word_ids = self.word_tokenizer.encode_words_as_ids([word.word for word in data_sequence])

        input_token_ids = sequence_char_ids[:-1]
        target_token_ids = sequence_word_ids[1:]

        inputs = {"token_ids": input_token_ids, "length": len(input_token_ids)}
        targets = {"token_ids": target_token_ids, "length": len(target_token_ids)}

        return inputs, targets

    def __len__(self):
        return len(self.flattened_corpus) // self.sequence_length

    @staticmethod
    def construct_corpus(
        data_path: Path, char_tokenizer: CharTokenizer, word_tokenizer: WordTokenizer, add_sentence_end: bool,
    ) -> Corpus:
        sentences = []
        with data_path.open() as data_file:
            for line in data_file:
                line = CharCorpusDataset.normalize_line(line)
                words = []
                for raw_word in line.split():
                    chars: List[Char] = []
                    chars.append(char_tokenizer.special_tokens["word_start_token"])
                    for raw_char in raw_word:
                        chars.append(raw_char)
                    chars.append(char_tokenizer.special_tokens["word_end_token"])
                    words.append(Word(chars=chars, word=raw_word))
                if add_sentence_end:
                    chars: List[Char] = []
                    chars.append(char_tokenizer.special_tokens["word_start_token"])
                    chars.append(char_tokenizer.special_tokens["sentence_end_token"])
                    chars.append(char_tokenizer.special_tokens["word_end_token"])
                    words.append(Word(chars=chars, word=word_tokenizer.special_tokens["sentence_end_token"]))
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
