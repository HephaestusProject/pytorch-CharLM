from dataclasses import dataclass
from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.char_tokenizer import CharTokenizer

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
        add_sentence_end,
        max_word_length,
        sequence_length,
    ):
        super(CharCorpusDataset, self).__init__()

        sentences = []
        with data_path.open() as data_file:
            for line in data_file:
                raw_words = line.strip().split()
                words = []
                for raw_word in raw_words:
                    chars: List[Char] = []
                    chars.append(char_tokenizer.special_tokensword_start_token)
                    if raw_word == word_tokenizer.unk_token:
                        chars.append(char_tokenizer.unk_token)
                    else:
                        chars.extend(raw_word)
                    chars.append(char_tokenizer.word_end_token)
                    words.append(Word(chars=chars, word=raw_word))
                if add_sentence_end:
                    words.append(
                        Word(chars=[char_tokenizer.sentence_end_token], word=word_tokenizer.sentence_end_token,)
                    )
                sentences.append(Sentence(words=words))

        corpus = Corpus(sentences=sentences)

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
            char_ids = self.char_tokenizer.encode_as_ids(word.chars[: self.max_word_length])
            sequence_char_ids.append(char_ids)

        sequence_word_ids = self.word_tokenizer.encode_as_ids([word.word for word in data_sequence])

        input_token_ids = sequence_char_ids[:-1]
        target_token_ids = sequence_word_ids[1:]

        inputs = {"token_ids": input_token_ids, "length": len(input_token_ids)}
        targets = {"token_ids": target_token_ids, "length": len(target_token_ids)}

        return inputs, targets

    def __len__(self):
        return len(self.flattened_corpus) // self.sequence_length
