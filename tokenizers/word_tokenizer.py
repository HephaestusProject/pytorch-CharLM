from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple


class WordTokenizer:
    def __init__(self, vocabulary: List[Tuple[int, str, int]], special_tokens: dict):
        self.token_to_id = {token: token_id for token_id, token, count in vocabulary}
        self.id_to_token = {token_id: token for token_id, token, count in vocabulary}

        self.special_token_ids = {token_name: self.token_to_id[token] for token_name, token in special_tokens.items()}

        self.special_tokens = special_tokens
        self.vocabulary = vocabulary

    @staticmethod
    def tokenize(sentence: str):
        return sentence.strip().split()

    @staticmethod
    def decode_tokens(tokens: List[str]) -> str:
        return " ".join(tokens)

    def encode_as_ids(self, sentence: str, unk_token_name: str = "unk_token") -> List[int]:
        return [
            self.token_to_id.get(token, self.special_token_ids[unk_token_name]) for token in self.tokenize(sentence)
        ]

    def encode_words_as_ids(self, words: List[str], unk_token_name: str = "unk_token"):
        return [self.token_to_id.get(token, self.special_token_ids[unk_token_name]) for token in words]

    def encode_as_tokens(self, sentence: str, unk_token_name: str = "unk_token") -> List[str]:
        return [
            (token if token in self.token_to_id else self.special_tokens[unk_token_name])
            for token in self.tokenize(sentence)
        ]

    def decode_ids(self, token_ids: List[int]) -> str:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]  # Raises error when token_id is not in vocab
        return self.decode_tokens(tokens)

    def decode_until_end(self, token_ids: List[int], end_token_name: str = "end_token") -> str:
        token_ids_until_end = []
        for token_id in token_ids:
            if token_id == self.special_token_ids[end_token_name]:
                break
            token_ids_until_end.append(token_id)
        return self.decode_ids(token_ids_until_end)

    @classmethod
    def build_from_generator(cls, sentences, special_tokens: dict, vocabulary_size: Optional[int] = None):
        def generate_tokens(s):
            for sentence in s:
                yield from WordTokenizer.tokenize(sentence)

        token_generator = generate_tokens(sentences)
        counter = Counter(token_generator)

        n = vocabulary_size - len(special_tokens) if vocabulary_size is not None else None
        most_commons = counter.most_common(n)

        most_commons_without_special_tokens = [
            (token, count) for token, count in most_commons if token not in special_tokens.values()
        ]
        special_tokens_with_count = [(special_token, 0) for special_token in special_tokens.values()]
        all_tokens = special_tokens_with_count + most_commons_without_special_tokens

        vocabulary = [(token_id, token, count) for token_id, (token, count) in enumerate(all_tokens)]

        instance = cls(vocabulary=vocabulary, special_tokens=special_tokens)

        return instance

    @classmethod
    def load(cls, vocabulary_path: Path, special_tokens: dict):
        vocabulary = []

        with vocabulary_path.open() as vocabulary_file:
            for line in vocabulary_file:
                if line.startswith("#"):
                    continue
                token_id, token, count = line.strip().split("\t")
                assert token == " " or len(token.strip()) > 0
                vocabulary.append((int(token_id), token, int(count)))

        instance = cls(vocabulary=vocabulary, special_tokens=special_tokens)
        return instance

    def save(self, vocabulary_path: Path):
        with vocabulary_path.open("w") as vocabulary_file:
            for token_id, token, count in sorted(self.vocabulary, key=lambda x: x[0]):
                line = f"{token_id}\t{token}\t{count}\n"
                vocabulary_file.write(line)

    def add_new_token(self, token: str):
        assert token not in self.token_to_id
        new_token_id = len(self.vocabulary)
        self.vocabulary.append((new_token_id, token, 1))
        self.token_to_id[token] = new_token_id
        self.id_to_token[new_token_id] = token

    def __len__(self):
        return len(self.vocabulary)
