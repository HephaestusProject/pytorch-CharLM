from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple


class CharTokenizer:
    def __init__(self, vocabulary: List[Tuple[int, str, int]], special_tokens: dict):
        self.token_to_id = {token: token_id for token_id, token, count in vocabulary}
        self.id_to_token = {token_id: token for token_id, token, count in vocabulary}

        self.special_token_ids = {
            token_name: self.token_to_id[token] for token_name, token in special_tokens.items()
        }

        self.special_tokens = special_tokens
        self.vocabulary = vocabulary

    @staticmethod
    def tokenize(sentence: str, unk_token="<unk>"):
        return [
            list(word) if word != unk_token else [unk_token] for word in sentence.strip().split()
        ]

    @staticmethod
    def decode_tokens(tokens: List[List[str]]) -> str:
        return " ".join("".join(chars) for chars in tokens)

    def encode_as_ids(self, sentence: str, unk_token_name: str = "unk_token") -> List[List[int]]:
        return [
            [self.token_to_id.get(char, self.special_token_ids[unk_token_name]) for char in chars]
            for chars in self.tokenize(sentence)
        ]

    def encode_chars_as_ids(self, chars: List[str], unk_token_name: str = "unk_token"):
        return [
            self.token_to_id.get(char, self.special_token_ids[unk_token_name]) for char in chars
        ]

    def encode_as_tokens(self, sentence: str, unk_token_name: str = "unk_token") -> List[List[str]]:
        return [
            [
                (char if char in self.token_to_id else self.special_tokens[unk_token_name])
                for char in chars
            ]
            for chars in self.tokenize(sentence)
        ]

    def decode_ids(self, token_ids: List[List[int]]) -> str:
        # Raises error when token_id is not in vocab
        tokens = [[self.id_to_token[char_id] for char_id in char_ids] for char_ids in token_ids]
        return self.decode_tokens(tokens)

    @classmethod
    def build_from_generator(
        cls, sentences, special_tokens: dict, vocabulary_size: Optional[int] = None
    ):
        def generate_tokens(s):
            for sentence in s:
                for chars in CharTokenizer.tokenize(sentence):
                    yield from chars

        token_generator = generate_tokens(sentences)
        counter = Counter(token_generator)

        n = vocabulary_size - len(special_tokens) if vocabulary_size is not None else None
        most_commons = counter.most_common(n)

        most_commons_without_special_tokens = [
            (token, count) for token, count in most_commons if token not in special_tokens.values()
        ]
        special_tokens_with_count = [
            (special_token, 0) for special_token in special_tokens.values()
        ]
        all_tokens = special_tokens_with_count + most_commons_without_special_tokens

        vocabulary = [
            (token_id, token, count) for token_id, (token, count) in enumerate(all_tokens)
        ]

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
