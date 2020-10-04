from pathlib import Path
import torch
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from dataset import CHAR_SPECIAL_TOKENS, WORD_SPECIAL_TOKENS, SENTENCE_END_TOKEN
from model import CharLM
from typing import List
from dataset import Char, PAD_TOKEN, UNK_TOKEN, Word
from torch.utils.data.dataloader import default_collate


class Predictor:
    def __init__(self, model, char_tokenizer, word_tokenizer, hparams):
        self.model = model.eval()
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
        self.hparams = hparams

    def predict(self, input_text: str):

        input_text = input_text.strip()

        words = []
        for raw_word in input_text.split():
            chars: List[Char] = []
            for raw_char in raw_word:
                chars.append(raw_char)
            if not (raw_word == UNK_TOKEN or raw_word == PAD_TOKEN):
                chars.insert(0, self.char_tokenizer.special_tokens["word_start_token"])
                chars.append(self.char_tokenizer.special_tokens["word_end_token"])
            words.append(Word(chars=chars, word=raw_word))

        input_sequence = words

        input_token_ids = []
        for word in input_sequence:
            chars = word.chars[: self.hparams["--max-word-length"]]
            while len(chars) < self.hparams["--max-word-length"]:
                chars.append(self.char_tokenizer.special_tokens["pad_token"])
            char_ids = self.char_tokenizer.encode_chars_as_ids(chars)
            input_token_ids.append(char_ids)

        output_tokens = []
        for _ in range(30):

            inputs = {
                "token_ids": torch.tensor(input_token_ids),
                "length": len(input_token_ids),
            }
            inputs_tensor = default_collate([inputs])

            outputs = self.model(inputs_tensor["token_ids"])
            last_word_log_probs = outputs[0, -1]
            last_word_log_probs[1] = -1e8
            topk_values, topk_indices = last_word_log_probs.topk(k=1)
            raw_word = self.word_tokenizer.decode_ids(topk_indices.numpy())

            output_tokens.append(raw_word)

            if raw_word == self.word_tokenizer.special_tokens["sentence_end_token"]:
                break

            char_ids = self.char_tokenizer.encode_as_ids(raw_word)[0]
            if not (raw_word == UNK_TOKEN or raw_word == PAD_TOKEN):
                char_ids.insert(0, self.char_tokenizer.special_token_ids["word_start_token"])
                char_ids.append(self.char_tokenizer.special_token_ids["word_end_token"])

            while len(char_ids) < self.hparams["--max-word-length"]:
                char_ids.append(self.char_tokenizer.special_token_ids["pad_token"])

            input_token_ids = [char_ids]

        prediction_text = " ".join(output_tokens)
        prediction_text = prediction_text.replace(SENTENCE_END_TOKEN, "</s>")
        prediction_text = prediction_text.replace(UNK_TOKEN, "<unk>")

        return input_text + " " + prediction_text

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        hparams = checkpoint["hyper_parameters"]

        char_tokenizer = CharTokenizer.load(
            vocabulary_path=hparams["--char-vocabulary-path"], special_tokens=CHAR_SPECIAL_TOKENS,
        )
        word_tokenizer = WordTokenizer.load(
            vocabulary_path=hparams["--word-vocabulary-path"], special_tokens=WORD_SPECIAL_TOKENS,
        )

        num_chars = len(char_tokenizer)
        num_words = len(word_tokenizer)
        char_pad_token_index = char_tokenizer.special_token_ids["pad_token"]

        model = CharLM(
            num_chars=num_chars,
            num_words=num_words,
            char_embedding_dim=hparams["--char-embedding-dim"],
            char_padding_idx=char_pad_token_index,
            char_conv_kernel_sizes=hparams["--char-conv-kernel-sizes"],
            char_conv_out_channels=hparams["--char-conv-out-channels"],
            use_batch_norm=hparams["--use-batch-norm"],
            num_highway_layers=hparams["--num-highway-layers"],
            hidden_dim=hparams["--hidden-dim"],
            dropout=hparams["--dropout"],
        )

        state_dict = {
            key[6:]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
        model.load_state_dict(state_dict)

        return cls(
            model=model,
            char_tokenizer=char_tokenizer,
            word_tokenizer=word_tokenizer,
            hparams=hparams,
        )
