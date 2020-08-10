from pytorch_lightning import LightningDataModule
from dataset import CHAR_SPECIAL_TOKENS, WORD_SPECIAL_TOKENS, CharCorpusDataset

from torch.utils.data import DataLoader
from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer
import subprocess
from pathlib import Path


class LanguageModelingDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        if hparams["--train-val-dir"] is None:
            hparams["--train-val-dir"] = Path()
        self.train_path = hparams["--train-val-dir"].joinpath(hparams["--train-path"])
        self.val_path = hparams["--train-val-dir"].joinpath(hparams["--val-path"])

        self.char_tokenizer = CharTokenizer.load(
            vocabulary_path=hparams["--char-vocabulary-path"], special_tokens=CHAR_SPECIAL_TOKENS
        )
        self.word_tokenizer = WordTokenizer.load(
            vocabulary_path=hparams["--word-vocabulary-path"], special_tokens=WORD_SPECIAL_TOKENS
        )

        self.hparams = hparams

    def prepare_data(self):
        subprocess.run("./download_ptb.sh", check=True)

    def setup(self):
        pass

    def train_dataloader(self):
        train_dataset = CharCorpusDataset(
            data_path=self.train_path,
            char_tokenizer=self.char_tokenizer,
            word_tokenizer=self.word_tokenizer,
            add_sentence_end=True,
            max_word_length=self.hparams["--max-word-length"],
            sequence_length=self.hparams["--sequence-length"],
        )

        return DataLoader(
            train_dataset,
            batch_size=self.hparams["--batch-size"],
            shuffle=True,
            num_workers=self.hparams["--num-workers"],
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = CharCorpusDataset(
            data_path=self.val_path,
            char_tokenizer=self.char_tokenizer,
            word_tokenizer=self.word_tokenizer,
            add_sentence_end=True,
            max_word_length=self.hparams["--max-word-length"],
            sequence_length=self.hparams["--sequence-length"],
        )

        return DataLoader(
            val_dataset,
            batch_size=self.hparams["--batch-size"],
            num_workers=self.hparams["--num-workers"],
            pin_memory=True,
        )

    def test_dataloader(self):
        pass
