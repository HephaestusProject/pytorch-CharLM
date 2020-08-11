from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from losses import TokenNLLLoss
from metrics import perplexity_score
from model import CharLM


class LanguageModelingLightningModel(LightningModule):
    def __init__(self, hparams, num_chars, num_words, pad_token_index):
        super().__init__()

        self.model = CharLM(
            num_chars=num_chars,
            num_words=num_words,
            char_embedding_dim=hparams["--char-embedding-dim"],
            word_embedding_dim=hparams["--word-embedding-dim"],
        )
        self.loss_function = TokenNLLLoss(ignore_index=pad_token_index)
        self.batch_size = None
        self.learning_rate = None

    def forward(self, inputs):
        outputs, _ = self.model(inputs["token_ids"], hidden=None)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs=outputs, targets=targets["token_ids"])
        perplexity = perplexity_score(loss)

        return {
            "loss": loss,
            "progress_bar": {"train_ppl": perplexity},
            "log": {"train_ppl": perplexity, "train_loss": loss},
        }

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs=outputs, targets=targets["token_ids"])
        perplexity = perplexity_score(loss)

        return {
            "loss": loss,
            "progress_bar": {"val_ppl": perplexity},
            "log": {"val_ppl": perplexity, "val_loss": loss},
        }

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=0.001)
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)
        return [optimizer], [lr_scheduler]
