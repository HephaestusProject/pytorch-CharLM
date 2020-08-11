import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule
from model import CharLM
from losses import TokenNLLLoss
from metrics import perplexity_score
from pathlib import Path


class LanguageModelingLightningModel(LightningModule):
    def __init__(self, hparams, char_vocab_size, word_vocab_size, pad_token_index):
        super().__init__()

        self.model = CharLM(
            char_emb_dim=100, word_emb_dim=100, vocab_size=10000, num_char=50
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
        return Adam(self.parameters(), lr=0.001)
