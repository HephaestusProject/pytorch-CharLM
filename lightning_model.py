from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from losses import TokenNLLLoss
from metrics import perplexity_score
from model import CharLM


class LanguageModelingLightningModel(LightningModule):
    def __init__(self, hparams, num_chars, num_words, char_pad_token_index):
        super().__init__()

        self.model = CharLM(
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
        self.loss_function = TokenNLLLoss(reduction="mean", ignore_index=-100)
        self.batch_size = None
        self.learning_rate = None
        self.hparams = hparams

    def forward(self, inputs):
        self.model.detach_state()
        outputs = self.model(inputs["token_ids"])
        return outputs

    def on_epoch_start(self):
        self.model.initialize_state()

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

    def on_validation_epoch_start(self):
        self.model.initialize_state()

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs=outputs, targets=targets["token_ids"])
        perplexity = perplexity_score(loss)

        return {"val_loss": loss, "val_ppl": perplexity}

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_ppl = torch.mean(torch.stack([output["val_ppl"] for output in outputs]))
        return {
            "progress_bar": {"val_loss": val_loss, "val_ppl": val_ppl},
            "log": {"val_loss": val_loss, "val_ppl": val_ppl},
        }

    def configure_optimizers(self):

        optimizer = SGD(self.parameters(), lr=self.hparams["--lr"])
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=0,
            verbose=True,
            threshold=1,
            threshold_mode="abs",
        )
        lr_scheduler_hparams = {
            "scheduler": lr_scheduler,
            "monitor": "val_ppl",
            "interval": "epoch",
            "frequency": 1,
            "reduce_on_plateau": True,
        }

        return [optimizer], [lr_scheduler_hparams]
