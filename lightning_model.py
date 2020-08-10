from pytorch_lightning import LightningModule
from model import CharLM


class LanguageModelingLightningModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.model = CharLM(char_emb_dim=hparams, word_emb_dim=hparams, vocab_size=hparams, num_char=...)

    def forward(self, inputs):
        pass

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs=outputs, targets=targets["token_ids"])
        perplexity = perplexity_score(loss)

        return {
            "loss": scaled_loss,
            "progress_bar": {"train_ppl": perplexity, "lr": current_lr, "epoch": self.current_epoch},
            "log": {"train_ppl": perplexity, "train_loss": scaled_loss, "lr": current_lr, "epoch": self.current_epoch},
        }

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):