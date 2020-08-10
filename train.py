"""
Usage:
    main.py train [options]
    main.py train (-h | --help)

Options:
    --train-val-dir <train-val-dir>  Prefix for train-path and val-path  [type: path]
    --train-path <train-path>  Path to train data  [type: path]
    --val-path <val-path>  Path to validation data  [type: path]
    --run-dir <run-dir>  Prefix path for run direcotry [default: results/runs] [type: path]
    --name <name>  Experiment name. Outputs will be saved to {run-dir}/{name}/{version} [default: run] [type: path]

    -h --help  Show this.
"""
from pytorch_lightning import LightningModule, Trainer
from dataloader import LanguageModelingDataModule
from lightning_model import LanguageModelingLightningModel

def train(hparams: dict):

    lm_data_module = LanguageModelingDataModule(hparams=hparams
    lm_lightning_model = LanguageModelingLightningModel(hparams=hparams)
)
    checkpoint_callback = ModelCheckpoint(
        filepath=logger_callback.log_dir.joinpath("checkpoints", "{epoch:0>3}_{val_ppl:.5f}"),
        monitor="val_ppl",
        save_top_k=1,
        mode="min",
        period=1,
    )

    trainer = Trainer(
        logger=logger_callback,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=False,
        gradient_clip_val=hparams["--gradient-clip-val"],
        gpus=-1 if not hparams["--cpu"] and torch.cuda.is_available() else None,
        progress_bar_refresh_rate=1,
        overfit_pct=hparams["--subset-ratio"],
        check_val_every_n_epoch=hparams["--check-val-every-n-epoch"],
        fast_dev_run=hparams["--debug"],
        accumulate_grad_batches=hparams["--accumulate-grad-batches"],
        max_epochs=hparams["--epochs"],
        val_check_interval=hparams["--val-check-interval"] or 1.0,
        log_save_interval=hparams["--log-interval"],
        row_log_interval=hparams["--log-interval"],
        distributed_backend="dp",
        precision=16 if not hparams["--disable-amp"] and not hparams["--cpu"] and torch.cuda.is_available() else 32,
        weights_summary="top",
        amp_level=hparams["--amp-level"],
        resume_from_checkpoint=hparams["--resume-from-checkpoint"],
        profiler=Profiler(),
    )
    trainer.fit(lm_model)


class LanguageModelingLightning(LightningModule):
    def __init__(self, hparams):

        if hparams["--train-val-dir"] is None:
            hparams["--train-val-dir"] = Path()
        hparams["train-path"] = hparams["--train-val-dir"].joinpath(hparams["--train-path"])
        hparams["val-path"] = hparams["--train-val-dir"].joinpath(hparams["--val-path"])

        self.model CharLM()

        self.char_tokenizer = CharTokenizer
        self.word_tokenizer = WordTokenizer
