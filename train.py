"""
Usage:
    main.py train [options]
    main.py train (-h | --help)

Options:
    --train-val-dir <train-val-dir>  Prefix for train-path and val-path  [type: path]
    --train-path <train-path>  Path to train data  [type: path]
    --val-path <val-path>  Path to validation data  [type: path]
    --word-vocabulary-path <word-vocabulary-path>  Load word tokenizer from [type: path]
    --char-vocabulary-path <char-vocabulary-path>  Load char tokenizer from [type: path]
    
    --max-word-length <max-word-length>  Maximum number of chars in a word [type: int]
    --sequence-length <sequence-length>  Number of timesteps to unroll for [type: int]
    
    --char-embedding-dim <char-embedding-dim>  For char embedding layer of model [type: int]
    --hidden-dim <hidden-dim>  Hidden dimension for LSTM [type: int]
    --num-highway-layers <num-highway-layers>  [type: int]
    --use-batch-norm  Use BatchNorm1d after character convolutions
    --dropout <dropout>  Dropout probability for LSTM and output layer [type: float]

    --gradient-clip-val <gradient-clip-val>  [type: float]
    --lr <lr>  Learning rate [type: float]

    --batch-size <batch-size>  [type: int]
    --num-workers <num-workers>  Number of processes in dataloader [type: int]
    --max-epochs <max-epochs>  Training epochs [type: int]

    --run-dir <run-dir>  Prefix path for run direcotry [default: results/runs] [type: path]
    --name <name>  Experiment name. Outputs will be saved to {run-dir}/{name}/{version} [default: run] [type: path]

    -h --help  Show this.
"""
from pathlib import Path

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback
import torch
from lightning_dataloader import LanguageModelingDataModule
from lightning_model import LanguageModelingLightningModel
from utils import get_next_version


def train(hparams: dict):

    seed_everything(0)

    lm_data_module = LanguageModelingDataModule(hparams=hparams)

    lm_lightning_model = LanguageModelingLightningModel(
        hparams=hparams,
        num_chars=len(lm_data_module.char_tokenizer),
        num_words=len(lm_data_module.word_tokenizer),
        char_pad_token_index=lm_data_module.char_tokenizer.special_token_ids["pad_token"],
    )

    root_dir = hparams["--run-dir"].joinpath(hparams["--name"])
    next_version = get_next_version(root_dir=root_dir)
    version_dir = root_dir.joinpath(next_version)
    version_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(
        name=str(hparams["--name"]), save_dir=str(version_dir), offline=True, version=next_version,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=version_dir.joinpath("checkpoints", "{epoch:0>3}_{val_ppl:.5f}"),
        monitor="val_ppl",
        save_top_k=1,
        mode="min",
        period=1,
    )

    trainer = Trainer(
        accumulate_grad_batches=1,
        amp_level="O2",
        # TODO: auto_scale_batch_size='binsearch',
        # TODO: auto_lr_find='learning_rate',
        benchmark=False,
        deterministic=True,
        callbacks=[LearningRateLogger(), StickingProgressBarCallback(hparams)],
        check_val_every_n_epoch=1,
        checkpoint_callback=checkpoint_callback,
        distributed_backend="dp",
        fast_dev_run=False,
        gpus=-1 if torch.cuda.is_available() else None,
        gradient_clip_val=hparams["--gradient-clip-val"],
        log_save_interval=10,
        logger=logger,
        max_epochs=hparams["--max-epochs"],
        num_sanity_val_steps=5,
        overfit_batches=0.0,
        precision=32,
        profiler=True,
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=None,
        row_log_interval=10,
        weights_summary="top",
    )
    trainer.fit(lm_lightning_model, lm_data_module)


class StickingProgressBarCallback(Callback):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

    def on_epoch_start(self, trainer, pl_module):
        print(" ")
