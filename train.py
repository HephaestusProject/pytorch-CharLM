"""
Usage:
    main.py train [options]
    main.py train (-h | --help)

Options:
    --train-val-dir <train-val-dir>  Prefix for train-path and val-path  [type: path]
    --train-path <train-path>  Path to train data  [type: path]
    --val-path <val-path>  Path to validation data  [type: path]
    --word-vocabulary-path <word-vocabulary-path>  [type: path]
    --char-vocabulary-path <char-vocabulary-path>  [type: path]
    --max-word-length <max-word-length>  [type: int]
    --sequence-length <sequence-length>  [type: int]
    --batch-size <batch-size>  [type: int]
    --num-workers <num-workers>  [type: int]

    --run-dir <run-dir>  Prefix path for run direcotry [default: results/runs] [type: path]
    --name <name>  Experiment name. Outputs will be saved to {run-dir}/{name}/{version} [default: run] [type: path]

    -h --help  Show this.
"""
from pytorch_lightning import LightningModule, Trainer, seed_everything
from dataloader import LanguageModelingDataModule
from lightning_model import LanguageModelingLightningModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from utils import next_version_dir_name


def train(hparams: dict):

    seed_everything(0)

    lm_data_module = LanguageModelingDataModule(hparams=hparams)

    lm_lightning_model = LanguageModelingLightningModel(
        hparams=hparams, char_vocab_size=100, word_vocab_size=100, pad_token_index=0
    )

    root_dir = hparams["--run-dir"].joinpath(hparams["--name"])
    version_dir = root_dir.joinpath(next_version_dir_name(root_dir))
    version_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(name=str(hparams["--name"]), save_dir=str(version_dir), offline=True)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=version_dir.joinpath("checkpoints", "{epoch:0>3}_{val_ppl:.5f}"),
        monitor="val_ppl",
        save_top_k=1,
        mode="min",
        period=1,
    )

    lr_logger = LearningRateLogger()

    trainer = Trainer(
        accumulate_grad_batches=1,
        amp_level="O2",
        # TODO: auto_scale_batch_size='binsearch',
        # TODO: auto_lr_find='learning_rate',
        benchmark=False,
        deterministic=True,
        callbacks=[lr_logger],
        check_val_every_n_epoch=1,
        checkpoint_callback=checkpoint_callback,
        distributed_backend="dp",
        fast_dev_run=False,
        gpus=None,
        gradient_clip_val=0.0,
        log_save_interval=10,
        logger=logger,
        max_epochs=100,
        num_sanity_val_steps=5,
        overfit_batches=0.01,
        precision=32,
        profiler=True,
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=None,
        weights_summary="top",
    )
    trainer.fit(lm_lightning_model, lm_data_module)

