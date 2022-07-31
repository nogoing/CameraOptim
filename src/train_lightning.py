import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase

from src.datamodules.co3d.dataset.dataset_zoo import dataset_zoo
from src.datamodules.co3d.dataset.dataloader_zoo import dataloader_zoo

from src.datamodules.nerf_synthetic.nerf_synthetic import NerfSyntheticDataset
from torch.utils.data import DataLoader

from src.models.model import NerFormer
from src.models.camera_optim_model import CameraOptimNerFormer
from src.utils import hydra_utils


log = hydra_utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Seed
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    if config.datamodule.dataset == "co3d":
        # Init CO3D Dataset
        datasets = dataset_zoo(
                category=config.datamodule.co3d_category,
                assert_single_seq=(config.datamodule.co3d_task == "singlesequence"),
                dataset_name=f"co3d_{config.datamodule.co3d_task}",
                test_on_train=False,
                load_point_clouds=False,
                test_restrict_sequence_id=config.datamodule.co3d_single_sequence_id,
            )

        # Init CO3D Dataset loader
        dataloaders = dataloader_zoo(
                datasets,
                dataset_name=f"co3d_{config.datamodule.co3d_task}",
                batch_size=(config.datamodule.N_src + config.datamodule.N_src_extra),
                # num_workers=1,
                dataset_len=config.datamodule.train_len,
                dataset_len_val=config.datamodule.val_len,
                images_per_seq_options=[100],
            )
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        # test_loader = dataloaders["test"]
    elif config.datamodule.dataset == "nerf_synthetic":
        nerf_synthetic_train = NerfSyntheticDataset(N_src=config.datamodule.N_src, mode="train", scenes=config.datamodule.scenes)
        nerf_synthetic_val = NerfSyntheticDataset(N_src=config.datamodule.N_src, mode="val", scenes=config.datamodule.scenes)
        train_loader = DataLoader(nerf_synthetic_train, batch_size=1, pin_memory=True, shuffle=True)
        val_loader = DataLoader(nerf_synthetic_val, batch_size=1, pin_memory=True, shuffle=True)

    # Init Nerformer model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger[0], _convert_="partial", detect_anomaly=True
    )
    # trainer = Trainer(gpus=config.trainer.gpus, num_nodes=config.trainer.num_nodes, precision=config.trainer.precision,
    #                     max_steps=config.trainer.max_steps, 
    #                     val_check_interval=config.trainer.val_check_interval,
    #                     callbacks=callbacks,
    #                     logger=logger,
    #                     num_sanity_val_steps=config.trainer.num_sanity_val_steps
    #                     )
    
    # Train the model
    log.info("Starting training!")
    trainer.fit(model, train_loader, val_loader)
    # model =  CameraOptimNerFormer.load_from_checkpoint("/home/kmuvcl/NVS/FastNeRFormer/src/models/step_18999.ckpt")
    # trainer.fit(model, train_loader)

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")