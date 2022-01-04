import os
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from co3d.dataset.dataset_zoo import dataset_zoo
from co3d.dataset.dataloader_zoo import dataloader_zoo

from network.model import NerFormer

from omegaconf import OmegaConf




def train(args):
    # CO3D Dataset
    datasets = dataset_zoo(
            category=args.co3d_category,
            assert_single_seq=(args.co3d_task == "singlesequence"),
            dataset_name=f"co3d_{args.co3d_task}",
            test_on_train=False,
            load_point_clouds=False,
            test_restrict_sequence_id=args.co3d_single_sequence_id,
        )

    # CO3D Dataset loader
    dataloaders = dataloader_zoo(
            datasets,
            dataset_name=f"co3d_{args.co3d_task}",
            batch_size=(args.N_src + args.N_src_extra),
            # num_workers=1,
            dataset_len=1000,
            dataset_len_val=10,
            images_per_seq_options=[100],
        )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Nerformer model
    model = NerFormer(160, args)
    
    # model checkpoint
    checkpoint_callback = ModelCheckpoint(
                                        dirpath=args.out_dir,
                                        filename="model_{step}",
                                        every_n_train_steps=args.log_weight_step,
                                        )

    # training
    trainer = pl.Trainer(gpus=1, num_nodes=1, precision=32,
                        max_steps=args.n_iters, 
                        val_check_interval=1.0,
                        callbacks=[checkpoint_callback],
                        num_sanity_val_steps=0
                        )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    args = OmegaConf.load("config_file.yaml")
    print(args, "\n\n")

    train(args)