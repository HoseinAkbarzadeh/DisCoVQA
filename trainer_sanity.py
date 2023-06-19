from argparse import ArgumentParser
import os
import sys
import random

from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy as Fabric_DDP #, DeepSpeedStrategy
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy as Lightning_DDP #, DeepSpeedStrategy
import numpy as np
import toml
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.environ['WORKSPACE_DIR'])
# local application imports
from datasets.LIVE_NFLX_II.livenflxii_video import LiveNFLXIIVideoHDF5
from datasets.transforms import VideoRandomCrop
from models.lightning import DisCoVQALightning


def setup_cmd():
    parser = ArgumentParser()

    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--num_devices', type=int, nargs='*', required=False)
    parser.add_argument('--rng', type=int, default=42)
    parser.add_argument('--config', type=str, default='sanity.toml')
    parser.add_argument('--prefix', type=str, default='sanity')
    parser.add_argument('--nbatch', type=int, default=1)

    return parser.parse_args()

def reduce_ds(ds, size):
    out = []
    for idx, d in enumerate(ds):
        if idx == size:
            break
        else:
            out.append(d)
    return out

if __name__ == '__main__':
    args = setup_cmd()
    conf = toml.load(args.config)
    devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')] if args.num_devices is None else args.num_devices

    # setting up the environment
    fabric = Fabric(
        accelerator='gpu',
        strategy=Fabric_DDP(find_unused_parameters=False),
        devices=devices
    )
    fabric.launch()

    # making the training more deterministic and less variable
    random.seed(args.rng)
    np.random.seed(args.rng)
    torch.manual_seed(args.rng)
    torch.cuda.manual_seed_all(args.rng)
    seed_everything(args.rng, workers=True)
    torch.set_float32_matmul_precision('medium')

    # setting the checkpointing callback
    ckpt_callback = ModelCheckpoint(os.path.join(os.environ['PROJECT_DIR'] ,conf['map']['checkpoint_dir'], 
                                                 os.environ['SLURM_JOB_ID'], f"proc{fabric.device}"),
                                    conf['map']['checkpoint_filename'],
                                    monitor='val_loss',
                                    save_top_k=1,
                                    save_last=True,
                                    mode='min')
    
    # setting the tensorboard logger
    tb_logger = TensorBoardLogger(os.environ['PROJECT_DIR'],
                                  name= conf['map']['tb_logger_dir'],
                                  version=f"{args.prefix}_{os.environ['SLURM_JOB_ID']}",
                                  default_hp_metric=False)

    # creating the trainer
    trainer = Trainer(
        accelerator='gpu',
        strategy=Lightning_DDP(find_unused_parameters=False),
        num_nodes=args.num_nodes,
        devices=devices,
        max_epochs=conf['run']['epoch'],
        logger=tb_logger,
        callbacks=[ckpt_callback, RichProgressBar()],
        log_every_n_steps=1
    )

    # preparation of train and val datasets
    img_tr = [
        VideoRandomCrop(conf['run']['resolution'])
    ]

    lbl_tr = [
        (lambda x: x.float().contiguous())
    ]

    ds = LiveNFLXIIVideoHDF5.from_csv(
        csvfile=os.path.join(os.environ['PROJECT_DIR'], conf['dataset']['train_csv']), 
        hdf5file=os.path.join(os.environ['SCRATCH_DIR'], conf['dataset']['hdf5_file']), 
        metric='PMOS', 
        frame_transforms=img_tr, 
        tgt_transforms=lbl_tr, 
        max_len=conf['dataset']['train_max_len']
    )

    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(ds, [int(0.9*len(ds)), len(ds)-int(0.9*len(ds))], generator=gen)

    train_ds = reduce_ds(train_ds, args.nbatch*conf['run']['batch_size'])
    val_ds = reduce_ds(val_ds, args.nbatch*conf['run']['batch_size'])

    train_loader = DataLoader(train_ds, batch_size=conf['run']['batch_size'], shuffle=True, 
                              pin_memory=False, drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=conf['run']['batch_size'], shuffle=False,
                            pin_memory=False, drop_last=False, num_workers=args.num_workers)
    
    # creating the model
    model = DisCoVQALightning(conf['run']['batch_size'], conf['run']['learning_rate'],
                              conf['run']['resolution'], conf['run']['weight_decay'])
    

    # training the model
    print("Start Training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # finalizing the training
    print("Training is over.")
    print(f"""Best val loss: {ckpt_callback.best_model_score}
Best checkpoint path: {ckpt_callback.best_model_path}
Last checkpoint_path: {ckpt_callback.last_model_path}""")
