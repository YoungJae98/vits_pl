import os
import torch
from vits.trainer import VitsTrainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from vits.utils import utils

if __name__ == '__main__':
        checkpoint_callback = ModelCheckpoint(
                dirpath='checkpoints/batch_64/',
                filename='vits_{epoch}',
                save_weights_only = True, 
                every_n_train_steps=200,
                )
        
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '55553'
    
        
        hps = utils.get_hparams()
                
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        wandb_logger = WandbLogger(project="vits_pl")
        
        pl_model = VitsTrainer(hps)
        devices = torch.cuda.device_count()
        trainer = pl.Trainer(
                accelerator='gpu', 
                devices=devices, 
                logger = wandb_logger, 
                max_epochs=1000, 
                val_check_interval=50,
                callbacks=[checkpoint_callback]
                )
        trainer.fit(pl_model)
