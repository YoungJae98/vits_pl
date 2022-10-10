import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import pytorch_lightning as pl
from vits.models.model import (
    SynthesizerTrn, 
    MultiPeriodDiscriminator
)
from vits.utils import utils
from vits.utils.utils import fix_seeds
from vits.models.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from vits.utils.data_utils import (
    TextAudioLoader,
    TextAudioCollate,
    DistributedBucketSampler
)
from vits.utils.mel_processing import (
    mel_spectrogram_torch, 
    spec_to_mel_torch
    )
from vits.text.symbols import symbols
from vits.models import commons

class VitsTrainer(pl.LightningModule):
    def __init__(
        self,
        hparams
    ):
        super(VitsTrainer, self).__init__()
        fix_seeds(seed=hparams.train.seed)
        self.hps = hparams
        self.batch_size = self.hps.train.batch_size
        self.num_workers = 4
        self.n_gpus = 1
        
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
            )
        self.net_d = MultiPeriodDiscriminator(self.hps.model.use_spectral_norm)

    def forward(self, batch):
        return self.net_g(batch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch
        y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths)
        
        mel = spec_to_mel_torch(
            spec, 
            self.hps.data.filter_length, 
            self.hps.data.n_mel_channels, 
            self.hps.data.sampling_rate,
            self.hps.data.mel_fmin, 
            self.hps.data.mel_fmax
            )
        y_mel = commons.slice_segments(mel, ids_slice, self.hps.train.segment_size // self.hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 
            self.hps.data.filter_length, 
            self.hps.data.n_mel_channels, 
            self.hps.data.sampling_rate, 
            self.hps.data.hop_length, 
            self.hps.data.win_length, 
            self.hps.data.mel_fmin, 
            self.hps.data.mel_fmax
            )
        y = commons.slice_segments(y, ids_slice * self.hps.data.hop_length, self.hps.train.segment_size)
        
        
        
        if optimizer_idx == 0:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hps.train.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            output = {'loss': loss_gen_all}
            
            self.log("loss_gen_all", loss_gen_all, sync_dist=True, prog_bar=True)
        if optimizer_idx == 1:
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)            
            output = {'loss': loss_disc}
            
            self.log("loss_disc", loss_disc, sync_dist=True, prog_bar=True)
            
        return output


    def validation_step(self, batch, batch_idx):
        x, x_lengths, spec, spec_lengths, y, y_lengths = batch
        
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        
        y_hat, attn, mask, _ = self.net_g.infer(x, x_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * self.hps.data.hop_length
        
        mel = spec_to_mel_torch(
            spec, 
            self.hps.data.filter_length, 
            self.hps.data.n_mel_channels, 
            self.hps.data.sampling_rate,
            self.hps.data.mel_fmin, 
            self.hps.data.mel_fmax
            )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 
            self.hps.data.filter_length, 
            self.hps.data.n_mel_channels, 
            self.hps.data.sampling_rate, 
            self.hps.data.hop_length, 
            self.hps.data.win_length, 
            self.hps.data.mel_fmin, 
            self.hps.data.mel_fmax
            )
        
        image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {
        "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
        }
        if self.global_step == 0:
            image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
            audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})
        
        return ""



    def configure_optimizers(self):
        lr = self.hps.train.learning_rate
        b = self.hps.train.betas
        eps = self.hps.train.eps
        lr_decay = self.hps.train.lr_decay
        
        optim_g = torch.optim.AdamW(self.net_g.parameters(), lr=lr, betas=b, eps=eps)
        optim_d = torch.optim.AdamW(self.net_d.parameters(), lr=lr, betas=b, eps=eps)
        
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay)
        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    # dataset:
    # dataloader
    def train_dataloader(self):
        trainset = TextAudioLoader(self.hps.data.training_files, self.hps.data)
        collate_fn = TextAudioCollate()
        train_sampler = DistributedBucketSampler(
                                trainset,
                                self.hps.train.batch_size,
                                [32,300,400,500,600,700,800,900,1000],
                                num_replicas=self.n_gpus,
                                rank=self.global_rank,
                                shuffle=True
                                )
        train_loader = DataLoader(trainset, 
                                num_workers=self.num_workers, 
                                shuffle=False,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                batch_sampler=train_sampler
                                )
        return train_loader

    def val_dataloader(self):
        valset = TextAudioLoader(self.hps.data.validation_files, self.hps.data)
        collate_fn = TextAudioCollate()
        val_loader = DataLoader(valset,
                                num_workers=self.num_workers, 
                                shuffle=False,
                                batch_size=self.hps.train.batch_size, 
                                pin_memory=True,
                                drop_last=False, 
                                collate_fn=collate_fn
                                )

        return val_loader
