import math
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch.core.module import LightningModule
from omegaconf import OmegaConf
from torchmetrics.classification import BinaryFBetaScore

import sys
sys.path.append("../submodules/pytorch-3dunet/")
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.partialconv3d import PartialConv3d
from pytorch3dunet.unet3d.partialconv2d import PartialConv2d

class Unet3D(LightningModule):
    def __init__(
        self,
        cfg,
        steps_per_epoch: int,
    ):
        super().__init__()

        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch

        cfg_encoder = {
            "name": cfg.model.encoder.name,
            # number of input channels to the model
            "in_channels": cfg.model.encoder.in_channels,
            # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
            "layer_order": cfg.model.encoder.layer_order,
            # number of features at each level of the U-Net
            "f_maps": OmegaConf.to_object(cfg.model.encoder.f_maps),
            # number of groups in the groupnorm
            "num_groups": cfg.model.encoder.num_groups,
            # number of pool operators at each level
            "pool_kernel_size": OmegaConf.to_object(cfg.model.encoder.pool_kernel_size),
        }
        cfg_decoder = {
            "name": cfg.model.decoder.name,
            # number of output channels
            "out_channels": cfg.model.decoder.out_channels,
            # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
            "layer_order": cfg.model.decoder.layer_order,
            # number of features at each level of the U-Net
            "f_maps": OmegaConf.to_object(cfg.model.decoder.f_maps),
            # number of groups in the groupnorm
            "num_groups": cfg.model.decoder.num_groups,
            # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
            "is_segmentation": cfg.model.decoder.is_segmentation,
            "upsampling_conv_kernel_size": OmegaConf.to_object(cfg.model.decoder.upsampling_conv_kernel_size),
            "scale_factor": OmegaConf.to_object(cfg.model.decoder.scale_factor),
            "upsampling_padding": OmegaConf.to_object(cfg.model.decoder.upsampling_padding),
        }

        self.encoder = get_model(cfg_encoder)
        #---
        # self-attention
        encoder_dim = OmegaConf.to_object(cfg.model.encoder.f_maps)
        self.attention_weight = nn.ModuleList([
            nn.Sequential(
                PartialConv3d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ) for dim in encoder_dim[::-1]
        ])
        # self-attention
        #---
        self.decoder = get_model(cfg_decoder)
        self.global_pool_map = nn.AdaptiveAvgPool3d(output_size=(1, None, None))

        self.final_conv = nn.Conv3d(self.decoder.out_channels, 1, kernel_size=1)

        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss(reduction='none', smooth_factor=cfg.label_smooth_factor)
        self.dice_fn = smp.losses.DiceLoss(mode="binary")
        self.loss_alpha = cfg.loss_alpha

        self.train_dice = BinaryFBetaScore(beta=0.5)
        self.val_dice = BinaryFBetaScore(beta=0.5)

    def _process_instance(self, x, batch_size, num_instances):

        _, encoders_features = self.encoder(x)
        #---
        # self-attention
        for i in range(len(encoders_features)):
            e = encoders_features[i]
            f = self.attention_weight[i](e)

            # (BN)CDHW -> NBCDHW
            BN, c, d, h, w = e.shape
            assert BN == batch_size * num_instances
            f = f.view(batch_size, num_instances, c, d, h, w)
            e = e.view(batch_size, num_instances, c, d, h, w)
            w = F.softmax(f, 1)
            e = (w * e).sum(1)
            encoders_features[i] = e
        # self-attention
        #---
        x = self.decoder(encoders_features[0], encoders_features)
        x = self.final_conv(x)

        return x

    def forward(self, batch: torch.Tensor):

        # BNCDHW -> (BN)CDHW
        bs, N, c, d, h, w = batch.shape
        batch = batch.view(bs*N, c, d, h, w)

        # Perform inference for each instance
        x = self._process_instance(batch, bs, N)

        y = self.global_pool_map(x).squeeze(dim=2)

        return y

    def setup_adamw_optimizer(self, learning_rate, weight_decay=1e-2):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_parameters, lr=learning_rate)

    def configure_optimizers(self):

        device_count = torch.cuda.device_count()

        optimizers = [
            self.setup_adamw_optimizer(
                learning_rate=self.cfg.model.lr * math.sqrt(device_count),
            ),
        ]

        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizers[0],
                    max_lr=self.cfg.model.lr * math.sqrt(device_count),
                    total_steps=self.steps_per_epoch * self.cfg.model.num_epochs,
                ),
                "interval": "step",
            }
        ]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):

        input = batch["input"]
        target = batch["target"]
        ignore_mask = batch["ignore_mask"]

        output = self(input)

        loss_bce = torch.mean(self.loss_fn(output, target) * ignore_mask)
        loss_dice = self.dice_fn(output * ignore_mask, target * ignore_mask)
        loss = (1 - self.loss_alpha) * loss_bce + self.loss_alpha * loss_dice

        self.train_dice(output * ignore_mask, target * ignore_mask)

        self.log_dict(
            {
                "train_loss": loss,
                "train_dice": self.train_dice,
            },
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        input = batch["input"]
        target = batch["target"]
        ignore_mask = batch["ignore_mask"]

        output = self(input)

        loss_bce = torch.mean(self.loss_fn(output, target) * ignore_mask)
        loss_dice = self.dice_fn(output * ignore_mask, target * ignore_mask)
        loss = (1 - self.loss_alpha) * loss_bce + self.loss_alpha * loss_dice

        self.val_dice(output * ignore_mask, target * ignore_mask)

        self.log_dict(
            {
                "val_loss": loss,
                "val_dice": self.val_dice,
            },
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )

class SegFormer(LightningModule):
    def __init__(
        self,
        cfg,
        steps_per_epoch: int,
    ):
        super().__init__()

        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch

        self.encoder = smp.Unet(
            encoder_name=cfg.model.encoder.backbone,
            encoder_weights="imagenet",
            in_channels=cfg.model.encoder.in_channels,
            classes=cfg.model.encoder.out_channels,
            activation=None,
        )
        #---
        # self-attention
        self.attention_weight = nn.Sequential(
            PartialConv2d(cfg.model.encoder.out_channels, cfg.model.encoder.out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # self-attention
        #---
        self.final_conv = nn.Conv2d(cfg.model.encoder.out_channels, 1, kernel_size=1)

        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss(reduction='none', smooth_factor=cfg.label_smooth_factor)
        self.dice_fn = smp.losses.DiceLoss(mode="binary")
        self.loss_alpha = cfg.loss_alpha

        self.train_dice = BinaryFBetaScore(beta=0.5)
        self.val_dice = BinaryFBetaScore(beta=0.5)

    def _process_instance(self, x, batch_size, num_instances):

        x = self.encoder(x)
        #---
        # self-attention
        e = x
        f = self.attention_weight(e)
        # (BN)CHW -> NBCHW
        BN, c, h, w = e.shape
        assert BN == batch_size * num_instances
        f = f.view(batch_size, num_instances, c, h, w)
        e = e.view(batch_size, num_instances, c, h, w)
        w = F.softmax(f, 1)
        e = (w * e).sum(1)
        x = e
        # self-attention
        #---
        x = self.final_conv(x)

        return x

    def forward(self, batch: torch.Tensor):

        # BNCHW -> (BN)CHW
        bs, N, c, h, w = batch.shape
        batch = batch.view(bs*N, c, h, w)

        # Perform inference for each instance
        y = self._process_instance(batch, bs, N)

        return y

    def setup_adamw_optimizer(self, learning_rate, weight_decay=1e-2):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_parameters, lr=learning_rate)

    def configure_optimizers(self):

        device_count = torch.cuda.device_count()

        optimizers = [
            self.setup_adamw_optimizer(
                learning_rate=self.cfg.model.lr * math.sqrt(device_count),
            ),
        ]

        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizers[0],
                    max_lr=self.cfg.model.lr * math.sqrt(device_count),
                    total_steps=self.steps_per_epoch * self.cfg.model.num_epochs,
                ),
                "interval": "step",
            }
        ]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):

        input = batch["input"]
        target = batch["target"]
        ignore_mask = batch["ignore_mask"]

        output = self(input)

        loss_bce = torch.mean(self.loss_fn(output, target) * ignore_mask)
        loss_dice = self.dice_fn(output * ignore_mask, target * ignore_mask)
        loss = (1 - self.loss_alpha) * loss_bce + self.loss_alpha * loss_dice

        self.train_dice(output * ignore_mask, target * ignore_mask)

        self.log_dict(
            {
                "train_loss": loss,
                "train_dice": self.train_dice,
            },
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        input = batch["input"]
        target = batch["target"]
        ignore_mask = batch["ignore_mask"]

        output = self(input)

        loss_bce = torch.mean(self.loss_fn(output, target) * ignore_mask)
        loss_dice = self.dice_fn(output * ignore_mask, target * ignore_mask)
        loss = (1 - self.loss_alpha) * loss_bce + self.loss_alpha * loss_dice

        self.val_dice(output * ignore_mask, target * ignore_mask)

        self.log_dict(
            {
                "val_loss": loss,
                "val_dice": self.val_dice,
            },
            on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )
