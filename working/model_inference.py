import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../submodules/pytorch-3dunet/")
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.partialconv3d import PartialConv3d
from pytorch3dunet.unet3d.partialconv2d import PartialConv2d

CFG_ENCODER_UNet3D = {
    "name": "Encoder3D",
    # number of input channels to the model
    "in_channels": 1,
    # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
    "layer_order": "gcr",
    # number of features at each level of the U-Net
    "f_maps": [32, 64, 128, 256],
    # number of groups in the groupnorm
    "num_groups": 8,
    # number of pool operators at each level
    "pool_kernel_size": [1, 2, 2],
}
CFG_DECODER_UNet3D = {
    "name": "Decoder3D",
    # number of output channels
    "out_channels": 32,
    # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
    "layer_order": "gcr",
    # number of features at each level of the U-Net
    "f_maps": [32, 64, 128, 256],
    # number of groups in the groupnorm
    "num_groups": 8,
    # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
    "is_segmentation": False,
}

class UNet3D(torch.nn.Module):
    def __init__(
        self,
        cfg_encoder=CFG_ENCODER_UNet3D,
        cfg_decoder=CFG_DECODER_UNet3D,
    ):
        super().__init__()

        self.encoder = get_model(cfg_encoder)
        #---
        # self-attention
        encoder_dim = cfg_encoder["f_maps"]
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

        # B(CHW) -> BCHW
        y = y.view(h, w)

        y = torch.sigmoid(y)

        return y

CFG_ENCODER_ResidualUNet3D = {
    "name": "ResidualEncoder3D",
    # number of input channels to the model
    "in_channels": 1,
    # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
    "layer_order": "gcr",
    # number of features at each level of the U-Net
    "f_maps": [32, 64, 128, 256],
    # number of groups in the groupnorm
    "num_groups": 8,
    # number of pool operators at each level
    "pool_kernel_size": [1, 2, 2],
}
CFG_DECODER_ResidualUNet3D = {
    "name": "ResidualDecoder3D",
    # number of output channels
    "out_channels": 32,
    # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
    "layer_order": "gcr",
    # number of features at each level of the U-Net
    "f_maps": [32, 64, 128, 256],
    # number of groups in the groupnorm
    "num_groups": 8,
    # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
    "is_segmentation": False,
    "upsampling_conv_kernel_size": [1, 3, 3],
    "scale_factor": (1, 2, 2),
    "upsampling_padding": [0, 1, 1],
}

class ResidualUNet3D(torch.nn.Module):
    def __init__(
        self,
        cfg_encoder=CFG_ENCODER_ResidualUNet3D,
        cfg_decoder=CFG_DECODER_ResidualUNet3D,
    ):
        super().__init__()

        self.encoder = get_model(cfg_encoder)
        #---
        # self-attention
        encoder_dim = cfg_encoder["f_maps"]
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

        # B(CHW) -> BCHW
        y = y.view(h, w)

        y = torch.sigmoid(y)

        return y

CFG_ENCODER_ResidualUNetSE3D = {
    "name": "ResidualEncoderSE3D",
    # number of input channels to the model
    "in_channels": 1,
    # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
    "layer_order": "gcr",
    # number of features at each level of the U-Net
    "f_maps": [32, 64, 128, 256],
    # number of groups in the groupnorm
    "num_groups": 8,
    # number of pool operators at each level
    "pool_kernel_size": [1, 2, 2],
}
CFG_DECODER_ResidualUNetSE3D = {
    "name": "ResidualDecoderSE3D",
    # number of output channels
    "out_channels": 32,
    # determines the order of operators in a single layer (crg - Conv3d+ReLU+GroupNorm)
    "layer_order": "gcr",
    # number of features at each level of the U-Net
    "f_maps": [32, 64, 128, 256],
    # number of groups in the groupnorm
    "num_groups": 8,
    # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
    "is_segmentation": False,
    "upsampling_conv_kernel_size": [1, 3, 3],
    "scale_factor": (1, 2, 2),
    "upsampling_padding": [0, 1, 1],
}

class ResidualUNetSE3D(torch.nn.Module):
    def __init__(
        self,
        cfg_encoder=CFG_ENCODER_ResidualUNetSE3D,
        cfg_decoder=CFG_DECODER_ResidualUNetSE3D,
    ):
        super().__init__()

        self.encoder = get_model(cfg_encoder)
        #---
        # self-attention
        encoder_dim = cfg_encoder["f_maps"]
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

        # B(CHW) -> BCHW
        y = y.view(h, w)

        y = torch.sigmoid(y)

        return y

CFG_ENCODER_SegFormer = {
    "backbone": "mit_b2",
    "in_channels": 3,
    "out_channels": 256,
}

class SegFormer(torch.nn.Module):
    def __init__(
        self,
        cfg_encoder=CFG_ENCODER_SegFormer,
    ):
        super().__init__()

        self.encoder = smp.Unet(
            encoder_name=cfg_encoder["backbone"],
            encoder_weights=None,
            in_channels=cfg_encoder["in_channels"],
            classes=cfg_encoder["out_channels"],
            activation=None,
        )
        #---
        # self-attention
        self.attention_weight = nn.Sequential(
            PartialConv2d(cfg_encoder["out_channels"], cfg_encoder["out_channels"], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # self-attention
        #---
        self.final_conv = nn.Conv2d(cfg_encoder["out_channels"], 1, kernel_size=1)

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

        # B(CHW) -> BCHW
        y = y.view(h, w)

        y = torch.sigmoid(y)

        return y
