
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_functions import *
from encoder_decoder import *


@dataclass
class JEPABitsConfig:
    # Encoder/Decoder parameters
    img_channels: int = 1
    out_dim_cnn: int = 11
    hidden_dim_fc: int = 96
    latent_bits: int = 64
    intermediate_channels: tuple = (8, 16, 32, 16)
    temperature: float = 1.0
    noise_magnitude: float = 0.0                        # for target branch
    decoder: bool = False
    
    # Predictor parameters
    action_dim: int = 4
    hidden_dim_pred: int = 128
    hidden_depth_pred: int = 1

    # Dataset/Task
    dataset_name: str = "8game"
    ice_slider: bool = False 
    
    # JEPA training parameters
    hinge_std: float = 0.4                             # for variance regularization
    max_bits_change: int = 6                            # for locality regularization
    min_bits_change: int = 1
    ema_decay: float = 0.99        
    
    jepa_loss: str = "cross-entropy"                    # "mse" or "cross-entropy" or "l1"
    normalize_cov: bool = True
    use_straight_through: bool = True 
    update_target_branch: bool = False
    both_branches: bool = False
    
    # Loss weights
    lambda_cov: float = 1.0
    lambda_pred: float = 1.0
    lambda_var: float = 1.0
    lambda_third: float = 1.0
    lambda_loc: float = 1.0
    lambda_dec: float = 0.0
    lambda_kl: float = 0.0

 

class Predictor(nn.Module):
    """
    MLP predictor that takes as input the concatenation of bits and action
    """
    def __init__(self, bits_dim: int, action_dim: int, hidden: int = 128, depth: int = 1):
        super().__init__()
        self.action_dim = action_dim
        layers_list = [nn.Linear(bits_dim + action_dim, hidden), nn.ReLU()]
        for _ in range(depth):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(nn.ReLU())
        layers_list.append(nn.Linear(hidden, bits_dim))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = nn.functional.one_hot(action, num_classes=self.action_dim).float() #one-hot encode actions
        x = torch.cat([x, action], dim=-1)
        return self.net(x)


class ResidualBlock(nn.Module):
    """Standard (3x3 conv + BN + ReLU) residual block"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn(self.conv(x)))
        return F.relu(x + h)


class ConvolutionalPredictor(nn.Module):
    """
    Convolutional predictor that takes as input the concatenation of bits and action
    """
    def __init__(
        self,
        grid_dim: int,
        latent_channels: int,
        action_dim: int,
        hidden_channels: int = None,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.grid_dim = grid_dim
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        hidden_channels = hidden_channels if hidden_channels is not None else latent_channels+action_dim
        in_ch = latent_channels + action_dim
        self.conv_in = nn.Conv2d(in_ch, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)])
        self.conv_out = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def _tile_action(self, action: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        action: (B,) integer actions
        returns: (B,A,H,W)
        """
        if action.dim() == 1:
            action_oh = F.one_hot(action.long(), num_classes=self.action_dim).float()
        else:
            raise ValueError("action must be shape (B,) or (B,A)")
        return action_oh[:, :, None, None].expand(-1, -1, h, w)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], self.latent_channels, self.grid_dim, self.grid_dim)
        a_map = self._tile_action(action, self.grid_dim, self.grid_dim)
        x = torch.cat([z, a_map], dim=1)  # (B, C+A, H, W)
        x = F.relu(self.bn(self.conv_in(x)))
        x = self.res_blocks(x)
        return self.conv_out(x).flatten(1)

    


class EMA(nn.Module):
    """
    Exponential moving average wrapper that maintains a copy of `model`
    and exposes a callable interface for forward passes with the EMA weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.99, device: Optional[torch.device] = None):
        super().__init__()
        # keep a live reference to the online model
        self.online_model = model
        self.decay = decay

        # make a deep copy for the EMA weights (no grads)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        if device is not None:
            self.ema_model.to(device)

        # initialize EMA weights to online weights
        self._copy_params_and_buffers(src=self.online_model, dst=self.ema_model, decay=None)

    @torch.no_grad()
    def _copy_params_and_buffers(self, src: nn.Module, dst: nn.Module, decay: Optional[float]):
        """
        Copy or EMA-update all state (parameters and buffers).
        If decay is None, it copies exactly. Otherwise it does: dst = decay*dst + (1-decay)*src
        """
        for dst_param, src_param in zip(dst.parameters(), src.parameters()):
            if decay is None:
                dst_param.data.copy_(src_param.data)
            else:
                dst_param.data.mul_(decay).add_(src_param.data, alpha=1.0 - decay)

        for dst_buf, src_buf in zip(dst.buffers(), src.buffers()):
            dst_buf.data.copy_(src_buf.data)
                

    @torch.no_grad()
    def update(self):
        """EMA update: ema = decay * ema + (1 - decay) * online (for params), copy buffers."""
        self._copy_params_and_buffers(src=self.online_model, dst=self.ema_model, decay=self.decay)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Forward through the EMA copy."""
        self.ema_model.eval()
        return self.ema_model(*args, **kwargs)




class JEPABits(nn.Module):

    def __init__(self, cfg: JEPABitsConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.ice_slider:
            self.encoder = BitEncoderIceSlider(cfg.img_channels, cfg.intermediate_channels, cfg.out_dim_cnn, cfg.hidden_dim_fc, cfg.latent_bits, \
                                                cfg.temperature, cfg.noise_magnitude)
            self.predictor = ConvolutionalPredictor(cfg.out_dim_cnn, cfg.intermediate_channels[-1], cfg.action_dim)
        else:
            self.encoder = BitEncoder(cfg.img_channels, cfg.intermediate_channels, cfg.out_dim_cnn, cfg.hidden_dim_fc, cfg.latent_bits, \
                                        cfg.temperature, cfg.noise_magnitude)
            self.predictor = Predictor(cfg.latent_bits, cfg.action_dim, cfg.hidden_dim_pred, cfg.hidden_depth_pred)
        self.encoder_ema = None  # for EMA of the encoder

        if cfg.decoder and cfg.ice_slider:
            self.decoder = BitDecoderIceslider(cfg.img_channels, cfg.intermediate_channels, cfg.out_dim_cnn, cfg.hidden_dim_fc, cfg.latent_bits)
        elif cfg.decoder:
            self.decoder = BitDecoder(cfg.img_channels, cfg.intermediate_channels, cfg.out_dim_cnn, cfg.hidden_dim_fc, cfg.latent_bits)
    
    
    def forward(self, x_ctx, action_ctx, x_tgt, compute_losses = True):
        # Context branch
        logits_ctx, probs_ctx, bits_ctx = self.encoder(x_ctx, sample=False)
        
        # Prediction branch
        bits_ctx = bits_ctx if compute_losses else bits_ctx.detach()
        z_ctx = bits_ctx if self.cfg.use_straight_through else probs_ctx
        pred = self.predictor(z_ctx, action_ctx)
        
        # Target branch
        if not self.cfg.both_branches:
            with torch.no_grad():
                if self.encoder_ema is None:
                    self.encoder_ema = EMA(self.encoder, self.cfg.ema_decay, device=x_tgt.device)
                else:
                    self.encoder_ema.update()
                logits_tgt, probs_tgt, bits_tgt = self.encoder_ema(x_tgt, sample=False)
        else:
            logits_tgt, probs_tgt, bits_tgt = self.encoder(x_tgt, sample=False)
        z_tgt = bits_tgt if self.cfg.use_straight_through else probs_tgt

        assert torch.isfinite(logits_ctx).any(), "NaNs/Infs in logits_ctx"
        assert torch.isfinite(logits_tgt).any(), "NaNs/Infs in logits_tgt"

        if not compute_losses:
            return pred, bits_tgt.detach()

        # JEPA loss (predict context -> target embedding)
        loss_pred = prediction_loss(pred, bits_tgt.detach(), self.cfg.jepa_loss)
        
        # locality loss
        if self.cfg.lambda_loc > 0.0:
            loss_loc = hamming_loss(probs_ctx, bits_tgt.detach(), (self.cfg.min_bits_change, self.cfg.max_bits_change))
        else:
            loss_loc = torch.tensor(0.0)
        
        # Bit regularizers losses
        regs = var_cov_cos_regularizers(probs_ctx, self.cfg.hinge_std, self.cfg.lambda_third>0., self.cfg.normalize_cov)
        loss_var = regs[0]
        loss_cov = regs[1]
        loss_third = regs[2]
        
        if self.cfg.update_target_branch:
            loss_pred2 = prediction_loss(z_tgt, (pred > 0.).float().detach(), self.cfg.jepa_loss)
            loss_pred = (loss_pred + loss_pred2) / 2
        
        if self.cfg.both_branches:
            regs_tgt = var_cov_cos_regularizers(probs_tgt, self.cfg.hinge_std, self.cfg.lambda_third>0., self.cfg.normalize_cov)
            loss_var = (loss_var + regs_tgt[0]) / 2
            loss_cov = (loss_cov + regs_tgt[1]) / 2
            loss_third = (loss_third + regs_tgt[2]) / 2

        # Decoder
        if self.cfg.decoder:
            if self.cfg.lambda_kl > 0.0:
                _, probs, _ = self.encoder(x_ctx, sample=True)
                x_rec = self.decoder(probs)
                if self.cfg.both_branches:
                    _, probs, _ = self.encoder(x_tgt, sample=True)
                    x_rec_tgt = self.decoder(probs)
            else:
                x_rec = self.decoder(z_ctx)
                if self.cfg.both_branches:
                    x_rec_tgt = self.decoder(z_tgt)
            loss_dec = F.mse_loss(x_rec, x_ctx)
            if self.cfg.both_branches:
                loss_dec = (loss_dec + F.mse_loss(x_rec_tgt, x_tgt)) / 2
        else:
            loss_dec = torch.tensor(0.0)

        # KL divergence (analytic for Bernoulli)
        if self.cfg.lambda_kl > 0.0:
            loss_kl = kl_divergence(probs_ctx)
            if self.cfg.both_branches:
                loss_kl = (loss_kl + kl_divergence(probs_tgt)) / 2
        else:
            loss_kl = torch.tensor(0.0)

        total = self.cfg.lambda_pred * loss_pred \
                + self.cfg.lambda_cov * loss_cov \
                + self.cfg.lambda_var * loss_var \
                + self.cfg.lambda_loc * loss_loc \
                + self.cfg.lambda_third * loss_third \
                + self.cfg.lambda_dec * loss_dec \
                + self.cfg.lambda_kl * loss_kl
        
        details = {
            "l_pred": loss_pred,
            "l_cov": loss_cov,
            "l_var": loss_var,
            "l_loc": loss_loc,
            "l_third": loss_third,
            "l_dec": loss_dec,
            "l_kl": loss_kl,
            "l_total": loss_pred + loss_var + loss_cov,
            "l_weighted": total,
            "p_mean": probs_ctx.mean(),
            "p_std": probs_ctx.std(dim=0).mean(),
            'b_mean': bits_ctx.mean(),
            'b_std': bits_ctx.std(dim=0).mean(),
        }
        
        return total, details, bits_tgt, probs_ctx, bits_ctx 
