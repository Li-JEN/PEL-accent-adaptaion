# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

from geomloss import SamplesLoss
# from scipy.stats import wasserstein_distance
import ot


class wd_criterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
    
    def forward(self, t,c):
        if t.shape[0] > c.shape[0]:
            pad = torch.zeros(t.shape[0]-c.shape[0]).to(t.device)
            c = torch.cat((c,pad),dim=0)
        else:
            pad = torch.zeros(c.shape[0]-t.shape[0]).to(t.device)
            t = torch.cat((t,pad),dim=0)
        # print(t.shape, c.shape)
        # Wass_xy = self.Loss(t.unsqueeze(0),c.unsqueeze(0))
        M = ot.utils.dist(t.unsqueeze(0),c.unsqueeze(0), 'euclidean')
        alpha = torch.tensor(ot.unif(len(t.unsqueeze(0)))).to(t.device)
        beta = torch.tensor(ot.unif(len(c.unsqueeze(0)))).to(t.device)
        M2 = pow(M, 2)
        pW = ot.emd2(alpha, beta, M2, numItermax=100000)
        pW = pow(pW, 1/2)
        
        return pW

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        if source.shape[1] > target.shape[1]:
            pad = torch.zeros(source.shape[0],source.shape[1]-target.shape[1]).to(source.device)
            target = torch.cat((target,pad),dim=1)
        else:
            pad = torch.zeros(target.shape[0],target.shape[1]-source.shape[1]).to(source.device)
            source = torch.cat((source,pad),dim=1)
        # print(source.shape)
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        # print(kernels.shape)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) -  torch.mean(YX)
        # target = F.normalize(target, p=2, dim=1)
        # source = F.normalize(source, p=2, dim=1)

        # loss = self.gaussian_kernel(target, target).mean() \
        #         + self.gaussian_kernel(source, source).mean() \
        #         - 2 * self.gaussian_kernel(target, source).mean()
        return loss
class FastSpeech2Loss(torch.nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False, use_mmd=False, use_swd=False, use_l2=False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        if use_mmd:
            self.mmd_criterion = MMD_loss()
        elif use_swd:
            self.wd_criterion = wd_criterion()
        elif use_l2:
            self.l2_criterion = torch.nn.MSELoss(reduction=reduction)    
        
    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        pair: torch.Tensor = None,
        pair_lens:torch.Tensor  = None,
        latent: list  = None,
        use_adapter = False,
        use_mmd = False,
        use_swd = False,
        use_l2 = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)
            if pair is not None:
                pair_masks = make_non_pad_mask(pair_lens).unsqueeze(-1).to(ys.device)
                pair = pair.squeeze(2).transpose(0,1)
            
        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)
        wd_loss = 0
        if pair is not None and not use_adapter:
            if use_mmd:
                for idx in range(1,len(latent),2):
                    pair_ = pair[int((idx-1)/2)+1].masked_fill(mask=~pair_masks, value=0)
                    latent_ = latent[idx].masked_fill(~out_masks, value=0)
                    pair_ = pair_.view(pair_.size(0),-1)
                    latent_ = latent_.view(latent_.size(0),-1)
                    loss = self.mmd_criterion(pair_,latent_)
                    wd_loss += loss
                wd_loss *= 0.1
            elif use_swd:
                for idx in range(1,len(latent),2):
                    for id in range(latent[idx].shape[0]):
                        latent_ = latent[idx][id].masked_select(out_masks[id])
                        pair_ = pair[int((idx-1)/2)+1][id].masked_select(pair_masks[id])
                        loss = self.wd_criterion(latent_, pair_)
                        # loss = self.mmd_criterion(pair_,latent_)
                        wd_loss += loss 
                    wd_loss /= latent[idx].shape[0]
                wd_loss *= 0.001
            elif use_l2:
                for idx in range(1,len(latent),2):
                    pair_ = pair[int((idx-1)/2)+1].masked_fill(mask=~pair_masks, value=0)
                    latent_ = latent[idx].masked_fill(~out_masks, value=0)
                    # print(pair_.shape, latent_.shape)
                    if pair_.shape[1] > latent_.shape[1]:
                        pad = torch.zeros(pair_.shape[0],pair_.shape[1]-latent_.shape[1],pair_.shape[2]).to(latent_.device)
                        latent_= torch.cat((latent_,pad),dim=1)
                    else:
                        pad = torch.zeros(pair_.shape[0], latent_.shape[1]-pair_.shape[1],pair_.shape[2]).to(pair_.device)
                        pair_ = torch.cat((pair_,pad),dim=1)
                    wd_loss += self.l2_criterion(pair_,latent_)
                wd_loss *= 0.001
            else:
                AssertionError("Using additional loss, but not select loss type")
            wd_loss /= (len(latent)/2)
            # print(wd_loss)
        elif pair is not None and use_adapter:
            if use_mmd:
                for idx in range(1,len(latent),2):
                    pair_ = pair[int((idx-1)/2)+1].masked_fill(mask=~pair_masks, value=0)
                    latent_ = latent[idx].masked_fill(~out_masks, value=0)
                    pair_ = pair_.view(pair_.size(0),-1)
                    latent_ = latent_.view(latent_.size(0),-1)
                    loss = self.mmd_criterion(pair_,latent_)
                    wd_loss += loss
                wd_loss *= 0.1
            elif use_swd:
                for idx in range(1,len(latent),2):
                    for id in range(latent[idx].shape[0]):
                        latent_ = latent[idx][id].masked_select(out_masks[id])
                        pair_ = pair[int((idx-1)/2)+1][id].masked_select(pair_masks[id])
                        wd_loss += self.wd_criterion(latent_, pair_)
                    wd_loss /= latent[idx].shape[0]
                wd_loss /= (len(latent)/2)
                wd_loss *= 0.001
            elif use_l2:
                for idx in range(1,len(latent),2):
                    pair_ = pair[int((idx-1)/2)+1].masked_fill(mask=~pair_masks, value=0)
                    latent_ = latent[idx].masked_fill(~out_masks, value=0)
                    # print(pair_.shape, latent_.shape)
                    if pair_.shape[1] > latent_.shape[1]:
                        pad = torch.zeros(pair_.shape[0],pair_.shape[1]-latent_.shape[1],pair_.shape[2]).to(latent_.device)
                        latent_= torch.cat((latent_,pad),dim=1)
                    else:
                        pad = torch.zeros(pair_.shape[0], latent_.shape[1]-pair_.shape[1],pair_.shape[2]).to(pair_.device)
                        pair_ = torch.cat((pair_,pad),dim=1)
                    wd_loss += self.l2_criterion(pair_,latent_)
                wd_loss *= 0.001
            else:
                AssertionError("Using additional loss, but not select loss type")
            
        
        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )
        if pair is not None:
            return l1_loss, duration_loss, pitch_loss, energy_loss, wd_loss
        else:
            return l1_loss, duration_loss, pitch_loss, energy_loss