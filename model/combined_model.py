import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from STAnomalyFormer.model.patch import Patch, PatchEncoder, RandomMasking
from STAnomalyFormer.model.tsfm import TemporalTransformer

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

class CustomTemporalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        n_heads: int,
        dim_fc: int = 128,
        dropout: float = 0.1,
        half: bool = False,
        return_attn: bool = False,
    ) -> None:
        super().__init__()
        self.half_ = half
        self.d_model = d_model
        self.attn = TemporalTransformer(
            d_model=d_model,
            dim_k=dim_k,
            dim_v=dim_v,
            n_heads=n_heads,
            dim_fc=dim_fc,
            dropout=dropout,
            half=half,
            return_attn=return_attn
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=dim_fc,
            kernel_size=1,
        )
        if not half:
            self.conv2 = nn.Conv1d(
                in_channels=dim_fc,
                out_channels=d_model,
                kernel_size=1,
            )
            self.norm2 = nn.LayerNorm(d_model)
        self.return_attn = return_attn

    def forward(self, x):
        x_, attn = self.attn(x)
        y = x = self.norm1(self.dropout(x + x_))
        y = self.dropout(torch.relu(self.conv1(y.transpose(-1, 1))))
        if self.half_:
            if self.return_attn:
                return y, attn
            return y
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)
        if self.return_attn:
            return output, attn
        return output

class CombinedAnomalyDetector(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        c_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.4,
        shared_embedding: bool = True,
        pe: str = 'zeros',
        learn_pe: bool = True,
        temperature: float = 50.0,
        anormly_ratio: float = 0.1,
    ):
        super().__init__()
        
        # Patch mechanism from STAnomalyFormer
        self.patch = Patch(seq_len, patch_len, stride)
        self.patch_encoder = PatchEncoder(
            c_in=c_in,
            num_patch=self.patch.num_patch,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            shared_embedding=shared_embedding,
            dropout=dropout,
            pe=pe,
            learn_pe=learn_pe
        )
        self.random_masking = RandomMasking(mask_ratio)
        
        # Custom temporal attention blocks
        self.encoder = nn.ModuleList([
            CustomTemporalAttention(
                d_model=d_model,
                dim_k=d_model // n_heads,
                dim_v=d_model // n_heads,
                n_heads=n_heads,
                dim_fc=d_ff,
                dropout=dropout,
                half=False,
                return_attn=True
            ) for _ in range(e_layers)
        ])
        
        self.decoder = nn.ModuleList([
            CustomTemporalAttention(
                d_model=d_model,
                dim_k=d_model // n_heads,
                dim_v=d_model // n_heads,
                n_heads=n_heads,
                dim_fc=d_ff,
                dropout=dropout,
                half=False,
                return_attn=True
            ) for _ in range(e_layers)
        ])
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, c_in)
        
        self.temperature = temperature
        self.anormly_ratio = anormly_ratio
        
    def forward(self, x):
        # x: [B, T, C]
        
        # Apply patch mechanism
        x_patched = self.patch(x)  # [B, num_patch, C, patch_len]
        x_masked = self.random_masking(x_patched)
        x_encoded = self.patch_encoder(x_masked)  # [B, C, num_patch, d_model]
        
        # Reshape for temporal attention
        B, C, num_patch, d_model = x_encoded.shape
        x_encoded = x_encoded.permute(0, 2, 1, 3).reshape(B * num_patch, C, d_model)
        
        # Encoding
        encoder_attn_list = []
        for encoder_layer in self.encoder:
            x_encoded, attn = encoder_layer(x_encoded)
            encoder_attn_list.append(attn)
        
        # Decoding
        decoder_attn_list = []
        for decoder_layer in self.decoder:
            x_decoded, attn = decoder_layer(x_encoded)
            decoder_attn_list.append(attn)
        
        # Projection
        x_projected = self.projection(x_decoded)
        
        # Reshape back
        x_projected = x_projected.reshape(B, num_patch, C, d_model)
        x_projected = x_projected.permute(0, 2, 1, 3)
        
        # Output
        output = self.output_layer(x_projected)
        
        # Calculate anomaly scores
        adv_loss = 0.0
        con_loss = 0.0
        
        for i in range(len(encoder_attn_list)):
            if i == 0:
                adv_loss = my_kl_loss(encoder_attn_list[i], 
                    (decoder_attn_list[i] / torch.unsqueeze(torch.sum(decoder_attn_list[i], dim=-1), dim=-1)).detach()) * self.temperature
                con_loss = my_kl_loss(
                    (decoder_attn_list[i] / torch.unsqueeze(torch.sum(decoder_attn_list[i], dim=-1), dim=-1)),
                    encoder_attn_list[i].detach()) * self.temperature
            else:
                adv_loss += my_kl_loss(encoder_attn_list[i], 
                    (decoder_attn_list[i] / torch.unsqueeze(torch.sum(decoder_attn_list[i], dim=-1), dim=-1)).detach()) * self.temperature
                con_loss += my_kl_loss(
                    (decoder_attn_list[i] / torch.unsqueeze(torch.sum(decoder_attn_list[i], dim=-1), dim=-1)),
                    encoder_attn_list[i].detach()) * self.temperature
        
        # Calculate anomaly scores
        metric = torch.softmax((adv_loss + con_loss), dim=-1)
        anomaly_scores = metric.detach().cpu().numpy()
        
        # Calculate threshold
        thresh = np.percentile(anomaly_scores, 100 - self.anormly_ratio)
        
        # Detect anomalies
        pred = (anomaly_scores > thresh).astype(int)
        
        # Get anomaly regions and timestamps
        anomaly_regions = []
        anomaly_timestamps = []
        
        anomaly_state = False
        for i in range(len(pred)):
            if pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                start_idx = i
            elif pred[i] == 0 and anomaly_state:
                anomaly_state = False
                end_idx = i - 1
                anomaly_regions.append((start_idx, end_idx))
                # Convert indices to timestamps
                start_time = start_idx * self.stride
                end_time = end_idx * self.stride + self.patch_len
                anomaly_timestamps.append((start_time, end_time))
        
        if anomaly_state:
            end_idx = len(pred) - 1
            anomaly_regions.append((start_idx, end_idx))
            start_time = start_idx * self.stride
            end_time = end_idx * self.stride + self.patch_len
            anomaly_timestamps.append((start_time, end_time))
        
        return {
            'output': output,
            'anomaly_scores': anomaly_scores,
            'anomaly_regions': anomaly_regions,
            'anomaly_timestamps': anomaly_timestamps,
            'threshold': thresh
        }

    def get_loss(self, output_dict):
        """Calculate the loss for training"""
        adv_loss = 0.0
        con_loss = 0.0
        
        for i in range(len(output_dict['encoder_attn'])):
            adv_loss += (torch.mean(my_kl_loss(output_dict['encoder_attn'][i], 
                (output_dict['decoder_attn'][i] / torch.unsqueeze(torch.sum(output_dict['decoder_attn'][i], dim=-1), dim=-1)).detach())) + 
                torch.mean(my_kl_loss((output_dict['decoder_attn'][i] / torch.unsqueeze(torch.sum(output_dict['decoder_attn'][i], dim=-1), dim=-1)).detach(),
                output_dict['encoder_attn'][i])))
            
            con_loss += (torch.mean(my_kl_loss(
                (output_dict['decoder_attn'][i] / torch.unsqueeze(torch.sum(output_dict['decoder_attn'][i], dim=-1), dim=-1)),
                output_dict['encoder_attn'][i].detach())) + 
                torch.mean(my_kl_loss(output_dict['encoder_attn'][i].detach(),
                (output_dict['decoder_attn'][i] / torch.unsqueeze(torch.sum(output_dict['decoder_attn'][i], dim=-1), dim=-1)))))
        
        adv_loss = adv_loss / len(output_dict['encoder_attn'])
        con_loss = con_loss / len(output_dict['encoder_attn'])
        
        return con_loss - adv_loss 