"""
Video+audio deepfake model — Xception (encoder) + Wav2Vec2-large-xlsr audio
branch, fused for a real/fake verdict. Fully separate from the image
DeepfakeNet model in app.py — nothing here touches that model.
"""
import os
os.environ["TORCH_HOME"] = "C:/torch_cache"

import torch
import torch.nn as nn
import timm
from transformers import Wav2Vec2Model

VIDEO_BACKBONE = "xception"
AUDIO_BACKBONE = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"


class VideoBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = timm.create_model(VIDEO_BACKBONE, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1).mean(dim=1)
        return self.proj(feats)


class AudioBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(AUDIO_BACKBONE, low_cpu_mem_usage=False)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, wav):
        wav = (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-6)
        out = self.encoder(wav).last_hidden_state
        feats = out.mean(dim=1)
        return self.proj(feats)


class FusionModel(nn.Module):
    def __init__(self, video_dim=256, audio_dim=128, num_classes=2):
        super().__init__()
        self.video_branch = VideoBranch(out_dim=video_dim)
        self.audio_branch = AudioBranch(out_dim=audio_dim)
        self.video_head = nn.Linear(video_dim, num_classes)
        self.audio_head = nn.Linear(audio_dim, num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(video_dim + audio_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, frames, wav):
        v_feat = self.video_branch(frames)
        a_feat = self.audio_branch(wav)
        v_logits = self.video_head(v_feat)
        a_logits = self.audio_head(a_feat)
        fused = torch.cat([v_feat, a_feat], dim=1)
        f_logits = self.fusion_head(fused)
        return f_logits, v_logits, a_logits
