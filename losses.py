import torch
import torch.nn as nn

class STELoss(nn.Module):
    def __init__(self, frame_length: int, overlap: float = 0.75):
        super().__init__()
        self.frame_length = int(frame_length)
        self.overlap = float(overlap)

        if self.frame_length < 1:
            raise ValueError("frame_length must be >= 1")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError("overlap must satisfy 0 <= overlap < 1")

        hop = int(round(self.frame_length * (1.0 - self.overlap)))
        self.hop_length = max(1, hop)

    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Shape mismatch: {output.shape} vs {target.shape}")

        if output.ndim == 2:
            output = output.unsqueeze(0)   # (1, C, T)
            target = target.unsqueeze(0)
        elif output.ndim != 3:
            raise ValueError(f"Expected (C,T) or (B,C,T), got {output.shape}")

        output_frames = output.unfold(dimension=-1, size=self.frame_length, step=self.hop_length)
        target_frames = target.unfold(dimension=-1, size=self.frame_length, step=self.hop_length)

        output_energy = (output_frames ** 2).sum(dim=-1)   # (B, C, M)
        target_energy = (target_frames ** 2).sum(dim=-1)   # (B, C, M)

        loss = (target_energy - output_energy).abs().mean() / self.frame_length
        return loss


class MSTELoss(nn.Module):
    def __init__(self, frame_lengths=(8, 16, 32, 64), overlap: float = 0.75):
        super().__init__()
        self.frame_lengths = tuple(int(n) for n in frame_lengths)
        self.overlap = float(overlap)

        self.losses = nn.ModuleList([
            STELoss(frame_length=n, overlap=self.overlap)
            for n in self.frame_lengths
        ])

    def forward(self, output, target):
        total = 0.0
        for loss_fn in self.losses:
            total = total + loss_fn(output, target)
        return total / len(self.losses)