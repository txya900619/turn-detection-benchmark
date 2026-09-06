from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from moshi.models import loaders as moshi_loaders
from torch import nn


class MimiLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel_embed = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        options = {
            "input_size": 128,
            "hidden_size": 512,
            "num_layers": 2,
            "batch_first": True,
            "dropout": 0.1,
            "bidirectional": False,
        }
        self.model1 = nn.LSTM(**options)
        self.model2 = nn.LSTM(**options)
        self.linear = nn.Linear(1_024, 5)

    def infer(self, features: torch.Tensor) -> torch.Tensor:
        values = self.mel_embed(features.permute(0, 1, 3, 2))
        hidden = torch.zeros(2, values.shape[0], 512, device=values.device)
        cell = torch.zeros_like(hidden)
        first, _ = self.model1(values[:, 0], (hidden, cell))
        second, _ = self.model2(values[:, 1], (hidden, cell))
        return self.linear(torch.cat((first, second), dim=-1))


def load_endpointer(path: Path, device: str) -> MimiLSTM:
    model = MimiLSTM()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


class MimiFeatureExtractor:
    def __init__(self, weights: Path, device: str) -> None:
        self.model = moshi_loaders.get_mimi(str(weights), device=device, num_codebooks=8).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, waveform: np.ndarray) -> torch.Tensor:
        tensor = (
            torch.from_numpy(np.asarray(waveform, dtype=np.float32))
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        latent = self.model.encode_to_latent(tensor, quantize=True)
        return self.model.upsample(latent)
