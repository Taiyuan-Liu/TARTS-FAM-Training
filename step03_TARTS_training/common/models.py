"""TARTS network cores, adapted from PetchMa/TARTS (see README and LICENSE)."""

from pathlib import Path

import timm
import torch
from torch import nn
from torchvision import models as torchvision_models


class WaveNet(nn.Module):
    """Author-compatible CNN + metadata predictor; outputs CCS Z4--Z28 in um."""

    def __init__(self, cnn_model="resnet101", n_predictor_layers=(256, 205, 69), n_zernikes=25):
        super().__init__()
        self.hparams = dict(cnn_model=cnn_model, n_predictor_layers=list(n_predictor_layers),
                            n_zernikes=n_zernikes)
        # Match the author's timm-first backbone selection, without downloading weights.
        if cnn_model.startswith("mobilenetv4") or cnn_model in timm.list_models():
            self.cnn = timm.create_model(cnn_model, pretrained=False, num_classes=0)
            feature_count = self.cnn.num_features
        else:
            self.cnn = getattr(torchvision_models, cnn_model)(weights=None)
            feature_count = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        layers = []
        previous = feature_count + 4
        for index, width in enumerate(n_predictor_layers):
            layers.extend([nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU()])
            if index == 0:
                layers.append(nn.Dropout(0.2))
            previous = width
        layers.append(nn.Linear(previous, n_zernikes))
        self.predictor = nn.Sequential(*layers)
        self.predictor_features = None
        self.channels_last = False

    def set_runtime(self, channels_last=False):
        self.channels_last = bool(channels_last)
        self.cnn.to(memory_format=torch.channels_last if self.channels_last else torch.contiguous_format)
        return self

    def forward(self, image, fx, fy, focal, band):
        image = image[:, None].repeat_interleave(3, dim=1)
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        features = self.cnn(image)
        if features.ndim == 4:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        features = torch.cat([features, fx, fy, focal, band], dim=1)
        self.predictor_features = self.predictor[:-1](features)
        return self.predictor[-1](self.predictor_features)


class AggregatorNet(nn.Module):
    """Author transformer: unmasked zero padding, last token, additive mean."""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward,
                 max_seq_length=200, num_zernikes=25):
        super().__init__()
        self.hparams = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                            dim_feedforward=dim_feedforward,
                            max_seq_length=max_seq_length, num_zernikes=num_zernikes)
        self.input_proj = nn.Linear(num_zernikes + 3, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=dim_feedforward,
                                          dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_zernikes)
        self.transformer_features = None

    def forward(self, inputs, cache_features=True):
        sequence, mean = inputs
        features = self.transformer_encoder(self.input_proj(sequence))[:, -1]
        if cache_features:
            self.transformer_features = features.float()
        return self.fc(features) + mean


def build_model(kind, hparams):
    """Build a randomly initialized network. No pretrained weights are downloaded."""
    if kind == "wavenet":
        keys = ("cnn_model", "n_predictor_layers", "n_zernikes")
        model = WaveNet(**{key: hparams[key] for key in keys})
        if hparams["n_zernikes"] != 25:
            raise ValueError("This dataset requires 25 output coefficients")
    elif kind == "aggregator":
        keys = ("d_model", "nhead", "num_layers", "dim_feedforward",
                "max_seq_length", "num_zernikes")
        model = AggregatorNet(**{key: hparams[key] for key in keys})
        if hparams["num_zernikes"] != 25 or hparams["max_seq_length"] != 200:
            raise ValueError("Expected the 25-mode, 200-token Aggregator")
    else:
        raise ValueError(kind)
    return model


def load_model(path, kind, device="cpu"):
    """Load a trusted local model.pt or resume.pt produced by this step."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False, mmap=True)
    if (payload.get("model_kind"), payload.get("frame"), payload.get("unit"), payload.get("noll")) != (
            kind, "CCS", "micron", list(range(4, 29))):
        raise ValueError("Expected the requested model with physical CCS Z4--Z28 in microns")
    model = build_model(kind, payload["model_hparams"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if kind == "wavenet":
        model.set_runtime(payload["config"].get("channels_last", False))
    return model.to(device), payload
