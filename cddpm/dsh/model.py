"""Conditional U-Net: predicts the diffusion noise and, optionally, the conditions."""
import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionalUNet(nn.Module):
    """U-Net with (32, 64, 128) channels, conditioned by concatenating an
    embedding of (c1, c2) to the input as extra channels.

    With predict_labels=True a small MLP predicts (c1, c2) back from the
    pooled bottleneck; this is the label loss used during training.
    The layers are created in the same order as in run 377, so a given seed
    gives the same initial weights.
    """

    def __init__(self, image_size=48, n_conditions=2, embed_channels=32,
                 label_hidden=128, predict_labels=True):
        super().__init__()
        self.predict_labels = predict_labels

        self.condition_embedding = nn.Sequential(
            nn.Linear(n_conditions, embed_channels), nn.ReLU(inplace=True),
            nn.Linear(embed_channels, embed_channels), nn.ReLU(inplace=True),
        )
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=1 + embed_channels,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D",) * 3,
            up_block_types=("UpBlock2D",) * 3,
        )

        if predict_labels:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.label_head = nn.Sequential(
                nn.Linear(self.unet.config.block_out_channels[-1], label_hidden), nn.ReLU(inplace=True),
                nn.Linear(label_hidden, n_conditions),
            )
            self._bottleneck = None
            self.unet.mid_block.register_forward_hook(self._store_bottleneck)

    def _store_bottleneck(self, module, inputs, output):
        self._bottleneck = output

    def forward(self, x, timesteps, conditions):
        """Returns {"noise": predicted noise} plus {"labels": predicted conditions}."""
        embedding = self.condition_embedding(conditions)
        embedding = embedding[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        out = {"noise": self.unet(torch.cat([x, embedding], dim=1), timesteps).sample}
        if self.predict_labels:
            out["labels"] = self.label_head(self.pool(self._bottleneck).flatten(1))
        return out
