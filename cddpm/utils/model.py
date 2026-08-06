# utils/model_big.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

class ConditionalUnet(nn.Module):
    """
    Scaled-up (≈2–2.5×) Conditional U-Net that predicts:
      (1) noise for diffusion,
      (2) the condition vector from the bottleneck.
    """
    def __init__(
        self,
        sample_size,
        in_channels,
        out_channels,
        num_conditions,
        condition_embed_dim,
        predict_labels=True
    ):
        super().__init__()
        self.num_conditions = num_conditions
        self.predict_labels = predict_labels

        # Scale up condition embedding width
        embed_dim = int(condition_embed_dim * 2)

        self.condition_embedding = nn.Sequential(
            nn.Linear(num_conditions, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # --- Main U-Net (wider channels) ---
        self.model = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels + embed_dim,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),      # 2× width
            down_block_types=("DownBlock2D",) * 3,
            up_block_types=("UpBlock2D",) * 3,
        )

        # --- Label head (bottleneck → condition prediction) ---
        if self.predict_labels:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self._label_mlp = None
            self._bottleneck = None

            def _mid_hook(module, inputs, output):
                self._bottleneck = output
            self.model.mid_block.register_forward_hook(_mid_hook)

    def _ensure_label_head(self, C_mid: int):
        """Build label MLP lazily once mid channels known."""
        if self._label_mlp is None:
            self._label_mlp = nn.Sequential(
                nn.Linear(C_mid, max(C_mid, 128)),
                nn.ReLU(inplace=True),
                nn.Linear(max(C_mid, 128), self.num_conditions),
            )
            self._label_mlp.to(
                next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )

    def forward(self, x, timesteps, conditions):
        cond_emb = self.condition_embedding(conditions)  # (B, E)
        B, _, H, W = x.shape
        cond_map = cond_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        noise_pred = self.model(torch.cat([x, cond_map], dim=1), timesteps).sample
        out = {"noise": noise_pred}

        if self.predict_labels:
            mid = self._bottleneck
            if mid is None:
                raise RuntimeError("Mid-block hook did not capture bottleneck features.")
            self._ensure_label_head(mid.shape[1])
            pooled = self.gap(mid).flatten(1)
            label_pred = self._label_mlp(pooled)
            out.update({"labels": label_pred, "bottleneck": mid})

        return out