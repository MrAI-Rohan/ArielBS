"""
Attentive ASPP for DeepLabV3+
Based on: "Modified DeepLabV3+ with multi-level context attention mechanism
           for colonoscopy polyp segmentation" (Gangrade et al., 2024)

Two attention mechanisms:
  1. ChannelAttention  — SE-Net style, applied after each of the 5 ASPP branches
  2. DepthwiseScaleAggregation — replaces the project (concat→1280→256) layer,
                                  learns 5 scalar weights (one per branch) and
                                  aggregates channel-independently (depthwise)

Usage:
    model = smp.DeepLabV3Plus(encoder_name="resnet101", ...)
    replace_aspp_with_attentive(model)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# 1. Channel Attention (SE-Net style)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    For a branch output of shape [B, C, H, W]:
      - Global Average Pool  → [B, C, 1, 1]
      - FC (C → C//reduction) → ReLU
      - FC (C//reduction → C) → Sigmoid   → attention weights [B, C, 1, 1]
      - Multiply with input   → [B, C, H, W]

    Args:
        channels:  number of channels in the branch output (256 for smp ASPP)
        reduction: bottleneck ratio for the FC layers (default 16)
    """
    def __init__(self, channels: int = 256, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)          # [B, C, H, W] → [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Flatten(),                            # [B, C, 1, 1] → [B, C]
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),                            # weights in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        weights = self.gap(x)                        # [B, C, 1, 1]
        weights = self.fc(weights)                   # [B, C]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * weights                           # [B, C, H, W]


# ---------------------------------------------------------------------------
# 2. Depthwise Scale Aggregation
# ---------------------------------------------------------------------------

class DepthwiseScaleAggregation(nn.Module):
    """
    Replaces the ASPP project layer (concat 1280→256).

    Given 5 branch outputs each of shape [B, C, H, W]:
      - Learns 5 scalar weights (one per branch), shared across all C channels
      - Weighted sum across branches, independently per channel (depthwise)
      - Sigmoid activation on the result
      - Output: [B, C, H, W]  — same shape the decoder expects

    This corresponds to Equation 2 in the paper:
        Ac = σ( Σ_l  Ws_l * Yc_l )

    Args:
        num_branches: number of ASPP branches (5 for standard DeepLabV3+)
        channels:     channels per branch (256 for smp ASPP)
    """
    def __init__(self, num_branches: int = 5, channels: int = 256):
        super().__init__()
        # 5 learnable scalar weights, one per branch
        # initialised to 1/num_branches so early training is balanced
        self.scale_weights = nn.Parameter(
            torch.full((num_branches,), 1.0 / num_branches)
        )
        self.num_branches = num_branches

    def forward(self, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            branch_outputs: one stacked tensor [B, num_branches, C, H, W]
        Returns:
            aggregated:     [B, C, H, W]
        """

        # scale_weights: [num_branches] → [1, num_branches, 1, 1, 1]  (broadcast)
        w = self.scale_weights.view(1, self.num_branches, 1, 1, 1)

        # weighted sum across branches → [B, C, H, W]
        aggregated = (branch_outputs * w).sum(dim=1)

        return torch.sigmoid(aggregated)             # σ as in Eq. 2


# ---------------------------------------------------------------------------
# 3. Attentive ASPP  (drop-in replacement for smp's ASPP)
# ---------------------------------------------------------------------------

class AttentiveASPP(nn.Module):
    """
    Drop-in replacement for smp's ASPP block.

    Keeps the 5 convolutional branches from the original smp ASPP unchanged,
    then adds:
      - ChannelAttention after each branch output
      - DepthwiseScaleAggregation instead of the concat + project layer

    Args:
        original_aspp: the smp ASPP instance taken from model.decoder.aspp[0]
        channels:      branch output channels (256)
        ca_reduction:  reduction ratio for ChannelAttention FC bottleneck
    """
    def __init__(
        self,
        original_aspp: nn.Module,
        channels: int = 256,
        ca_reduction: int = 16,
    ):
        super().__init__()

        # keep the 5 branches exactly as smp built them
        self.convs = original_aspp.convs   # ModuleList of 5 branches

        num_branches = len(self.convs)     # 5

        # one ChannelAttention per branch
        self.channel_attentions = nn.ModuleList(
            [ChannelAttention(channels, ca_reduction) for _ in range(num_branches)]
        )

        # Depthwise aggregation
        self.project = nn.Sequential(
            DepthwiseScaleAggregation(num_branches, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2048, H, W]  (last encoder feature map)

        attended = []
        for conv, ca in zip(self.convs, self.channel_attentions):
            branch_out = conv(x)            # [B, 256, H, W]
            branch_out = ca(branch_out)     # channel attention → [B, 256, H, W]
            attended.append(branch_out)
        
        # stack → [B, num_branches, C, H, W]
        stacked = torch.stack(attended, dim=1)

        # depthwise scale aggregation → [B, 256, H, W]
        out = self.project(stacked)
        return out


# ---------------------------------------------------------------------------
# 4. Helper: swap smp ASPP in-place
# ---------------------------------------------------------------------------

def replace_aspp_with_attentive(
    model: nn.Module,
    channels: int = 256,
    ca_reduction: int = 16,
    ) -> nn.Module:
    """
    Replaces the ASPP block inside an smp.DeepLabV3Plus model with AttentiveASPP.

    smp's decoder.aspp is a Sequential:
        (0): ASPP          ← replaced
        (1): SeparableConv2d
        (2): BatchNorm2d
        (3): ReLU

    Only index 0 is swapped; the rest of the decoder is untouched.

    Args:
        model:        smp.DeepLabV3Plus instance
        channels:     ASPP branch output channels (256)
        ca_reduction: ChannelAttention bottleneck reduction ratio

    Returns:
        model with AttentiveASPP in place (modified in-place and returned)
    """
    original_aspp = model.decoder.aspp[0]   # smp ASPP instance
    attentive_aspp = AttentiveASPP(
        original_aspp=original_aspp,
        channels=channels,
        ca_reduction=ca_reduction,
    )
    model.decoder.aspp[0] = attentive_aspp
    return model
