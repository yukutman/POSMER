import torch
import torch.nn as nn
from torch.nn import functional as F

from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone

from .vim_model import PatchEmbed, ViMBlock, Mambassm


def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    model_dict = model.state_dict()
    new_state_dict = {}
    matched = 0

    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched += 1

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print(f'Loaded {matched} layers.')
    return model


# Helper Functions for Window Partitioning
def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): size of window
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    h_w = H // window_size
    w_w = W // window_size

    # Reshape to separate windows
    x = x.view(B, C, h_w, window_size, w_w, window_size)

    # Permute to group window parts together
    # (B, C, h_w, win, w_w, win) -> (B, h_w, w_w, win, win, C)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()

    # Merge batch and window counts, and flatten the window pixels
    # (B * num_windows, window_area, C)
    windows = x.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size*window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        C (int): Channels
    Returns:
        x: (B, C, H, W)
    """
    h_w = H // window_size
    w_w = W // window_size

    # Recover Batch size
    B = int(windows.shape[0] / (h_w * w_w))

    # Reshape back to grid
    x = windows.view(B, h_w, w_w, window_size, window_size, C)

    # Permute back to (B, C, H, W) order
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

    # Final View
    x = x.view(B, C, H, W)
    return x


class LandmarkGatedMamba(nn.Module):
    """
    Fuses Visual features and Landmark features using a Gated SSM.
    Uses YOUR custom Mambassm backend.
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        # FiLM Generator
        self.film_generator = nn.Linear(d_model, d_model * 2)

        # Initialize FiLM to identity
        nn.init.constant_(self.film_generator.weight, 0)
        nn.init.constant_(self.film_generator.bias, 0)
        self.film_generator.bias.data[:d_model] = 1

        # Use YOUR custom Mambassm from vim_model.py
        self.ssm = Mambassm(d_inner=d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_visual, x_landmark):
        # x_visual: [B_total, N, d_model]
        # x_landmark: [B_face, N, d_model]

        # Handle batch size mismatch due to windows
        num_windows = x_visual.shape[0] // x_landmark.shape[0]

        # 1. Global Landmark Context
        context = x_landmark.mean(dim=1)

        # 2. Generate Alpha/Beta
        film_params = self.film_generator(context)
        alpha, beta = torch.split(film_params, x_visual.shape[-1], dim=1)

        # 3. Broadcast to all windows
        alpha = alpha.repeat_interleave(num_windows, dim=0).unsqueeze(1)
        beta = beta.repeat_interleave(num_windows, dim=0).unsqueeze(1)

        # 4. Modulate & Scan
        x_modulated = (alpha * x_visual) + beta
        return x_visual + self.ssm(self.norm(x_modulated))


class BiMambaClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=7, depth=2, d_state=16):
        super().__init__()

        # We use a ModuleList to stack multiple Bi-Mamba blocks
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # [CHANGE] Use ViMBlock from your backend instead of high-level Mamba
            self.layers.append(
                ViMBlock(
                    dim=embed_dim,
                    d_state=d_state,
                    expand=2
                    # drop_path=0. # Optional: add drop path if needed
                )
            )

        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Sequence_Len, Dim)

        for layer in self.layers:
            # 1. Forward Pass
            # ViMBlock is unidirectional, so this scans Left->Right
            x_fwd = layer(x)

            # 2. Backward Pass
            # Flip the sequence along dimension 1 (time/sequence dim)
            x_rev = torch.flip(x, dims=[1])

            # Run the same ViMBlock on flipped data (Right->Left scan)
            x_bwd = layer(x_rev)

            # Flip back to original order
            x_bwd = torch.flip(x_bwd, dims=[1])

            # 3. Fusion (Add them up)
            # This creates a "Global Receptive Field"
            x = x_fwd + x_bwd

        # Global Average Pooling
        x = x.mean(dim=1)

        x = self.norm_f(x)
        x = self.head(x)
        return x


class pyramid_trans_expr2(nn.Module):
    def __init__(self, img_size=224, num_classes=7,
                 dims=[64, 128, 256], d_model=768,
                 window_sizes=[7, 7, 7],
                 d_state=16,
                 ir50_path='pretrain/ir50.pth',
                 facenet_path='pretrain/mobilefacenet_model_best.pth.tar'):
        super().__init__()

        self.window_sizes = window_sizes

        # --- A. Backbones ---
        self.face_backbone = MobileFaceNet([112, 112], 136)
        if facenet_path:
            self.face_backbone = load_pretrained_weights(self.face_backbone, facenet_path)
            for p in self.face_backbone.parameters(): p.requires_grad = False

        self.visual_backbone = Backbone(50, 0.0, 'ir')
        if ir50_path:
            self.visual_backbone = load_pretrained_weights(self.visual_backbone, ir50_path)

        # --- B. Pyramid Fusion Stages ---
        self.downsample1 = nn.Conv2d(dims[0], dims[0], 3, 2, 1)
        self.downsample2 = nn.Conv2d(dims[1], dims[1], 3, 2, 1)
        self.downsample3 = nn.Conv2d(dims[2], dims[2], 3, 2, 1)

        self.face_proj = nn.Conv2d(512, 256, 3, 1, 1)

        self.fusion_stage1 = LandmarkGatedMamba(d_model=dims[0])
        self.fusion_stage2 = LandmarkGatedMamba(d_model=dims[1])
        self.fusion_stage3 = LandmarkGatedMamba(d_model=dims[2])

        # --- C. Stage Projections ---
        self.proj_stage1 = nn.Sequential(
            nn.Conv2d(dims[0], d_model, 3, 2, 1),
            nn.Conv2d(d_model, d_model, 3, 2, 1)
        )
        self.proj_stage2 = nn.Sequential(
            nn.Conv2d(dims[1], d_model, 3, 2, 1)
        )
        self.proj_stage3 = PatchEmbed(in_c=256, embed_dim=d_model)

        # --- D. Backend ---
        # [CHANGE] Use the explicitly defined BiMambaClassifier
        self.vim_backend = BiMambaClassifier(embed_dim=d_model, depth=2, num_classes=num_classes, d_state=d_state)

    def forward(self, x):
        # 1. Geometry Features
        x_face = F.interpolate(x, size=112)
        with torch.no_grad():
            x_face1, x_face2, x_face3 = self.face_backbone(x_face)
        x_face3 = self.face_proj(x_face3)

        x_face1 = x_face1.flatten(2).transpose(1, 2)
        x_face2 = x_face2.flatten(2).transpose(1, 2)
        x_face3 = x_face3.flatten(2).transpose(1, 2)

        # 2. Visual Features
        x_vis1, x_vis2, x_vis3 = self.visual_backbone(x)
        x_vis1, x_vis2, x_vis3 = self.downsample1(x_vis1), self.downsample2(x_vis2), self.downsample3(x_vis3)

        # --- 3. Window Partition & Fusion  ---

        # Stage 1
        B, C1, H1, W1 = x_vis1.shape
        x_vis1_win = window_partition(x_vis1, self.window_sizes[0])
        fused1_win = self.fusion_stage1(x_vis1_win, x_face1)
        fused1 = window_reverse(fused1_win, self.window_sizes[0], H1, W1, C1)

        # Stage 2
        B, C2, H2, W2 = x_vis2.shape
        x_vis2_win = window_partition(x_vis2, self.window_sizes[1])
        fused2_win = self.fusion_stage2(x_vis2_win, x_face2)
        fused2 = window_reverse(fused2_win, self.window_sizes[1], H2, W2, C2)

        # Stage 3
        B, C3, H3, W3 = x_vis3.shape
        x_vis3_win = window_partition(x_vis3, self.window_sizes[2])
        fused3_win = self.fusion_stage3(x_vis3_win, x_face3)
        fused3 = window_reverse(fused3_win, self.window_sizes[2], H3, W3, C3)

        # 4. Project
        token1 = self.proj_stage1(fused1).flatten(2).transpose(1, 2)
        token2 = self.proj_stage2(fused2).flatten(2).transpose(1, 2)
        token3 = self.proj_stage3(fused3)

        # 5. Concatenate
        tokens = torch.cat([token1, token2, token3], dim=1)

        # 6. Run Bi-Directional Backend
        return self.vim_backend(tokens)