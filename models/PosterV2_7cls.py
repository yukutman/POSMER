import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_

# --- Imports from your project ---
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
# IMPORT your modular Mamba classes
from .vim_model import VisionMamba, PatchEmbed, Mambassm


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


class LandmarkGatedMamba(nn.Module):
    """
    Fuses Visual features and Landmark features using a Gated SSM.
    Terminology: Mamba Fusion (instead of Cross-Attention).
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        # FiLM Generator: Predicts modulation parameters
        self.film_generator = nn.Linear(d_model, d_model * 2)

        # Initialize FiLM to identity (neutral start)
        nn.init.constant_(self.film_generator.weight, 0)
        nn.init.constant_(self.film_generator.bias, 0)
        self.film_generator.bias.data[:d_model] = 1

        # The Mamba Engine
        self.ssm = Mambassm(d_inner=d_model, d_state=d_state)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_visual, x_landmark):
        # x_visual: [B_total, N, d_model]
        # x_landmark: [B_face, N, d_model]

        # Handle batch size mismatch (if using windows)
        num_windows = x_visual.shape[0] // x_landmark.shape[0]

        # 1. Global Landmark Context
        context = x_landmark.mean(dim=1)

        # 2. Generate Alpha/Beta (Gate Parameters)
        film_params = self.film_generator(context)
        alpha, beta = torch.split(film_params, x_visual.shape[-1], dim=1)

        # 3. Broadcast to match visual features
        alpha = alpha.repeat_interleave(num_windows, dim=0).unsqueeze(1)
        beta = beta.repeat_interleave(num_windows, dim=0).unsqueeze(1)

        # 4. Modulate (Gating) & Scan (SSM)
        x_modulated = (alpha * x_visual) + beta
        return x_visual + self.ssm(self.norm(x_modulated))


class pyramid_trans_expr2(nn.Module):
    def __init__(self, img_size=224, num_classes=7,
                 dims=[64, 128, 256], d_model=768,  # Renamed embed_dim -> d_model
                 ir50_path='models/pretrain/ir50.pth',
                 facenet_path='models/pretrain/mobilefacenet_model_best.pth.tar'):
        super().__init__()

        # --- A. Feature Extractors (Backbones) ---
        self.face_backbone = MobileFaceNet([112, 112], 136)
        if facenet_path:
            self.face_backbone = load_pretrained_weights(self.face_backbone, facenet_path)
            for p in self.face_backbone.parameters(): p.requires_grad = False

        self.visual_backbone = Backbone(50, 0.0, 'ir')
        if ir50_path:
            self.visual_backbone = load_pretrained_weights(self.visual_backbone, ir50_path)

        # --- B. Pyramid Fusion Stages ---
        # Downsampling convolutions
        self.downsample1 = nn.Conv2d(dims[0], dims[0], 3, 2, 1)
        self.downsample2 = nn.Conv2d(dims[1], dims[1], 3, 2, 1)
        self.downsample3 = nn.Conv2d(dims[2], dims[2], 3, 2, 1)

        # Face feature projection
        self.face_proj = nn.Conv2d(512, 256, 3, 1, 1)

        # Mamba Fusion Layers (Replaces Attention)
        self.fusion_stage1 = LandmarkGatedMamba(d_model=dims[0])
        self.fusion_stage2 = LandmarkGatedMamba(d_model=dims[1])
        self.fusion_stage3 = LandmarkGatedMamba(d_model=dims[2])

        # --- C. Stage Projections (Replaces Q/K/V) ---
        # Projects each pyramid stage to the final d_model size
        self.proj_stage1 = nn.Sequential(
            nn.Conv2d(dims[0], d_model, 3, 2, 1),
            nn.Conv2d(d_model, d_model, 3, 2, 1)
        )
        self.proj_stage2 = nn.Sequential(
            nn.Conv2d(dims[1], d_model, 3, 2, 1)
        )
        # Using PatchEmbed here as a generic 1x1 conv projection helper
        self.proj_stage3 = PatchEmbed(in_c=256, embed_dim=d_model)

        # --- D. Vision Mamba Backend ---
        # This replaces the VisionTransformer
        self.vim_backend = VisionMamba(embed_dim=d_model, depth=2, num_classes=num_classes)

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        # 1. Extract Geometry Features (FaceNet)
        x_face = F.interpolate(x, size=112)
        with torch.no_grad():
            x_face1, x_face2, x_face3 = self.face_backbone(x_face)
        x_face3 = self.face_proj(x_face3)

        # Flatten: [B, C, H, W] -> [B, L, C]
        x_face1 = x_face1.flatten(2).transpose(1, 2)
        x_face2 = x_face2.flatten(2).transpose(1, 2)
        x_face3 = x_face3.flatten(2).transpose(1, 2)

        # 2. Extract Visual Features (IR50)
        x_vis1, x_vis2, x_vis3 = self.visual_backbone(x)
        x_vis1, x_vis2, x_vis3 = self.downsample1(x_vis1), self.downsample2(x_vis2), self.downsample3(x_vis3)

        # Flatten Visual Features
        x_vis1_flat = x_vis1.flatten(2).transpose(1, 2)
        x_vis2_flat = x_vis2.flatten(2).transpose(1, 2)
        x_vis3_flat = x_vis3.flatten(2).transpose(1, 2)

        # 3. Mamba Fusion
        # Fuse visual texture with landmark geometry
        fused1 = self.fusion_stage1(x_vis1_flat, x_face1)
        fused2 = self.fusion_stage2(x_vis2_flat, x_face2)
        fused3 = self.fusion_stage3(x_vis3_flat, x_face3)

        # 4. Project to Common Dimension (d_model)
        B = x.shape[0]
        # Reshape back to 2D for Conv projections
        fused1 = fused1.transpose(1, 2).view(B, -1, x_vis1.shape[2], x_vis1.shape[3])
        fused2 = fused2.transpose(1, 2).view(B, -1, x_vis2.shape[2], x_vis2.shape[3])
        fused3 = fused3.transpose(1, 2).view(B, -1, x_vis3.shape[2], x_vis3.shape[3])

        # Project stages
        token1 = self.proj_stage1(fused1).flatten(2).transpose(1, 2)
        token2 = self.proj_stage2(fused2).flatten(2).transpose(1, 2)
        token3 = self.proj_stage3(fused3)

        # 5. Concatenate & Run ViM
        tokens = torch.cat([token1, token2, token3], dim=1)

        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)

        # Vision Mamba Forward Pass
        return self.vim_backend(tokens)