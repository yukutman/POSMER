import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath, trunc_normal_

# --- 1. The Core Engine (Custom CUDA Mamba) ---
class Mambassm(nn.Module):
    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank else d_inner // 16
        
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_inner)

        # 1. Forward Scan
        out_forward = self._scan(x)

        # 2. Backward Scan (Flip sequence, scan, flip back)
        x_flip = torch.flip(x, dims=[1])
        out_backward = torch.flip(self._scan(x_flip), dims=[1])

        # 3. Combine (Average)
        return (out_forward + out_backward) / 2

    def _scan(self, x):
        """Helper to run the actual selective scan logic"""
        x_dbc = self.x_proj(x)
        (delta_rank, B, C) = x_dbc.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        u = x.transpose(1, 2)
        delta = self.dt_proj(delta_rank).transpose(1, 2)

        # Parameters
        A = -torch.exp(self.A_log.float())
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)
        D = self.D.float()
        delta_bias = self.dt_proj.bias.float()

        # Run Mamba Scan (Official Kernel)
        y = selective_scan_fn(
            u, delta, A, B, C, D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False
        )
        return y.transpose(1, 2)

# --- 2. The Mixer Block ---
class ViMBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, drop_path=0.):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            kernel_size=4, 
            groups=self.d_inner, 
            padding=3
        )
        self.act = nn.SiLU()
        self.mixer = Mambassm(d_inner=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :x.shape[2]]
        x = x.transpose(1, 2)
        
        x = self.act(x)
        x = self.mixer(x)
        x = x * self.act(res)
        x = self.out_proj(x)
        
        return residual + self.drop_path(x)

# --- 3. Helpers ---
class PatchEmbed(nn.Module):
    """ Used in PosterV2 to project the pyramid features """
    def __init__(self, img_size=14, patch_size=16, in_c=256, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=1) 
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# --- 4. The Backbone (VisionMamba) ---
class VisionMamba(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 depth=2, 
                 num_classes=7, 
                 d_state=16, 
                 expand=2, 
                 drop_path_rate=0.1,
                 **kwargs): 
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.Sequential(*[
            ViMBlock(
                dim=embed_dim, 
                d_state=d_state, 
                expand=expand, 
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))