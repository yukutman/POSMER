import torch
import torch.nn as nn
from torch.nn import functional as F
from models.mobilefacenet import MobileFaceNet
from models.ir50 import Backbone
from timm.models.layers import DropPath
from thop import profile
from mamba_ssm import Mamba


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=14, patch_size=16, in_c=256, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(256, 768, kernel_size=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x: (B, H, W, C)
        window_size: window size

    Returns:
        local window features (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class window(nn.Module):
    def __init__(self, window_size, dim):
        super(window, self).__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.norm(x)
        shortcut = x
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        return x_windows, shortcut


def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    return x.permute(0, 3, 1, 2)


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_reverse(windows, window_size, H, W, h_w, w_w):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class feedforward(nn.Module):
    def __init__(self, dim, window_size, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0., layer_scale=None):
        super(feedforward, self).__init__()
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.window_size = window_size
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, attn_windows, shortcut):
        B, H, W, C = shortcut.shape
        h_w = int(torch.div(H, self.window_size).item())
        w_w = int(torch.div(W, self.window_size).item())
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm(x)))
        return x


class pyramid_trans_expr2(nn.Module):
    def __init__(self, img_size=224, num_classes=7, window_size=[28, 14, 7], num_heads=[2, 4, 8], dims=[64, 128, 256],
                 embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.num_heads = num_heads
        self.dim_head = []
        for num_head, dim in zip(num_heads, dims):
            self.dim_head.append(int(torch.div(dim, num_head).item()))
        self.num_classes = num_classes
        self.window_size = window_size
        self.N = [win * win for win in window_size]
        self.face_landback = MobileFaceNet([112, 112], 136)
        face_landback_checkpoint = torch.load(r'models/pretrain/mobilefacenet_model_best.pth.tar',
                                              map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        for param in self.face_landback.parameters():
            param.requires_grad = False

        self.BIM = BiMambaClassifier(embed_dim=embed_dim, num_classes=num_classes, depth=2)

        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load(r'models/pretrain/ir50.pth', map_location=lambda storage, loc: storage)

        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        # change with LandmarkGatedMamba instead of WindowAttentionGlobal
        self.attn1 = LandmarkGatedMamba(dim=dims[0])
        self.attn2 = LandmarkGatedMamba(dim=dims[1])
        self.attn3 = LandmarkGatedMamba(dim=dims[2])

        self.window1 = window(window_size=window_size[0], dim=dims[0])
        self.window2 = window(window_size=window_size[1], dim=dims[1])
        self.window3 = window(window_size=window_size[2], dim=dims[2])

        self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[2], kernel_size=3, stride=2, padding=1)

        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]

        self.ffn1 = feedforward(dim=dims[0], window_size=window_size[0], layer_scale=1e-5, drop_path=dpr[0])
        self.ffn2 = feedforward(dim=dims[1], window_size=window_size[1], layer_scale=1e-5, drop_path=dpr[1])
        self.ffn3 = feedforward(dim=dims[2], window_size=window_size[2], layer_scale=1e-5, drop_path=dpr[2])

        self.last_face_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.proj1 = nn.Sequential(nn.Conv2d(dims[0], 768, kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1))
        self.proj2 = nn.Sequential(nn.Conv2d(dims[1], 768, kernel_size=3, stride=2, padding=1))
        self.proj3 = PatchEmbed(img_size=14, patch_size=14, in_c=256, embed_dim=768)

    def forward(self, x):
        x_face = F.interpolate(x, size=112)
        x_face1, x_face2, x_face3 = self.face_landback(x_face)
        x_face3 = self.last_face_conv(x_face3)
        x_face1, x_face2, x_face3 = _to_channel_last(x_face1), _to_channel_last(x_face2), _to_channel_last(x_face3)

        x_ir1, x_ir2, x_ir3 = self.ir_back(x)

        x_ir1, x_ir2, x_ir3 = self.conv1(x_ir1), self.conv2(x_ir2), self.conv3(x_ir3)
        x_window1, shortcut1 = self.window1(x_ir1)
        x_window2, shortcut2 = self.window2(x_ir2)
        x_window3, shortcut3 = self.window3(x_ir3)

        o1, o2, o3 = self.attn1(x_window1, x_face1), self.attn2(x_window2, x_face2), self.attn3(x_window3, x_face3)

        o1, o2, o3 = self.ffn1(o1, shortcut1), self.ffn2(o2, shortcut2), self.ffn3(o3, shortcut3)

        o1, o2, o3 = _to_channel_first(o1), _to_channel_first(o2), _to_channel_first(o3)

        o1, o2, o3 = self.proj1(o1).flatten(2).transpose(1, 2), self.proj2(o2).flatten(2).transpose(1,
                                                                                                    2), self.proj3(
            o3)

        o = torch.cat([o1, o2, o3], dim=1)

        out = self.BIM(o)
        return out


class LandmarkGatedMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # 1. The Gating Mechanism (FiLM)
        # We project the landmark summary into 2 * dim (Alpha and Beta)
        self.film_generator = nn.Linear(dim, dim * 2)

        # Initialize alpha to 1 and beta to 0 (Identity transformation)
        # This helps stability at the start of training
        nn.init.constant_(self.film_generator.weight, 0)
        nn.init.constant_(self.film_generator.bias, 0)
        self.film_generator.bias.data[:dim] = 1  # Set alpha part to 1

        # 2. The Mamba Block (Replaces the Attention mixing)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()  # Simple activation for the gate

    def forward(self, x_img, x_lm):
        if x_lm.dim() == 4:  # to be safe from 4D input
            # Flatten spatial dims: (B, H, W, C) -> (B, H*W, C)
            x_lm = x_lm.reshape(x_lm.shape[0], -1, x_lm.shape[-1])

        B_windows, N, C = x_img.shape
        B_face, N_lm, C_lm = x_lm.shape

        # Calculate how many windows per image
        num_windows = B_windows // B_face

        # 1. Compute Summary (Batch, Dim)
        g = x_lm.mean(dim=1)

        # 2. Generate Alpha/Beta (Batch, Dim)
        film_params = self.film_generator(g)
        alpha, beta = torch.split(film_params, C, dim=1)

        # 3. Repeat for all windows
        # We need to stretch (Batch, Dim) -> (Batch * Num_Windows, 1, Dim)
        alpha = alpha.repeat_interleave(num_windows, dim=0).unsqueeze(1)
        beta = beta.repeat_interleave(num_windows, dim=0).unsqueeze(1)

        # 4. Modulate & Mamba
        x_modulated = (alpha * x_img) + beta
        x_out = self.mamba(self.norm(x_modulated))

        return x_out + x_img


class BiMambaClassifier(nn.Module):  # Bi-Directional Mamba Classifier for Head instead of ViT
    def __init__(self, embed_dim=768, num_classes=7, depth=2, d_state=16):
        super().__init__()

        # We use a ModuleList to stack multiple Bi-Mamba blocks
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                Mamba(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=4,
                    expand=2
                )
            )

        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Sequence_Len, Dim)

        for layer in self.layers:
            # 1. Forward Pass
            x_fwd = layer(x)

            # 2. Backward Pass
            # Flip the sequence along dimension 1 (time/sequence dim)
            x_rev = torch.flip(x, dims=[1])
            x_bwd = layer(x_rev)
            x_bwd = torch.flip(x_bwd, dims=[1])  # Flip back to original order

            # 3. Fusion (Add them up)
            # This creates a "Global Receptive Field"
            x = x_fwd + x_bwd

        # Global Average Pooling
        x = x.mean(dim=1)

        x = self.norm_f(x)
        x = self.head(x)
        return x


def compute_param_flop():
    model = pyramid_trans_expr2()
    img = torch.rand(size=(1, 3, 224, 224))
    flops, params = profile(model, inputs=(img,))
    print(f'flops:{flops / 1000 ** 3}G,params:{params / 1000 ** 2}M')
