import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis
from thop import profile
from thop import clever_format


class Spatial_Enhance_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None, size=None):
        """Implementation of SAEM: Spatial Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block if not specifed reduced to half
        """
        super(Spatial_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=size * size,
                out_channels=1,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """

        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)
        t1 = t1.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*HW*TF --> B*TF*HW
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*HW
        Affinity_M = Affinity_M.view(batch_size, 1, x1.size(2), x1.size(3))   # B*1*H*W

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1


class Spectral_Enhance_Module(nn.Module):
    def __init__(self, in_channels, in_channels2, inter_channels=None, inter_channels2=None):
        """Implementation of SEEM: Spectral Enhancement Module
        args:
            in_channels: original channel size
            inter_channels: channel size inside the block
        """
        super(Spectral_Enhance_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.in_channels2 = in_channels2
        self.inter_channels2 = inter_channels2

        if self.inter_channels is None:
            self.inter_channels = in_channels
            if self.inter_channels == 0:
                self.inter_channels = 1
        if self.inter_channels2 is None:
            self.inter_channels2 = in_channels2
            if self.inter_channels2 == 0:
                self.inter_channels2 = 1

        # dimension == 2
        conv_nd = nn.Conv2d
        # max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        # define Transformation 1 and 2
        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels2, out_channels=self.inter_channels2, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            bn(self.inter_channels2),
            nn.Sigmoid()
        )

        self.dim_reduce = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels2,
                out_channels=1,
                kernel_size=1,
                bias=False,
            )
        )

    def forward(self, x1, x2):
        """
        args
            x: (N, C, H, W)
        """
        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)
        t2 = self.T2(x2).view(batch_size, self.inter_channels2, -1)
        t2 = t2.permute(0, 2, 1)
        Affinity_M = torch.matmul(t1, t2)

        Affinity_M = Affinity_M.permute(0, 2, 1)  # B*C1*C2 --> B*C2*C1
        Affinity_M = self.dim_reduce(Affinity_M)  # B*1*C1
        Affinity_M = Affinity_M.view(batch_size, x1.size(1), 1, 1)  # B*C1*1*1

        x1 = x1 * Affinity_M.expand_as(x1)

        return x1

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, fusion_matrix=None, **kwargs):

            return self.fn(self.norm(x), **kwargs)

class PreNorm_Matrix(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm_Matrix, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, fusion_matrix=None, **kwargs):
        if fusion_matrix is not None:
            return self.fn(self.norm(x), fusion_matrix, **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.fusion_matrix = FusionMatrixGenerator(dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class AttentionFusionMatrix(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(AttentionFusionMatrix, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.fusion_matrix = FusionMatrixGenerator(dim)

    def forward(self, x, fusion_matrix):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        k = torch.einsum('bhnd,bdd->bhnd', q, fusion_matrix)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self, dataset):
        super(DetailNode, self).__init__()
        if dataset == 'Houston':
            inp, oup = 72, 72
            self.shffleconv = nn.Conv2d(144, 144, kernel_size=1, stride=1, padding=0, bias=True)
        elif dataset == 'Trento':
            inp, oup = 31, 31
            self.shffleconv = nn.Conv2d(62, 62, kernel_size=1, stride=1, padding=0, bias=True)
        elif dataset == 'MUUFL':
            inp, oup = 32, 32
            self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=inp, oup=oup, expand_ratio=2)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * self.theta_rho(z2) + self.theta_eta(z2)
        return z1, z2


class FusionMatrixGenerator(nn.Module):
    def __init__(self, dim):
        super(FusionMatrixGenerator, self).__init__()
        self.fc = nn.Linear(dim * 2, dim * dim, bias=False)

    def forward(self, hs, lidar):
        hs_mean = hs.mean(dim=1)
        lidar_mean = lidar.mean(dim=1)
        x = torch.cat([hs_mean, lidar_mean], dim=-1)
        fusion_matrix = self.fc(x)
        fusion_matrix = torch.sigmoid(fusion_matrix)
        return fusion_matrix.view(hs.size(0), hs.size(2), hs.size(2))

class Vison_transformer(nn.Module):
    def __init__(self, num_patches, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super(Vison_transformer, self).__init__()

        self.patch_to_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.to_latent = nn.Identity()

    def forward(self, x, fusion_matrix=None):
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim] (16, 1, 32)
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        if fusion_matrix is not None:
            x = self.transformer(x, fusion_matrix)
        else:
            x = self.transformer(x)
        x = x[:, 0]
        x = self.to_latent(x)
        return x

class CrossModalityFusion(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout):
        super(CrossModalityFusion, self).__init__()
        self.patch_to_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        self.cls_token_hsi, self.cls_token_lidar = nn.Parameter(torch.randn(1, 1, dim)), nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_hsi, self.pos_embedding_lidar = nn.Parameter(torch.randn(1, num_patches + 1, dim)), nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.layers = nn.ModuleList([])
        self.fusion_matrix = FusionMatrixGenerator(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_Matrix(dim, AttentionFusionMatrix(dim, heads=heads, dim_head=dim_head, dropout=dropout), ),
                PreNorm_Matrix(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def _exchange_embedding(self, hsi, lidar):
        hsi_embed1 = hsi[:, 0:hsi.size(1)//2, :]
        hsi_embed2 = hsi[:, hsi.size(1)//2::, :]
        lidar_embed1 = hsi[:, 0:lidar.size(1)//2, :]
        lidar_embed2 = hsi[:, lidar.size(1)//2::, :]
        hsi = torch.cat([hsi_embed1, lidar_embed2], dim=1)
        lidar = torch.cat([lidar_embed1, hsi_embed2], dim=1)
        return hsi,lidar

    def _exchange_token(self, hsi, lidar):
        hsi_token = hsi[:, 0]
        hsi_remain_token = hsi[:, 1:]
        lidar_token = lidar[:, 0]
        lidar_remain_token = lidar[:, 1:]
        hsi_new_token = torch.cat([lidar_token.unsqueeze(1), hsi_remain_token], dim=1)
        lidar_new_token = torch.cat([hsi_token.unsqueeze(1), lidar_remain_token], dim=1)
        return hsi_new_token, lidar_new_token

    def forward(self, hsi, lidar):
        hsi_embedding, lidar_embedding = self.patch_to_embedding(hsi), self.patch_to_embedding(lidar) #hsi_embedding:(16, 144, 225) lidar_embedding:(16, 144, 225)
        fusion_matrix = self.fusion_matrix(hsi_embedding, lidar_embedding)
        hsi_emb_ex, lidar_emb_ex = self._exchange_embedding(hsi_embedding, lidar_embedding)
        b, n, _ = hsi_emb_ex.shape
        cls_token_hsi = repeat(self.cls_token_hsi, '() n d -> b n d', b=b)
        cls_token_lidar = repeat(self.cls_token_lidar, '() n d -> b n d', b=b)
        hsi_emb_ex = torch.cat((cls_token_hsi, hsi_emb_ex), dim=1)
        lidar_emb_ex = torch.cat((cls_token_lidar, lidar_emb_ex), dim=1)
        hsi_emb_ex += self.pos_embedding_hsi[:, :(n + 1)]
        lidar_emb_ex += self.pos_embedding_lidar[:, :(n + 1)]
        for attn, ff in self.layers:
            hsi_emb_ex = attn(hsi_emb_ex, fusion_matrix) + hsi_emb_ex
            hsi_emb_ex = ff(hsi_emb_ex) + hsi_emb_ex
            lidar_emb_ex = attn(lidar_emb_ex, fusion_matrix) + lidar_emb_ex
            lidar_emb_ex = ff(lidar_emb_ex) + lidar_emb_ex
        hsi_emb_ex, lidar_emb_ex = self._exchange_token(hsi_emb_ex, lidar_emb_ex)
        for attn, ff in self.layers:
            hsi_emb_ex = attn(hsi_emb_ex, fusion_matrix) + hsi_emb_ex
            hsi_emb_ex = ff(hsi_emb_ex) + hsi_emb_ex
            lidar_emb_ex = attn(lidar_emb_ex, fusion_matrix) + lidar_emb_ex
            lidar_emb_ex = ff(lidar_emb_ex) + lidar_emb_ex
        hsi_emb_ex = hsi_emb_ex[:, 0]
        lidar_emb_ex = lidar_emb_ex[:, 0]
        output = hsi_emb_ex + lidar_emb_ex
        return output

class LocalFeatureExtraction(nn.Module):
    def __init__(self, num_layers, dataset, flag):
        super(LocalFeatureExtraction, self).__init__()
        INNmodules = [DetailNode(dataset) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
        self.flag = flag

    def forward(self, x):
        if self.flag == None:
            z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2 : (x.shape[1])]
            for layer in self.net:
                z1, z2 = layer(z1, z2)
            return torch.cat((z1, z2), dim=1)
        else:
            X_token = x[:, 1]
            X = x[:, 1:x.shape[1]]
            z1, z2 = X[:, :X.shape[1] // 2], X[:, X.shape[1] // 2: (X.shape[1])]
            for layer in self.net:
                z1, z2 = layer(z1, z2)
            return torch.cat((X_token.unsqueeze(1), z1, z2), dim=1)

# class DWConv(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
    #     super(DWConv, self).__init__()
    #     self.depthwiseconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
    #     self.pintwiseconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
    # def forward(self, x):
    #     x = self.depthwiseconv(x)
    #     x = self.pintwiseconv(x)
    #     return x

class Conv3_D(nn.Module):
    def __init__(self, in_chan, dim, kernel_size, padding, groups):
        super().__init__()
        self.conv3D = nn.Conv3d(in_chan, dim, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, groups=groups, bias=False)
    def forward(self, x):
        x = self.conv3D(x)
        return x

class DWSepConv3d(nn.Module):
    def __init__(self, dim):
        super(DWSepConv3d, self).__init__()
        self.conv_s = Conv3_D(dim, dim, (3, 1, 1), (1, 0, 0), dim)
        self.conv_t = Conv3_D(dim, dim, (1, 1, 1), (0, 0, 0), 1)
        self.bn_s = nn.BatchNorm3d(dim, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()


    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = DWSepConv3d(dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class SpectralFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

class Network(nn.Module):
    def __init__(self, input_channels, window_size, num_classes, dataset):
        super(Network, self).__init__()
        # Initial convolution

        self.initial_conv = nn.Sequential(
            nn.Conv2d(2 if dataset == 'MUUFL' else 1, input_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channels)
        )
        # Embedding for patches
        self.patch_to_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        # Spatial and Spectral enhancement modules
        self.spatial_enhance_module = Spatial_Enhance_Module(input_channels, 60, window_size)
        self.spectral_enhance_module = Spectral_Enhance_Module(input_channels, input_channels)
        # Local Feature Extraction
        # is_trento_dataset = (dataset == 'Trento')
        self.localfe = LocalFeatureExtraction(num_layers=3, dataset=dataset, flag=1 if dataset == 'Trento' else None)
        # Vision Transformers
        self.vit_combined = Vison_transformer(window_size ** 2, (input_channels + 2) if dataset == 'MUUFL' else (input_channels + 1), 2, 3, 32, 64, dropout=0.12)
        self.vit_hsi_lidar = Vison_transformer(window_size ** 2,  input_channels, 2, 3, 32, 64, dropout=0.12)
        self.crossmodalityfusion = CrossModalityFusion(input_channels, window_size ** 2, 1, 3, input_channels, 64, 0.12)
        # Classifier layer
        self.classifier = nn.Linear((input_channels * 4 + 2) if dataset == 'MUUFL' else (input_channels * 4 + 1), num_classes, bias=False)
        # Pooling Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
        )
    def _process_enhancements(self, hsi_input, lidar_input_loca, hsi_input_loca, lidar_input):
        hsi_enhanced = self.spatial_enhance_module(hsi_input_loca, lidar_input)
        lidar_enhanced = self.spectral_enhance_module(hsi_input, lidar_input_loca)
        fused_features = torch.cat([hsi_enhanced, lidar_enhanced], dim=1)
        return fused_features

    def forward(self, hsi_input, lidar_input):
        combined_input = torch.cat([hsi_input, lidar_input], 1)
        combined_token = self.vit_combined(combined_input)
        lidar_input = self.initial_conv(lidar_input)
        hsi_input_loca = self.localfe(hsi_input)
        lidar_input_loca = self.localfe(lidar_input)
        cross_hsi_lidar_token = self.crossmodalityfusion(hsi_input, lidar_input)
        hsi_lidar = hsi_input + lidar_input
        hsi_lidar_token = self.vit_hsi_lidar(hsi_lidar)
        fused_features = self._process_enhancements(hsi_input, lidar_input_loca, hsi_input_loca, lidar_input)
        fused_features = self.fusion_layer(fused_features)
        fused_features = torch.flatten(self.global_avg_pool(fused_features), 1)
        final_features = torch.cat([combined_token, hsi_lidar_token, cross_hsi_lidar_token, fused_features], dim=-1)
        return self.classifier(final_features)
