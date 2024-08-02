import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        print(f"Embedder output shape: {out.shape}")  # 打印嵌入后的形状
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(DenseLayer, self).__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim)

        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation

    def forward(self, x):
        print(f"DenseLayer input shape: {x.shape}")  # 打印输入形状
        out = self.linear_layer(x)
        print(f"DenseLayer linear output shape: {out.shape}")  # 打印线性层输出形状
        out = self.activation(out)
        return out


class NeRFDecoder(nn.Module):
    def __init__(self, geometry_kwargs, radiance_kwargs, sdf_feat_dim, rgb_feat_dim):
        super(NeRFDecoder, self).__init__()
        self.geometry_net = PropMLP(**geometry_kwargs, input_feat_dim=sdf_feat_dim)
        self.radiance_net = NerfMLP(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, feat, view_dirs=None):
        # Get initial geometry features from PropMLP
        geometry, h = self.geometry_net(feat, return_h=True)
        print(f"Geometry shape: {geometry.shape}")  # 打印几何特征形状
        print(f"Intermediate feature shape: {h.shape}")  # 打印中间特征形状

        # Get final RGB and density from NerfMLP
        rgb_and_density = self.radiance_net(geometry, view_dirs=view_dirs)
        print(f"RGB and density shape: {rgb_and_density.shape}")  # 打印最终RGB和密度形状

        return rgb_and_density


class PropMLP(nn.Module):
    def __init__(self, W=128, D=4, skips=[], input_feat_dim=16, n_freq=-1, weight_norm=False, concat_qp=False,
                 use_view_dirs=False, use_normals=False, use_dot_prod=False):
        super(PropMLP, self).__init__()
        self.use_view_dirs = use_view_dirs
        self.use_dot_prod = use_dot_prod
        self.use_normals = use_normals
        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips

        # Build the MLP
        layers = []

        for l in range(D + 1):
            if l == D:
                out_dim = 1
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)


    def forward(self, feat, return_h=False):
        feat = self.embed_fn(feat)
        print(f"Feature shape before layers: {feat.shape}")
        h = feat

        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        if return_h:  # return feature
            return h[..., :1], h[..., 1:]
        else:
            return h[..., :1]


class NerfMLP(nn.Module):
    def __init__(self, W=128, D=4, skips=[], input_feat_dim=16, n_freq=-1, weight_norm=False, concat_qp=False,
                 use_view_dirs=False, use_normals=False, use_dot_prod=False):
        super(NerfMLP, self).__init__()
        self.use_view_dirs = use_view_dirs
        self.use_normals = use_normals
        self.use_dot_prod = use_dot_prod
        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []

        # Build the MLP
        for l in range(D + 1):
            if l == D:
                out_dim = 1
            elif l + 1 in self.skips:
                out_dim = W - input_ch
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, return_h=False):
        feat = self.embed_fn(feat)
        print(f"Feature shape before layers: {feat.shape}")

        h = feat

        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        if return_h:  # return feature
            return h[..., :1], h[..., 1:]
        else:
            return h[..., :1]


class NeRF360Decoder(nn.Module):
    def __init__(self, geometry_kwargs, radiance_kwargs, sdf_feat_dim, rgb_feat_dim):
        super(NeRF360Decoder, self).__init__()
        self.prop_mlp = PropMLP(**geometry_kwargs, input_feat_dim=sdf_feat_dim)
        self.nerf_mlp = NerfMLP(**radiance_kwargs, input_feat_dim=rgb_feat_dim)

    def forward(self, feat, view_dirs=None):
        # Get initial geometry features from PropMLP
        geometry, h = self.prop_mlp(feat, return_h=True)
        print(f"PropMLP output geometry shape: {geometry.shape}")  # 打印PropMLP输出的几何形状
        print(f"PropMLP intermediate feature shape: {h.shape}")  # 打印PropMLP中间特征形状

        # Get final RGB and density from NerfMLP
        rgb_and_density = self.nerf_mlp(geometry, view_dirs=view_dirs)
        print(f"Final RGB and density shape: {rgb_and_density.shape}")  # 打印最终RGB和密度形状

        return rgb_and_density
