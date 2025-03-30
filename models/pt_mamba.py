from typing import Optional, Tuple
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, trunc_normal_
# from logger import get_missing_parameters_message, get_unexpected_parameters_message

# from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation, farthest_point_sample, index_points

import numpy as np

import warnings
warnings.filterwarnings("ignore")

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None




# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Mamba Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Mamba block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None, ):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


# def fps(data, number):
#     '''
#         data B N 3
#         number int
#     '''
#     fps_idx = pointnet2_utils.furthest_point_sample(data, number)
#     fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
#     return fps_data


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).reshape(B, N, 1)
    dist += torch.sum(dst ** 2, -1).reshape(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number (k-nearest neighbors) for each query point
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # Compute square distances between each query point and all points
    sqrdists = square_distance(new_xyz, xyz)
    # Sort distances to get the nearest neighbors
    group_idx = sqrdists.argsort(dim=-1)[:, :, :nsample]  # Indices of the k-nearest points
    return group_idx


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, pts, fps_index):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        xyz = pts[:,:,:3]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        # fps_index = farthest_point_sample(xyz, self.num_group)  # B G 3
        # center = index_points(xyz, fps_index)
        feats = index_points(pts, fps_index)
        center = feats[:,:,:3]
        # knn to get the neighborhood
        idx = query_knn(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).reshape(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.reshape(-1)
        neighborhood = pts.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, -1).contiguous()
        # normalize
        neighborhood[:,:,:,:3] = neighborhood[:,:,:,:3] - center.unsqueeze(2)
        return neighborhood, feats


class Encoder(nn.Module):
    def __init__(self, in_channel, encoder_channel):
        super().__init__()
        self.in_channel = in_channel
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(in_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, -1)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class MixerModelForSegmentation(MixerModel):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path: int = 0.1,
            fetch_idx: Tuple[int] = [3, 7, 11],
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MixerModel, self).__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.fetch_idx = fetch_idx

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        feature_list = []
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            if idx in self.fetch_idx:
                if not self.fused_add_norm:
                    residual_output = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states_output = self.norm_f(residual_output.to(dtype=self.norm_f.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                    hidden_states_output = fused_add_norm_fn(
                        hidden_states,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                feature_list.append(hidden_states_output)
        return feature_list


class Config:
    def __init__(self):
        self.trans_dim = 512
        self.depth = 12 #12 # 6 #9
        self.drop_path_rate = 0.1
        self.fetch_idx = [3,7,11] #[3, 7, 11] # 1, 3, 5
        self.group_size = 32


        self.rms_norm = False
        self.use_cls_token = False
        self.drop_path = 0.2
        self.drop_out = 0
        

    def __contains__(self, key):
        return key in self.__dict__

cfg = Config()

def reorder(group_input_tokens, pos, center, series_idx_array):

    group_input_tokens_list = []
    pos_list = []
    center_list = []


    for i in range(series_idx_array.shape[1]):

        sorted_pts_idx = series_idx_array[:,i,:] # b n
        sorted_pts_idx_expanded = sorted_pts_idx.unsqueeze(-1).expand(-1, -1, group_input_tokens.size(-1))  # 将 sorted_pts_idx 扩展为 (B, N, D)
        sorted_center_idx_expanded = sorted_pts_idx.unsqueeze(-1).expand(-1, -1, center.size(-1))  # 将 sorted_pts_idx 扩展为 (B, N, D)
        
        sorted_group_input_tokens = torch.gather(group_input_tokens, 1, sorted_pts_idx_expanded)
        sorted_pos = torch.gather(pos, 1, sorted_pts_idx_expanded)
        sorted_center = torch.gather(center, 1, sorted_center_idx_expanded)

        group_input_tokens_list.append(sorted_group_input_tokens)
        pos_list.append(sorted_pos)
        center_list.append(sorted_center)
    
    group_input_tokens = torch.cat((group_input_tokens_list), dim=1)
    pos = torch.cat((pos_list), dim=1)
    center = torch.cat((center_list), dim=1)

    return group_input_tokens, pos, center


class get_model(nn.Module):
    def __init__(self, args, config=cfg):
        super().__init__()

        in_channel = 6
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = args.num_category

        self.group_size = config.group_size
        self.num_group = args.spoints
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = self.trans_dim
        self.encoder = Encoder(in_channel = in_channel, encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.blocks = MixerModelForSegmentation(d_model=self.trans_dim,
                                                n_layer=self.depth,
                                                rms_norm=config.rms_norm,
                                                drop_path=config.drop_path,
                                                fetch_idx=config.fetch_idx)

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)
        self.drop_path_rate = config.drop_path_rate
        self.drop_path_block = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                                 nn.BatchNorm1d(64),
        #                                 nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=int(self.trans_dim*3) + in_channel, mlp=[self.trans_dim * 4, 1024])

        self.convs1 = nn.Conv1d(self.trans_dim * 6 + 1024, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

   

    def forward(self, xyz, fps_index_array, series_idx_arrays):
    
        ###################################################################
        #xyz1 = xyz[:, :4, :]
        #xyz2 = xyz[:, 7:, :] # 8 channels in total, delete the 3 absolute coordinate channels. raw input channel is 11
        #pts = torch.cat([xyz1,xyz2], dim=1)
        #####################################################################
        pts = xyz
        
        ###################################################################
        
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2)  # B N C
        # divide the point cloud in the same form. This is important
        neighborhood, feats = self.group_divider(pts, fps_index_array[:,0,:])
        group_input_tokens = self.encoder(neighborhood)  # B G D

        center = feats[:,:,:3]
        pos = self.pos_embed(center)

        # reordering strategy
        group_input_tokens, pos, center = reorder(group_input_tokens, pos, center, series_idx_arrays[:, 0, :, :])

        # final input
        x = group_input_tokens

        feature_list = self.blocks(x, pos)

        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list), dim=1)  # 1152
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.reshape(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.reshape(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # cls_label_one_hot = cls_label.reshape(B, 16, 1)
        # cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)

        f_level_0 = self.propagation_0(pts[:,:,:3].transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, weight=None):
        gold = gold.contiguous().view(-1)


        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()


        return loss

class get_loss_weighted(nn.Module):
    def __init__(self):
      super(get_loss_weighted, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.cross_entropy(pred, target, weight)
        #total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':

    B, N, D = 2, 2048, 11  
    input = torch.randn(B, D, N).cuda() 

    model = get_model(8).cuda()

    y = model(input)

    print(y.shape)