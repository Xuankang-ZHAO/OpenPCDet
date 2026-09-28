"""Per-channel QAT for the NuScenes SECOND residual 3D backbone.

Weight fake-quant is signed symmetric INT8 on each output channel of every
sparse convolution. Activation fake-quant is signed symmetric INT8 with one
scale per convolution output. The residual add stays in floating point.
VFE, BEV backbone, and the detection head are not quantized.
"""
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone import SparseBasicBlock, post_act_block


class VoxelResBackBone8x_HWQAT(nn.Module):
    _ACT_KEYS = (
        'input',
        'conv_input',
        'c1b0_conv1', 'c1b0_out',
        'c1b1_conv1', 'c1b1_out',
        'c2_down',
        'c2b0_conv1', 'c2b0_out',
        'c2b1_conv1', 'c2b1_out',
        'c3_down',
        'c3b0_conv1', 'c3b0_out',
        'c3b1_conv1', 'c3b1_out',
        'c4_down',
        'c4b0_conv1', 'c4b0_out',
        'c4b1_conv1', 'c4b1_out',
        'conv_out',
    )

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self._hw_qat_enabled = False
        self._hw_calibration_enabled = False
        self._hw_fake_quant_enabled = True
        self._hw_eval_quant_enabled = False
        self._hw_observer_mode = 'max'
        self._hw_observer_momentum = 0.95
        self._hw_weight_quant = 'per_channel'

        for key in self._ACT_KEYS:
            self.register_buffer(self._scale_name(key), torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer(self._seen_name(key), torch.tensor(False, dtype=torch.bool))

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )
        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )
        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )
        self.conv4 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = self.model_cfg.get('last_pad', 0)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    @staticmethod
    def _scale_name(key):
        return '_hw_act_scale_' + key

    @staticmethod
    def _seen_name(key):
        return '_hw_act_seen_' + key

    def _get_act_scale(self, key):
        return getattr(self, self._scale_name(key))

    def _set_act_scale(self, key, scale):
        getattr(self, self._scale_name(key)).data.copy_(scale.detach().float().cpu())
        getattr(self, self._seen_name(key)).data.fill_(True)

    def _update_activation_observer(self, key, features: Tensor):
        if features.numel() == 0:
            return self._get_act_scale(key).to(features.device)

        qmax = 127.0
        cur_scale = (features.detach().abs().amax() / qmax).clamp_min(1e-8).to(torch.float32)
        scale_buf = self._get_act_scale(key)
        seen_buf = getattr(self, self._seen_name(key))
        seen = bool(seen_buf.item())

        if (not seen) or self._hw_observer_mode == 'max':
            new_scale = torch.maximum(scale_buf.to(cur_scale.device), cur_scale) if seen else cur_scale
        elif self._hw_observer_mode == 'momentum':
            new_scale = scale_buf.to(cur_scale.device) * self._hw_observer_momentum + cur_scale * (1.0 - self._hw_observer_momentum)
        else:
            raise ValueError('Unsupported observer mode: %s' % self._hw_observer_mode)

        self._set_act_scale(key, new_scale)
        return new_scale.to(features.device)

    def _activation_fake_quant(self, key, features: Tensor, ste=True):
        update_observer = self._hw_calibration_enabled or (self.training and self._hw_qat_enabled)
        if update_observer:
            scale = self._update_activation_observer(key, features)
        else:
            scale = self._get_act_scale(key).to(features.device).clamp_min(1e-8)

        do_quant = self._hw_fake_quant_enabled and (
            (self.training and self._hw_qat_enabled) or self._hw_eval_quant_enabled
        )
        if not do_quant:
            return features

        q = (features / scale).round().clamp(-127, 127)
        dq = q * scale
        if ste and self.training:
            return dq.detach() + (features - features.detach())
        return dq

    def _quant_sparse_tensor(self, key, sparse_tensor, ste=True):
        return replace_feature(sparse_tensor, self._activation_fake_quant(key, sparse_tensor.features, ste=ste))

    @staticmethod
    def _is_spconv_conv(module):
        return isinstance(module, (spconv.SubMConv3d, spconv.SparseConv3d, spconv.SparseInverseConv3d))

    @staticmethod
    def _out_channel_axis(conv, weight):
        if weight.dim() == 0:
            return 0
        if weight.shape[0] == conv.out_channels:
            return 0
        if weight.shape[-1] == conv.out_channels:
            return weight.dim() - 1
        return 0

    def _weight_scale(self, conv, weight):
        qmax = 127.0
        if self._hw_weight_quant == 'per_tensor':
            return (weight.detach().abs().amax() / qmax).clamp_min(1e-8)
        if self._hw_weight_quant != 'per_channel':
            raise ValueError('Unsupported weight quant mode: %s' % self._hw_weight_quant)
        axis = self._out_channel_axis(conv, weight)
        reduce_dims = [idx for idx in range(weight.dim()) if idx != axis]
        return (weight.detach().abs().amax(dim=reduce_dims) / qmax).clamp_min(1e-8)

    def _view_scale_as_weight(self, conv, weight, scale):
        if scale.dim() == 0:
            return scale
        axis = self._out_channel_axis(conv, weight)
        shape = [1] * weight.dim()
        shape[axis] = scale.numel()
        return scale.view(*shape)

    def _fake_quant_weight(self, conv):
        weight = getattr(conv, 'weight', None)
        if weight is None or not isinstance(weight, torch.nn.Parameter):
            return None
        scale = self._weight_scale(conv, weight).to(weight.device)
        scale_view = self._view_scale_as_weight(conv, weight, scale)
        q = (weight / scale_view).round().clamp(-127, 127)
        return q * scale_view

    def _apply_weight_fake_quant_all(self):
        replaced = []
        for module in self.modules():
            if not self._is_spconv_conv(module):
                continue
            weight = getattr(module, 'weight', None)
            if weight is None or not isinstance(weight, torch.nn.Parameter):
                continue
            dq = self._fake_quant_weight(module)
            if dq is None:
                continue
            setattr(module, '_hw_orig_weight_data', weight.data.clone())
            with torch.no_grad():
                weight.data.copy_(dq)
            replaced.append(module)
        return replaced

    @staticmethod
    def _restore_weight_fake_quant(replaced):
        for module in replaced:
            orig = getattr(module, '_hw_orig_weight_data', None)
            if orig is not None:
                with torch.no_grad():
                    module.weight.data.copy_(orig)
                delattr(module, '_hw_orig_weight_data')

    def enable_hw_qat(self, enable=True, weight_quant='per_channel', observer='max',
                      observer_momentum=0.95, fake_quant=True):
        self._hw_qat_enabled = bool(enable)
        self._hw_fake_quant_enabled = bool(fake_quant)
        self._hw_weight_quant = weight_quant
        self._hw_observer_mode = observer
        self._hw_observer_momentum = float(observer_momentum)

    def enable_hw_eval_quant(self, enable=True):
        self._hw_eval_quant_enabled = bool(enable)
        if enable:
            self._hw_fake_quant_enabled = True

    def num_quant_convs(self):
        return sum(1 for module in self.modules() if self._is_spconv_conv(module))

    def _forward_post_act(self, seq, sparse_tensor, act_key, ste):
        out = seq(sparse_tensor)
        return self._quant_sparse_tensor(act_key, out, ste=ste)

    def _forward_res_block(self, block, sparse_tensor, key_conv1, key_out, ste):
        identity = sparse_tensor
        out = block.conv1(sparse_tensor)
        out = replace_feature(out, block.bn1(out.features))
        out = replace_feature(out, block.relu(out.features))
        out = self._quant_sparse_tensor(key_conv1, out, ste=ste)

        out = block.conv2(out)
        out = replace_feature(out, block.bn2(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, block.relu(out.features))
        return self._quant_sparse_tensor(key_out, out, ste=ste)

    def _forward_layers(self, input_sp_tensor, ste):
        x = self._forward_post_act(self.conv_input, input_sp_tensor, 'conv_input', ste)

        x = self._forward_res_block(self.conv1[0], x, 'c1b0_conv1', 'c1b0_out', ste)
        x_conv1 = self._forward_res_block(self.conv1[1], x, 'c1b1_conv1', 'c1b1_out', ste)

        x = self._forward_post_act(self.conv2[0], x_conv1, 'c2_down', ste)
        x = self._forward_res_block(self.conv2[1], x, 'c2b0_conv1', 'c2b0_out', ste)
        x_conv2 = self._forward_res_block(self.conv2[2], x, 'c2b1_conv1', 'c2b1_out', ste)

        x = self._forward_post_act(self.conv3[0], x_conv2, 'c3_down', ste)
        x = self._forward_res_block(self.conv3[1], x, 'c3b0_conv1', 'c3b0_out', ste)
        x_conv3 = self._forward_res_block(self.conv3[2], x, 'c3b1_conv1', 'c3b1_out', ste)

        x = self._forward_post_act(self.conv4[0], x_conv3, 'c4_down', ste)
        x = self._forward_res_block(self.conv4[1], x, 'c4b0_conv1', 'c4b0_out', ste)
        x_conv4 = self._forward_res_block(self.conv4[2], x, 'c4b1_conv1', 'c4b1_out', ste)

        out = self._forward_post_act(self.conv_out, x_conv4, 'conv_out', ste)
        return out, x_conv1, x_conv2, x_conv3, x_conv4

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        ste = self.training and self._hw_qat_enabled
        voxel_features = self._activation_fake_quant('input', voxel_features, ste=ste)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        do_weight_quant = self._hw_fake_quant_enabled and (
            (self.training and self._hw_qat_enabled) or self._hw_eval_quant_enabled
        )
        replaced = self._apply_weight_fake_quant_all() if do_weight_quant else []
        try:
            out, x_conv1, x_conv2, x_conv3, x_conv4 = self._forward_layers(input_sp_tensor, ste)
        finally:
            if replaced:
                self._restore_weight_fake_quant(replaced)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict
