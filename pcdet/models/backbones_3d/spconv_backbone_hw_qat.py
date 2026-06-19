from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    return spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )


class VoxelBackBone8x_HWQAT(nn.Module):
    """Hardware-constrained QAT version of SECOND VoxelBackBone8x.

    The module keeps OpenPCDet's original sparse-conv topology, but adds:
      - signed symmetric activation observers/fake-quant per backbone layer
      - signed symmetric weight fake-quant, per-output-channel by default
      - export helpers for layer inventory, activation scales, weight scales

    It is intentionally narrow: only the 3D backbone is quantized while the rest
    of SECOND remains FP32.
    """

    _ACT_KEYS = (
        'input',
        'l00_conv_input',
        'l01_conv1',
        'l02_conv2_0',
        'l03_conv2_1',
        'l04_conv2_2',
        'l05_conv3_0',
        'l06_conv3_1',
        'l07_conv3_2',
        'l08_conv4_0',
        'l09_conv4_1',
        'l10_conv4_2',
        'l11_conv_out',
    )

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self._hw_qat_enabled = False
        self._hw_calibration_enabled = False
        self._hw_fake_quant_enabled = True
        self._hw_reference_enabled = False
        self._hw_observer_mode = 'max'
        self._hw_observer_momentum = 0.95
        self._hw_weight_quant = 'per_channel'
        self._hw_rounding = True
        self._hw_qparams = None

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
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
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

        if (not bool(seen_buf.item())) or self._hw_observer_mode == 'max':
            new_scale = torch.maximum(scale_buf.to(cur_scale.device), cur_scale) if bool(seen_buf.item()) else cur_scale
        elif self._hw_observer_mode == 'momentum':
            new_scale = scale_buf.to(cur_scale.device) * self._hw_observer_momentum + cur_scale * (1.0 - self._hw_observer_momentum)
        else:
            raise ValueError('Unsupported observer mode: %s' % self._hw_observer_mode)

        self._set_act_scale(key, new_scale)
        return new_scale.to(features.device)

    def _activation_fake_quant(self, key, features: Tensor, ste=True):
        if self._hw_qat_enabled or self._hw_calibration_enabled:
            scale = self._update_activation_observer(key, features)
        else:
            scale = self._get_act_scale(key).to(features.device).clamp_min(1e-8)

        should_quant = (
            self._hw_fake_quant_enabled and
            ((self._hw_qat_enabled and self.training) or self._hw_reference_enabled)
        )
        if not should_quant:
            return features

        q = (features / scale).round().clamp(-127, 127)
        dq = q * scale
        if ste:
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

    def enable_calibration(self, enable=True, observer='max', observer_momentum=0.95):
        self._hw_calibration_enabled = bool(enable)
        self._hw_observer_mode = observer
        self._hw_observer_momentum = float(observer_momentum)

    def freeze_activation_observers(self):
        self._hw_calibration_enabled = False

    def reset_activation_observers(self):
        for key in self._ACT_KEYS:
            getattr(self, self._scale_name(key)).data.fill_(1.0)
            getattr(self, self._seen_name(key)).data.fill_(False)

    def enable_hw_reference(self, enable=True, qparams=None):
        self._hw_reference_enabled = bool(enable)
        self._hw_qparams = qparams

    def get_activation_qparams(self):
        rows = []
        for idx, key in enumerate(self._ACT_KEYS):
            rows.append({
                'activation_id': idx,
                'name': key,
                'scale': float(self._get_act_scale(key).detach().cpu()),
                'zero_point': 0,
                'signed': True,
                'qmin': -127,
                'qmax': 127,
                'seen': bool(getattr(self, self._seen_name(key)).item()),
            })
        return rows

    def get_layer_modules(self):
        return [
            ('conv_input.0', self.conv_input[0], self.conv_input[1], self.conv_input[2], 'l00_conv_input'),
            ('conv1.0.0', self.conv1[0][0], self.conv1[0][1], self.conv1[0][2], 'l01_conv1'),
            ('conv2.0.0', self.conv2[0][0], self.conv2[0][1], self.conv2[0][2], 'l02_conv2_0'),
            ('conv2.1.0', self.conv2[1][0], self.conv2[1][1], self.conv2[1][2], 'l03_conv2_1'),
            ('conv2.2.0', self.conv2[2][0], self.conv2[2][1], self.conv2[2][2], 'l04_conv2_2'),
            ('conv3.0.0', self.conv3[0][0], self.conv3[0][1], self.conv3[0][2], 'l05_conv3_0'),
            ('conv3.1.0', self.conv3[1][0], self.conv3[1][1], self.conv3[1][2], 'l06_conv3_1'),
            ('conv3.2.0', self.conv3[2][0], self.conv3[2][1], self.conv3[2][2], 'l07_conv3_2'),
            ('conv4.0.0', self.conv4[0][0], self.conv4[0][1], self.conv4[0][2], 'l08_conv4_0'),
            ('conv4.1.0', self.conv4[1][0], self.conv4[1][1], self.conv4[1][2], 'l09_conv4_1'),
            ('conv4.2.0', self.conv4[2][0], self.conv4[2][1], self.conv4[2][2], 'l10_conv4_2'),
            ('conv_out.0', self.conv_out[0], self.conv_out[1], self.conv_out[2], 'l11_conv_out'),
        ]

    def get_layer_inventory(self):
        rows = []
        for layer_id, (name, conv, bn, relu, act_key) in enumerate(self.get_layer_modules()):
            rows.append({
                'layer_id': layer_id,
                'module_name': name,
                'conv_type': type(conv).__name__,
                'kernel': tuple(conv.kernel_size),
                'stride': tuple(conv.stride),
                'cin': int(conv.in_channels),
                'cout': int(conv.out_channels),
                'has_bn': bn is not None,
                'has_relu': relu is not None,
                'has_residual': False,
                'activation_key': act_key,
            })
        return rows

    def get_weight_qparams(self):
        rows = []
        for layer_id, (name, conv, _bn, _relu, _act_key) in enumerate(self.get_layer_modules()):
            weight = conv.weight.detach()
            scale = self._weight_scale(conv, weight).detach().cpu()
            q = (weight.detach().cpu() / self._view_scale_as_weight(conv, weight.detach().cpu(), scale)).round().clamp(-127, 127).to(torch.int8)
            if scale.dim() == 0:
                rows.append({
                    'layer_id': layer_id,
                    'module_name': name,
                    'channel': -1,
                    'scale': float(scale),
                    'zero_point': 0,
                    'qmin': int(q.min()),
                    'qmax': int(q.max()),
                })
            else:
                for channel, channel_scale in enumerate(scale.view(-1)):
                    rows.append({
                        'layer_id': layer_id,
                        'module_name': name,
                        'channel': channel,
                        'scale': float(channel_scale),
                        'zero_point': 0,
                        'qmin': int(q.min()),
                        'qmax': int(q.max()),
                    })
        return rows

    def _hw_shift_requant(self, features, bias_int, shift, sy, relu_en=True):
        bias = bias_int.to(features.device).float().view(1, -1)
        shift = shift.to(features.device).float().view(1, -1)
        acc = features + bias
        if self._hw_rounding:
            rounding = torch.where(shift > 0, torch.pow(2.0, shift - 1.0), torch.zeros_like(shift))
            acc = acc + rounding
        q = torch.floor(acc / torch.pow(2.0, shift)).clamp(-127, 127)
        if relu_en:
            q = torch.clamp_min(q, 0)
        return q * float(sy)

    def _forward_hw_reference_layer(self, sparse_tensor, layer_id, layer_info):
        name, conv, _bn, relu, _act_key = layer_info
        qparam = self._hw_qparams[layer_id]
        sx = float(qparam['sx'])
        sy = float(qparam['sy'])
        weight_int8 = qparam['weight_int8'].to(conv.weight.device).float().contiguous()
        bias_int = qparam['bias_int'].to(conv.weight.device)
        shift = qparam['shift'].to(conv.weight.device)

        q_features = (sparse_tensor.features / max(sx, 1e-12)).round().clamp(-127, 127)
        sparse_tensor = replace_feature(sparse_tensor, q_features)

        orig_weight = conv.weight.data.clone()
        try:
            with torch.no_grad():
                conv.weight.data.copy_(weight_int8)
            out = conv(sparse_tensor)
        finally:
            with torch.no_grad():
                conv.weight.data.copy_(orig_weight)

        out_features = self._hw_shift_requant(
            out.features,
            bias_int=bias_int,
            shift=shift,
            sy=sy,
            relu_en=(relu is not None),
        )
        return replace_feature(out, out_features)

    def _forward_hw_reference(self, input_sp_tensor):
        layer_modules = self.get_layer_modules()
        current = input_sp_tensor
        x_conv1 = x_conv2 = x_conv3 = x_conv4 = None

        for layer_id, layer_info in enumerate(layer_modules):
            current = self._forward_hw_reference_layer(current, layer_id, layer_info)
            if layer_id == 1:
                x_conv1 = current
            elif layer_id == 4:
                x_conv2 = current
            elif layer_id == 7:
                x_conv3 = current
            elif layer_id == 10:
                x_conv4 = current

        return current, x_conv1, x_conv2, x_conv3, x_conv4

    def _forward_layers(self, input_sp_tensor):
        x = self.conv_input(input_sp_tensor)
        x = self._quant_sparse_tensor('l00_conv_input', x, ste=self.training)

        x_conv1 = self.conv1(x)
        x_conv1 = self._quant_sparse_tensor('l01_conv1', x_conv1, ste=self.training)

        x_conv2 = self.conv2[0](x_conv1)
        x_conv2 = self._quant_sparse_tensor('l02_conv2_0', x_conv2, ste=self.training)
        x_conv2 = self.conv2[1](x_conv2)
        x_conv2 = self._quant_sparse_tensor('l03_conv2_1', x_conv2, ste=self.training)
        x_conv2 = self.conv2[2](x_conv2)
        x_conv2 = self._quant_sparse_tensor('l04_conv2_2', x_conv2, ste=self.training)

        x_conv3 = self.conv3[0](x_conv2)
        x_conv3 = self._quant_sparse_tensor('l05_conv3_0', x_conv3, ste=self.training)
        x_conv3 = self.conv3[1](x_conv3)
        x_conv3 = self._quant_sparse_tensor('l06_conv3_1', x_conv3, ste=self.training)
        x_conv3 = self.conv3[2](x_conv3)
        x_conv3 = self._quant_sparse_tensor('l07_conv3_2', x_conv3, ste=self.training)

        x_conv4 = self.conv4[0](x_conv3)
        x_conv4 = self._quant_sparse_tensor('l08_conv4_0', x_conv4, ste=self.training)
        x_conv4 = self.conv4[1](x_conv4)
        x_conv4 = self._quant_sparse_tensor('l09_conv4_1', x_conv4, ste=self.training)
        x_conv4 = self.conv4[2](x_conv4)
        x_conv4 = self._quant_sparse_tensor('l10_conv4_2', x_conv4, ste=self.training)

        out = self.conv_out(x_conv4)
        out = self._quant_sparse_tensor('l11_conv_out', out, ste=self.training)

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

        if self._hw_reference_enabled and self._hw_qparams is not None:
            out, x_conv1, x_conv2, x_conv3, x_conv4 = self._forward_hw_reference(input_sp_tensor)
        else:
            replaced = []
            if self._hw_qat_enabled and self.training and self._hw_fake_quant_enabled:
                replaced = self._apply_weight_fake_quant_all()
            elif self._hw_reference_enabled:
                replaced = self._apply_weight_fake_quant_all()

            try:
                out, x_conv1, x_conv2, x_conv3, x_conv4 = self._forward_layers(input_sp_tensor)
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
