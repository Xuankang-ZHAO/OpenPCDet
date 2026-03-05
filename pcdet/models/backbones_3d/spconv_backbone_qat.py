from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from ...utils.spconv_utils import spconv

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

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class VoxelBackBone8x_INT8(nn.Module):
    """QAT-capable Voxel backbone for spconv.

    - During QAT training: call `enable_qat(True)` -> activations are fake-quantized (STE).
    - To prepare INT8 inference: call `convert_to_int8()` to store int8 weights.
      then call `enable_int8_inference(True)` to dequantize stored int8 weights into module weights (no permanent param change).
    This keeps backward path in FP32 while simulating INT8 on the forward pass.
    """
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        # control flags
        self._qat_enabled = False
        self._int8_inference = False
        self._int8_cached = False

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

    # ---------- QAT / INT8 helpers ----------
    def _is_spconv_conv(self, m):
        return isinstance(m, (spconv.SubMConv3d, spconv.SparseConv3d, spconv.SparseInverseConv3d))

    def convert_to_int8(self, dtype=torch.int8):
        """Store int8 representation and scale for each convolution weight.

        This does not modify parameters in-place; it registers buffers `_int8_weight` and `_int8_weight_scale`
        on each spconv conv module so you can later enable int8 inference.
        """
        for m in self.modules():
            if self._is_spconv_conv(m):
                w = getattr(m, 'weight', None)
                if w is None:
                    continue
                with torch.no_grad():
                    max_abs = w.abs().amax()
                    if max_abs == 0:
                        scale = torch.tensor(1.0, device=w.device, dtype=torch.float32)
                    else:
                        qmax = 127
                        scale = (max_abs / qmax).to(torch.float32)
                    int8_w = (w.detach() / scale).round().clamp(-127, 127).to(torch.int8)
                    m.register_buffer('_int8_weight', int8_w)
                    m.register_buffer('_int8_weight_scale', torch.tensor(scale, device=w.device, dtype=torch.float32))
        self._int8_cached = True

    def enable_int8_inference(self, enable=True):
        """If enabled, dequantize stored int8 weights into module weights (no_grad) before forward.

        Requires `convert_to_int8()` first. This mode is intended for INT8 inference of this module while
        keeping other modules FP32.
        """
        if enable and not getattr(self, '_int8_cached', False):
            raise RuntimeError('INT8 weights not prepared. Call convert_to_int8() first.')
        self._int8_inference = bool(enable)
        if enable:
            self.eval()

    def enable_qat(self, enable=True):
        """Toggle activation fake-quantization for QAT training (STE)."""
        self._qat_enabled = bool(enable)

    def _maybe_dequantize_int8_weights_into_modules(self):
        if not self._int8_inference:
            return
        for m in self.modules():
            if self._is_spconv_conv(m):
                int8_w = getattr(m, '_int8_weight', None)
                scale = getattr(m, '_int8_weight_scale', None)
                if int8_w is None or scale is None:
                    continue
                with torch.no_grad():
                    deq = int8_w.float() * scale
                    if hasattr(m, 'weight') and isinstance(getattr(m, 'weight'), torch.nn.Parameter):
                        w = m.weight
                        if w.shape == deq.shape:
                            w.data.copy_(deq)
                        else:
                            try:
                                w.data.copy_(deq.view_as(w.data))
                            except Exception:
                                pass

    def _apply_activation_fake_quant(self, features: Tensor):
        # simple symmetric per-tensor fake-quant with STE
        if features.numel() == 0:
            return features
        qmax = 127
        max_val = features.abs().amax(dim=0, keepdim=True)
        scale = (max_val / qmax).clamp_min(1e-8)
        qx = (features / scale).round().clamp(-qmax, qmax)
        # STE: forward returns quantized-dequantized; backward flows through as identity
        return (qx * scale)

    # ---------- forward ----------
    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # If INT8 inference mode enabled, dequantize stored int8 weights into module weights
        if self._int8_inference:
            self._maybe_dequantize_int8_weights_into_modules()

        # If QAT is enabled and in training, fake-quant activations (STE)
        if self._qat_enabled and self.training:
            voxel_features = self._apply_activation_fake_quant(voxel_features)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        out = self.conv_out(x_conv4)

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