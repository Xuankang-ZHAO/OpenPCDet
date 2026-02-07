import json
import torch
import numpy as np


def _shape_of(x):
    # returns a JSON-serializable shape description
    if isinstance(x, torch.Tensor):
        return list(x.size())
    try:
        # numpy array
        return list(np.shape(x))
    except Exception:
        return str(type(x))


class ActivationRecorder:
    """Record module forward inputs/outputs (shapes) and parameter shapes.

    Usage:
      rec = ActivationRecorder()
      rec.register_hooks(model)
      with torch.no_grad(): model(batch)
      rec.records  # list of dicts per hooked module
      rec.get_weight_info(model)
    """

    def __init__(self):
        self.hooks = []
        self.records = []

    def _hook(self, name, module, inp, out):
        try:
            in_shapes = None
            if isinstance(inp, (list, tuple)):
                in_shapes = [_shape_of(x) for x in inp]
            else:
                in_shapes = _shape_of(inp)

            out_shapes = None
            if isinstance(out, (list, tuple)):
                out_shapes = [_shape_of(x) for x in out]
            else:
                out_shapes = _shape_of(out)

            rec = {
                'name': name,
                'module_type': module.__class__.__name__,
                'input_shape': in_shapes,
                'output_shape': out_shapes,
            }
            self.records.append(rec)
        except Exception as e:
            # best-effort recording
            self.records.append({'name': name, 'module_type': module.__class__.__name__, 'error': str(e)})

    def register_hooks(self, model, module_filter=None):
        """Register forward hooks on modules.

        module_filter: optional callable(name, module) -> bool to select modules to hook.
        If None, hooks will be attached to all leaf modules (modules with no children) and
        common conv/linear/bn blocks.
        """
        for name, module in model.named_modules():
            if module_filter is not None and not module_filter(name, module):
                continue
            # avoid hooking the top-level container twice
            # attach to modules that are likely useful (conv, linear, bn, spconv variants) or leaves
            clsname = module.__class__.__name__.lower()
            attach = False
            if len(list(module.children())) == 0:
                attach = True
            if 'conv' in clsname or 'linear' in clsname or 'bn' in clsname or 'spconv' in clsname or 'subm' in clsname:
                attach = True

            if attach:
                h = module.register_forward_hook(lambda m, inp, out, n=name: self._hook(n, m, inp, out))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def clear(self):
        self.records = []

    def get_weight_info(self, model):
        weights = []
        for name, p in model.named_parameters():
            try:
                weights.append({
                    'name': name,
                    'shape': list(p.size()),
                    'numel': p.numel(),
                    'dtype': str(p.dtype),
                    'requires_grad': p.requires_grad,
                })
            except Exception:
                weights.append({'name': name, 'shape': 'error'})
        return weights

    def save(self, path):
        out = {'records': self.records}
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)


def summarize_records(records):
    """Return a concise summary (list of tuples) of records for display/printing."""
    summary = []
    for r in records:
        name = r.get('name')
        t = r.get('module_type')
        inp = r.get('input_shape')
        out = r.get('output_shape')
        summary.append({'name': name, 'type': t, 'in': inp, 'out': out})
    return summary
