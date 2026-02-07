#!/usr/bin/env python3
"""Run SECOND inference once and record layer input/output shapes and parameter shapes.

This script supports CPU and CUDA devices. Use `--device cuda` to select the
default CUDA device, or `--device cuda:0` to select a specific GPU index.

Usage:
    python mycode/second_step.py --cfg tools/cfgs/kitti_models/second.yaml \
            --pcd mycode/000008.bin --ckpt mycode/second_7862.pth --device cuda:0

The script writes `second_trace.json` into the specified output directory.
The trace contains per-module input/output shapes and parameter shapes (numel, dtype).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='tools/cfgs/kitti_models/second.yaml')
    parser.add_argument('--pcd', type=str, default='mycode/000008.bin')
    parser.add_argument('--ckpt', type=str, default='mycode/second_7862.pth')
    parser.add_argument('--device', type=str, default='auto', help='cpu|cuda|auto')
    parser.add_argument('--out_dir', type=str, default='mycode/second_trace_outputs')
    args = parser.parse_args()

    # project root: parent of mycode/ (so imports like `pcdet` and `tools` work)
    PRJ_ROOT = Path(__file__).resolve().parents[1]
    if str(PRJ_ROOT) not in sys.path:
        sys.path.insert(0, str(PRJ_ROOT))

    from pcdet.config import cfg, cfg_from_yaml_file
    from tools.demo import DemoDataset
    from pcdet.models import build_network
    from mycode.second_instrument import ActivationRecorder, summarize_records
    from pcdet.utils import common_utils

    cfg_from_yaml_file(str(Path(args.cfg)), cfg)

    # Resolve device: 'auto' -> cuda if available else cpu. Supports 'cpu', 'cuda', 'cuda:0', etc.
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset and process one sample (this will run point_feature_encoder + data_processor -> voxels)
    pcd_path = Path(args.pcd)
    demo_ds = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=pcd_path, ext='.bin')
    if len(demo_ds) == 0:
        raise RuntimeError(f'No samples found in {pcd_path}')

    sample = demo_ds[0]
    print('Prepared sample keys:', list(sample.keys()))

    # Collate into a batch (numpy arrays)
    batch = demo_ds.collate_batch([sample])

    # Convert numpy arrays to torch tensors on chosen device (custom conversion to support CPU)
    def load_data_to_device(batch_dict, device):
        import numpy as _np
        import torch as _torch
        out = {}
        for key, val in batch_dict.items():
            if key in ['frame_id','metadata','calib','image_paths','ori_shape','img_process_infos']:
                out[key] = val
            elif key == 'camera_imgs':
                out[key] = val
            elif isinstance(val, _np.ndarray):
                if key in ['image_shape']:
                    out[key] = _torch.from_numpy(val).int().to(device)
                else:
                    out[key] = _torch.from_numpy(val).float().to(device)
            else:
                out[key] = val
        out['batch_size'] = batch_dict.get('batch_size', 1)
        return out

    batch_torch = load_data_to_device(batch, device)

    # Build model and load checkpoint. If running on CUDA, weights are loaded
    # onto GPU by passing to_cpu=False to `load_params_from_file`.
    # If you prefer to keep checkpoint loading on CPU then move to GPU later,
    # set to_cpu=True (this script uses to_cpu=(device.type=='cpu')).
    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_ds)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print('Warning: checkpoint not found:', ckpt_path)
    model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=(device.type=='cpu'))
    model.to(device)
    model.eval()

    # Attach hooks and record shapes
    rec = ActivationRecorder()

    def module_filter(name, module):
        cls = module.__class__.__name__.lower()
        return ('conv' in cls) or ('linear' in cls) or ('bn' in cls) or (len(list(module.children()))==0)

    rec.register_hooks(model, module_filter=module_filter)
    weights = rec.get_weight_info(model)
    print('Collected parameter shapes:', len(weights))

    # Run forward pass (no grad)
    with torch.no_grad():
        # many detectors implement forward to accept batch dict
        try:
            _ = model.forward(batch_torch)
        except Exception:
            # fallback to calling model as callable
            _ = model(batch_torch)

    rec.remove_hooks()

    act_summary = summarize_records(rec.records)
    trace = {'activations': act_summary, 'weights': weights}
    trace_path = out_dir / 'second_trace.json'
    with open(trace_path, 'w') as f:
        json.dump(trace, f, indent=2)
    print('Saved trace to', trace_path)

    # Print short samples
    for i, r in enumerate(act_summary[:30]):
        print(f"{i:02d}: {r['name']} | {r['type']} | in={r['in']} -> out={r['out']}")

    weights_sorted = sorted(weights, key=lambda x: x.get('numel',0), reverse=True)
    print('\nTop parameter tensors by size:')
    for w in weights_sorted[:10]:
        print(w['name'], w['shape'], 'numel=', w['numel'])


if __name__ == '__main__':
    main()
