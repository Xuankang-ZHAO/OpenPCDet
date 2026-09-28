"""Run SECOND-MultiHead (CBGS) on the local NuScenes mini set.

Paths are resolved from this file, so the script can be launched from any
working directory:

    python pcdet/models/cbgs_second_multihead/infer_mini.py --num_frames 5
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def load_cfg():
    cfg_dir = HERE / 'cfgs'
    prev_cwd = Path.cwd()
    os.chdir(cfg_dir)
    try:
        cfg_from_yaml_file('cbgs_second_multihead.yaml', cfg)
    finally:
        os.chdir(prev_cwd)
    cfg.DATA_CONFIG.DATA_PATH = str(REPO_ROOT / 'data' / 'nuscenes')
    cfg.DATA_CONFIG.VERSION = 'v1.0-mini'
    return cfg


def to_serializable(anno):
    boxes = np.asarray(anno['boxes_lidar'])
    names = [str(x) for x in np.asarray(anno['name']).tolist()]
    scores = np.asarray(anno['score']).tolist()
    metadata = anno.get('metadata', {})
    token = metadata.get('token') if isinstance(metadata, dict) else None
    return {
        'frame_id': str(anno['frame_id']),
        'token': token,
        'num_boxes': int(len(names)),
        'names': names,
        'scores': [float(s) for s in scores],
        'boxes_lidar': boxes.astype(float).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='SECOND-MultiHead mini inference')
    parser.add_argument('--num_frames', type=int, default=5, help='number of val frames to run')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument(
        '--ckpt',
        type=str,
        default=str(HERE / 'cbgs_second_multihead_nds6229_updated.pth'),
    )
    parser.add_argument('--output_dir', type=str, default=str(HERE / 'output'))
    args = parser.parse_args()

    logger = common_utils.create_logger()
    model_cfg = load_cfg()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('DATA_PATH: %s' % model_cfg.DATA_CONFIG.DATA_PATH)
    logger.info('VERSION: %s' % model_cfg.DATA_CONFIG.VERSION)
    logger.info('ckpt: %s' % args.ckpt)

    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=model_cfg.DATA_CONFIG,
        class_names=model_cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )
    if len(dataset) < args.num_frames:
        raise RuntimeError('val set has %d frames, fewer than requested %d' % (len(dataset), args.num_frames))

    model = build_network(
        model_cfg=model_cfg.MODEL,
        num_class=len(model_cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    records = []
    num_seen = 0
    with torch.no_grad():
        for batch_dict in dataloader:
            load_data_to_gpu(batch_dict)
            start = time.time()
            pred_dicts, _ = model(batch_dict)
            elapsed_ms = (time.time() - start) * 1000.0

            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, dataset.class_names
            )
            for anno in annos:
                record = to_serializable(anno)
                record['infer_ms'] = elapsed_ms
                counts = Counter(record['names'])
                top_score = max(record['scores']) if record['scores'] else 0.0
                logger.info(
                    'frame %d/%d  id=%s  boxes=%d  top_score=%.3f  %.1f ms  %s'
                    % (
                        num_seen + 1,
                        args.num_frames,
                        record['frame_id'],
                        record['num_boxes'],
                        top_score,
                        elapsed_ms,
                        dict(counts),
                    )
                )
                records.append(record)
                num_seen += 1
                if num_seen >= args.num_frames:
                    break
            if num_seen >= args.num_frames:
                break

    result_path = output_dir / 'mini_val_5frames.json'
    with open(result_path, 'w') as f:
        json.dump(records, f)
    logger.info('saved %d frames to %s' % (len(records), result_path))


if __name__ == '__main__':
    main()
