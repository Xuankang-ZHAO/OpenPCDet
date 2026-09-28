# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9737921445950644, 'recall/rcnn_0.3': 0.9734445603058742, 'recall/roi_0.5': 0.9644768856447689, 'recall/rcnn_0.5': 0.9660757733750435, 'recall/roi_0.7': 0.8025026068821689, 'recall/rcnn_0.7': 0.8491484184914841, 'Car_aos/easy_R40': 98.20096405344249, 'Car_aos/moderate_R40': 93.98768604158704, 'Car_aos/hard_R40': 91.71684158141159, 'Car_3d/easy_R40': 91.33198739349308, 'Car_3d/moderate_R40': 81.98653962290194, 'Car_3d/hard_R40': 79.68987421711604, 'Car_bev/easy_R40': 95.14411154383812, 'Car_bev/moderate_R40': 90.28859135655142, 'Car_bev/hard_R40': 88.31086208124657, 'Car_image/easy_R40': 98.22767455070259, 'Car_image/moderate_R40': 94.19564421025424, 'Car_image/hard_R40': 91.99342634819467}
```
