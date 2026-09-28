# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9710114702815432, 'recall/rcnn_0.3': 0.9711505039972194, 'recall/roi_0.5': 0.9617657281890859, 'recall/rcnn_0.5': 0.9630865484880083, 'recall/roi_0.7': 0.7939520333680917, 'recall/rcnn_0.7': 0.8463677441779631, 'Car_aos/easy_R40': 98.5031335141389, 'Car_aos/moderate_R40': 94.14697231914488, 'Car_aos/hard_R40': 91.66982441051654, 'Car_3d/easy_R40': 91.33607994373628, 'Car_3d/moderate_R40': 82.15467348875235, 'Car_3d/hard_R40': 79.71803702390429, 'Car_bev/easy_R40': 93.20460260557486, 'Car_bev/moderate_R40': 90.57982085804494, 'Car_bev/hard_R40': 88.29643025203107, 'Car_image/easy_R40': 98.52731611803328, 'Car_image/moderate_R40': 94.36190972483139, 'Car_image/hard_R40': 91.97500982775514}
```
