# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.972957942301008, 'recall/rcnn_0.3': 0.9728189085853319, 'recall/roi_0.5': 0.964893986791797, 'recall/rcnn_0.5': 0.9660062565172054, 'recall/roi_0.7': 0.7905457073340285, 'recall/rcnn_0.7': 0.8479666319082377, 'Car_aos/easy_R40': 98.25368037720479, 'Car_aos/moderate_R40': 93.97907338039577, 'Car_aos/hard_R40': 91.56922524188114, 'Car_3d/easy_R40': 90.8091770176697, 'Car_3d/moderate_R40': 81.89711090453014, 'Car_3d/hard_R40': 79.4432438757722, 'Car_bev/easy_R40': 93.0822971390321, 'Car_bev/moderate_R40': 90.3382770098497, 'Car_bev/hard_R40': 88.12444612920005, 'Car_image/easy_R40': 98.27978424308326, 'Car_image/moderate_R40': 94.16856981064885, 'Car_image/hard_R40': 91.85648580196253}
```
