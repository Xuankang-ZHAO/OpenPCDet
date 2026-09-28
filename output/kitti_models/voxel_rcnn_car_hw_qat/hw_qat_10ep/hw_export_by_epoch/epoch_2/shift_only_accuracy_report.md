# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9758776503302051, 'recall/rcnn_0.3': 0.975669099756691, 'recall/roi_0.5': 0.9655891553701773, 'recall/rcnn_0.5': 0.9687869308307264, 'recall/roi_0.7': 0.7958985053875565, 'recall/rcnn_0.7': 0.8468543621828294, 'Car_aos/easy_R40': 98.73937495941657, 'Car_aos/moderate_R40': 93.99635760771619, 'Car_aos/hard_R40': 91.68516596290375, 'Car_3d/easy_R40': 91.5260197707538, 'Car_3d/moderate_R40': 82.00350890142806, 'Car_3d/hard_R40': 79.54427774739837, 'Car_bev/easy_R40': 95.66678101039736, 'Car_bev/moderate_R40': 90.32346512154696, 'Car_bev/hard_R40': 88.32614285658981, 'Car_image/easy_R40': 98.82041696721909, 'Car_image/moderate_R40': 94.23892456084413, 'Car_image/hard_R40': 91.98752784926013}
```
