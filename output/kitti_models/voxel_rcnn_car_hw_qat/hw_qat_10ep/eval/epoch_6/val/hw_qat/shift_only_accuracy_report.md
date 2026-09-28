# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9721237400069517, 'recall/rcnn_0.3': 0.9719847062912756, 'recall/roi_0.5': 0.9639207507820646, 'recall/rcnn_0.5': 0.9645464025026069, 'recall/roi_0.7': 0.7965936739659367, 'recall/rcnn_0.7': 0.8471324296141814, 'Car_aos/easy_R40': 98.14369874858272, 'Car_aos/moderate_R40': 94.05710811530277, 'Car_aos/hard_R40': 91.60491543196082, 'Car_3d/easy_R40': 91.29300881117315, 'Car_3d/moderate_R40': 82.0909334350962, 'Car_3d/hard_R40': 79.54363929588845, 'Car_bev/easy_R40': 93.184955144269, 'Car_bev/moderate_R40': 90.49372278391283, 'Car_bev/hard_R40': 88.18574406078513, 'Car_image/easy_R40': 98.17368580351751, 'Car_image/moderate_R40': 94.2749694972815, 'Car_image/hard_R40': 91.9317534709778}
```
