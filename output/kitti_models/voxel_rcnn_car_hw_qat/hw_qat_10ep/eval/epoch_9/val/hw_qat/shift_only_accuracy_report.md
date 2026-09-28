# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9718456725755996, 'recall/rcnn_0.3': 0.9717761557177615, 'recall/roi_0.5': 0.96232186305179, 'recall/rcnn_0.5': 0.9641293013555787, 'recall/roi_0.7': 0.791936044490789, 'recall/rcnn_0.7': 0.8449774070212026, 'Car_aos/easy_R40': 98.22012176541057, 'Car_aos/moderate_R40': 94.16088798930863, 'Car_aos/hard_R40': 91.70246271561989, 'Car_3d/easy_R40': 90.77117363641221, 'Car_3d/moderate_R40': 81.74738240200631, 'Car_3d/hard_R40': 79.35200878543462, 'Car_bev/easy_R40': 93.05214816348328, 'Car_bev/moderate_R40': 90.3527131186916, 'Car_bev/hard_R40': 88.17385830921583, 'Car_image/easy_R40': 98.24021678364142, 'Car_image/moderate_R40': 94.36593934146345, 'Car_image/hard_R40': 92.02975766422648}
```
