# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9644073687869308, 'recall/rcnn_0.3': 0.9644768856447689, 'recall/roi_0.5': 0.9516857838025721, 'recall/rcnn_0.5': 0.9550921098366354, 'recall/roi_0.7': 0.7762947514772333, 'recall/rcnn_0.7': 0.8310740354535975, 'Car_aos/easy_R40': 97.7854288673958, 'Car_aos/moderate_R40': 93.35342977216472, 'Car_aos/hard_R40': 91.12377770630872, 'Car_3d/easy_R40': 91.15394362352363, 'Car_3d/moderate_R40': 81.83161881017182, 'Car_3d/hard_R40': 79.27813482823491, 'Car_bev/easy_R40': 94.80127881992061, 'Car_bev/moderate_R40': 89.81898638169517, 'Car_bev/hard_R40': 87.74462248248844, 'Car_image/easy_R40': 97.80162890363954, 'Car_image/moderate_R40': 93.57893508305685, 'Car_image/hard_R40': 91.49432674416325}
```
