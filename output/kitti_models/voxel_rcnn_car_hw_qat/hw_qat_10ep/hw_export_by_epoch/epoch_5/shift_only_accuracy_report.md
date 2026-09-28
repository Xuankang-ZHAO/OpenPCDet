# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9733750434480362, 'recall/rcnn_0.3': 0.9733750434480362, 'recall/roi_0.5': 0.9644768856447689, 'recall/rcnn_0.5': 0.964963503649635, 'recall/roi_0.7': 0.7952033368091762, 'recall/rcnn_0.7': 0.8435175530066041, 'Car_aos/easy_R40': 98.1014005202136, 'Car_aos/moderate_R40': 93.87065251370518, 'Car_aos/hard_R40': 91.47323933675648, 'Car_3d/easy_R40': 91.19389488618506, 'Car_3d/moderate_R40': 81.99626484218626, 'Car_3d/hard_R40': 79.50705406984453, 'Car_bev/easy_R40': 94.55883092569243, 'Car_bev/moderate_R40': 90.1516963279183, 'Car_bev/hard_R40': 87.99330601257056, 'Car_image/easy_R40': 98.13286744848779, 'Car_image/moderate_R40': 94.09085820720995, 'Car_image/hard_R40': 91.80455050234627}
```
