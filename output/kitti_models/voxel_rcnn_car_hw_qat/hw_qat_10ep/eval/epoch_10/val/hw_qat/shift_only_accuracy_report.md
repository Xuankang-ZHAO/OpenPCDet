# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9709419534237053, 'recall/rcnn_0.3': 0.9709419534237053, 'recall/roi_0.5': 0.962113312478276, 'recall/rcnn_0.5': 0.9635731664928745, 'recall/roi_0.7': 0.79471671880431, 'recall/rcnn_0.7': 0.8469933958985054, 'Car_aos/easy_R40': 98.42105994505613, 'Car_aos/moderate_R40': 94.12626024661378, 'Car_aos/hard_R40': 91.67587894431782, 'Car_3d/easy_R40': 91.18055183262935, 'Car_3d/moderate_R40': 81.97031527635336, 'Car_3d/hard_R40': 79.51692521455487, 'Car_bev/easy_R40': 95.1214802599134, 'Car_bev/moderate_R40': 90.5947286364276, 'Car_bev/hard_R40': 88.31968251104678, 'Car_image/easy_R40': 98.44628293091408, 'Car_image/moderate_R40': 94.32805294030628, 'Car_image/hard_R40': 91.98850291281323}
```
