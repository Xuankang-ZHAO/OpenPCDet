# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.9717066388599235, 'recall/rcnn_0.3': 0.9716371220020855, 'recall/roi_0.5': 0.9604449078901633, 'recall/rcnn_0.5': 0.9623913799096281, 'recall/roi_0.7': 0.7870698644421272, 'recall/rcnn_0.7': 0.8358011817865832, 'Car_aos/easy_R40': 98.06732595969757, 'Car_aos/moderate_R40': 93.6410682498653, 'Car_aos/hard_R40': 91.24788212101026, 'Car_3d/easy_R40': 91.30605165803347, 'Car_3d/moderate_R40': 81.71455607918074, 'Car_3d/hard_R40': 79.19706935509832, 'Car_bev/easy_R40': 94.67722574057751, 'Car_bev/moderate_R40': 89.71827923216836, 'Car_bev/hard_R40': 87.68161030511202, 'Car_image/easy_R40': 98.09068493377906, 'Car_image/moderate_R40': 93.86839710083217, 'Car_image/hard_R40': 91.59510669808715}
```
