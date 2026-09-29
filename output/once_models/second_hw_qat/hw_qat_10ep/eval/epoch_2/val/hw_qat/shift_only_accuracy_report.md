# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8761039420969919, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7791119768033821, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5117311140131477, 'AP_Vehicle/overall': 66.43466657272573, 'AP_Vehicle/0-30m': 81.50786949441738, 'AP_Vehicle/30-50m': 57.601410010326724, 'AP_Vehicle/50m-inf': 40.29696575766457, 'AP_Pedestrian/overall': 16.59720040270542, 'AP_Pedestrian/0-30m': 17.983367246678874, 'AP_Pedestrian/30-50m': 15.002559278837623, 'AP_Pedestrian/50m-inf': 11.420753089081279, 'AP_Cyclist/overall': 47.02086833723691, 'AP_Cyclist/0-30m': 59.72200739072541, 'AP_Cyclist/30-50m': 40.01450830645615, 'AP_Cyclist/50m-inf': 24.014255147839922, 'AP_mean/overall': 43.350911770889354, 'AP_mean/0-30m': 53.07108137727389, 'AP_mean/30-50m': 37.5394925318735, 'AP_mean/50m-inf': 25.24399133152859}
```
