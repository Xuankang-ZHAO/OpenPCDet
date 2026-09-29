# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8876469156023816, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7854312844463136, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5177516102614046, 'AP_Vehicle/overall': 65.79530090440186, 'AP_Vehicle/0-30m': 79.15026865788957, 'AP_Vehicle/30-50m': 57.144349359508304, 'AP_Vehicle/50m-inf': 41.36198417179699, 'AP_Pedestrian/overall': 18.24237056494596, 'AP_Pedestrian/0-30m': 20.12431363594013, 'AP_Pedestrian/30-50m': 17.08597083891021, 'AP_Pedestrian/50m-inf': 12.75207342218426, 'AP_Cyclist/overall': 47.655382206989664, 'AP_Cyclist/0-30m': 60.318462916748985, 'AP_Cyclist/30-50m': 40.231953288231594, 'AP_Cyclist/50m-inf': 25.4563047339537, 'AP_mean/overall': 43.89768455877916, 'AP_mean/0-30m': 53.19768173685956, 'AP_mean/30-50m': 38.1540911622167, 'AP_mean/50m-inf': 26.523454109311654}
```
