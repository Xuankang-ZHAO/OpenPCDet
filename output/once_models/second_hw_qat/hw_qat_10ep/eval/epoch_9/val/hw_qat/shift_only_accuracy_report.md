# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8677040217800306, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7735009628367162, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5154828570796166, 'AP_Vehicle/overall': 66.29128113669543, 'AP_Vehicle/0-30m': 79.26797998955263, 'AP_Vehicle/30-50m': 58.03652304560531, 'AP_Vehicle/50m-inf': 41.80468406310744, 'AP_Pedestrian/overall': 15.453202292215936, 'AP_Pedestrian/0-30m': 16.91104148203703, 'AP_Pedestrian/30-50m': 14.57659854375205, 'AP_Pedestrian/50m-inf': 10.899675872739785, 'AP_Cyclist/overall': 47.82454868704713, 'AP_Cyclist/0-30m': 59.7338690861292, 'AP_Cyclist/30-50m': 41.841350183433356, 'AP_Cyclist/50m-inf': 24.74692971802096, 'AP_mean/overall': 43.189677371986164, 'AP_mean/0-30m': 51.970963519239625, 'AP_mean/30-50m': 38.151490590930244, 'AP_mean/50m-inf': 25.817096551289396}
```
