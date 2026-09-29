# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8852896257110604, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7923039465238274, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5300250116204431, 'AP_Vehicle/overall': 67.04821873775131, 'AP_Vehicle/0-30m': 81.73626587823405, 'AP_Vehicle/30-50m': 58.49990006630743, 'AP_Vehicle/50m-inf': 41.87721766731592, 'AP_Pedestrian/overall': 16.989297242477015, 'AP_Pedestrian/0-30m': 18.931771317642024, 'AP_Pedestrian/30-50m': 14.939440008811692, 'AP_Pedestrian/50m-inf': 12.248051955951915, 'AP_Cyclist/overall': 48.33559710264275, 'AP_Cyclist/0-30m': 60.02007515845631, 'AP_Cyclist/30-50m': 42.4445854424428, 'AP_Cyclist/50m-inf': 26.280246812121185, 'AP_mean/overall': 44.1243710276237, 'AP_mean/0-30m': 53.562704118110794, 'AP_mean/30-50m': 38.62797517252064, 'AP_mean/50m-inf': 26.80183881179634}
```
