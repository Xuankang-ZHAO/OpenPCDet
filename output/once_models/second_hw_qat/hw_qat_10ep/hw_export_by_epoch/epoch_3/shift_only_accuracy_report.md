# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8798003497200027, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.786427322428562, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5253547001925674, 'AP_Vehicle/overall': 66.7972474192136, 'AP_Vehicle/0-30m': 81.50643959270997, 'AP_Vehicle/30-50m': 58.499461765683044, 'AP_Vehicle/50m-inf': 41.378817067406196, 'AP_Pedestrian/overall': 17.257701210480768, 'AP_Pedestrian/0-30m': 19.058687430532938, 'AP_Pedestrian/30-50m': 15.71028761176079, 'AP_Pedestrian/50m-inf': 12.246831783898461, 'AP_Cyclist/overall': 49.812827628653565, 'AP_Cyclist/0-30m': 62.28951677134774, 'AP_Cyclist/30-50m': 43.017983371463195, 'AP_Cyclist/50m-inf': 26.939095714062834, 'AP_mean/overall': 44.62259208611598, 'AP_mean/0-30m': 54.28488126486355, 'AP_mean/30-50m': 39.07591091630235, 'AP_mean/50m-inf': 26.854914855122498}
```
