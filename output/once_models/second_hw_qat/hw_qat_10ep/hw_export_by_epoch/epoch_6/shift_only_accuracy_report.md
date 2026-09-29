# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8737798534717457, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7759357223488789, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.509705836782576, 'AP_Vehicle/overall': 64.47325250030046, 'AP_Vehicle/0-30m': 78.75848037640009, 'AP_Vehicle/30-50m': 56.22071237479502, 'AP_Vehicle/50m-inf': 39.936720897552824, 'AP_Pedestrian/overall': 17.264996582911518, 'AP_Pedestrian/0-30m': 19.04471168950211, 'AP_Pedestrian/30-50m': 15.853235486457418, 'AP_Pedestrian/50m-inf': 11.696665984002468, 'AP_Cyclist/overall': 47.64919410026568, 'AP_Cyclist/0-30m': 59.88329160609383, 'AP_Cyclist/30-50m': 40.98505129488961, 'AP_Cyclist/50m-inf': 24.913190157420694, 'AP_mean/overall': 43.12914772782589, 'AP_mean/0-30m': 52.56216122399868, 'AP_mean/30-50m': 37.68633305204735, 'AP_mean/50m-inf': 25.515525679658662}
```
