# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8861971269837756, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7886518072555834, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5227539343500299, 'AP_Vehicle/overall': 65.80654648777634, 'AP_Vehicle/0-30m': 79.19428581313663, 'AP_Vehicle/30-50m': 57.416255636208255, 'AP_Vehicle/50m-inf': 40.837860937241686, 'AP_Pedestrian/overall': 19.442764769260773, 'AP_Pedestrian/0-30m': 21.678721168684763, 'AP_Pedestrian/30-50m': 16.86604282885044, 'AP_Pedestrian/50m-inf': 13.01282916888874, 'AP_Cyclist/overall': 48.73180153183593, 'AP_Cyclist/0-30m': 61.09905364582491, 'AP_Cyclist/30-50m': 41.34291569235975, 'AP_Cyclist/50m-inf': 27.109864855681785, 'AP_mean/overall': 44.66037092962435, 'AP_mean/0-30m': 53.9906868758821, 'AP_mean/30-50m': 38.541738052472816, 'AP_mean/50m-inf': 26.986851653937407}
```
