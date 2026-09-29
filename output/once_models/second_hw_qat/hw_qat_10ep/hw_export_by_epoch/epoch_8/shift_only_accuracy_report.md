# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8722636623209898, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7801080147856305, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5274906483100555, 'AP_Vehicle/overall': 66.83370205408815, 'AP_Vehicle/0-30m': 81.22787014868003, 'AP_Vehicle/30-50m': 58.74599402032254, 'AP_Vehicle/50m-inf': 41.785069344972094, 'AP_Pedestrian/overall': 17.04299798866805, 'AP_Pedestrian/0-30m': 19.564993005205302, 'AP_Pedestrian/30-50m': 15.240003516903219, 'AP_Pedestrian/50m-inf': 11.123985988879975, 'AP_Cyclist/overall': 48.80649234454166, 'AP_Cyclist/0-30m': 61.88286838235455, 'AP_Cyclist/30-50m': 41.86871829653717, 'AP_Cyclist/50m-inf': 25.43075058113341, 'AP_mean/overall': 44.22773079576595, 'AP_mean/0-30m': 54.225243845413296, 'AP_mean/30-50m': 38.618238611254306, 'AP_mean/50m-inf': 26.113268638328492}
```
