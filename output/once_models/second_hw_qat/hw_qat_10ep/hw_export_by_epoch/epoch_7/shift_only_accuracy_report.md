# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.870271586356493, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7738993780296155, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5136457203568029, 'AP_Vehicle/overall': 66.31416980900988, 'AP_Vehicle/0-30m': 80.792123469278, 'AP_Vehicle/30-50m': 57.87692982787824, 'AP_Vehicle/50m-inf': 41.74403260214401, 'AP_Pedestrian/overall': 15.37749823440326, 'AP_Pedestrian/0-30m': 17.759635148159283, 'AP_Pedestrian/30-50m': 13.573717081415168, 'AP_Pedestrian/50m-inf': 10.295933041327816, 'AP_Cyclist/overall': 47.35563728683802, 'AP_Cyclist/0-30m': 59.373065012295726, 'AP_Cyclist/30-50m': 41.35923184406809, 'AP_Cyclist/50m-inf': 24.31077809334495, 'AP_mean/overall': 43.01576844341705, 'AP_mean/0-30m': 52.641607876577666, 'AP_mean/30-50m': 37.603292917787165, 'AP_mean/50m-inf': 25.45024791227226}
```
