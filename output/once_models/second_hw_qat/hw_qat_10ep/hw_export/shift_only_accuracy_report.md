# SECOND HW-QAT HW Reference Evaluation

This report records OpenPCDet metrics from the HW-equivalent integer reference path.
The 3D backbone reference uses int8-valued sparse features/weights, integer bias,
per-channel shift-only requantization, clamp, and ReLU before dequantizing back to FP32.
Exact bias/shift checks are in requant_shift_stats.csv.

```text
{'recall/roi_0.3': 0.0, 'recall/rcnn_0.3': 0.8690431395117201, 'recall/roi_0.5': 0.0, 'recall/rcnn_0.5': 0.7755815755107461, 'recall/roi_0.7': 0.0, 'recall/rcnn_0.7': 0.5185816419132783, 'AP_Vehicle/overall': 66.46416832128304, 'AP_Vehicle/0-30m': 80.85573016932031, 'AP_Vehicle/30-50m': 58.15824512096888, 'AP_Vehicle/50m-inf': 41.93436712508107, 'AP_Pedestrian/overall': 15.585865676800148, 'AP_Pedestrian/0-30m': 17.12805876839023, 'AP_Pedestrian/30-50m': 14.973176505574271, 'AP_Pedestrian/50m-inf': 10.849827521814314, 'AP_Cyclist/overall': 48.35338946086883, 'AP_Cyclist/0-30m': 60.23038240702494, 'AP_Cyclist/30-50m': 42.27492695991371, 'AP_Cyclist/50m-inf': 25.146900532911843, 'AP_mean/overall': 43.46780781965068, 'AP_mean/0-30m': 52.73805711491183, 'AP_mean/30-50m': 38.46878286215229, 'AP_mean/50m-inf': 25.977031726602405}
```
