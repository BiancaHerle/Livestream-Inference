# Livestream-Inference
This Project runs object detection and tracking using multiple models in parallel on a live video stream.

The input video stream has 20 FPS and a resolution of 600x800 pixels.
The inference in based on pytorch framework, importing pre-trained FasterRCNN_ResNet50_FPN and FasterRCNN_MobileNet_V3_Large_FPN.
The output video stream has bounding boxes overlayed.
GFLOPS and frame rate differ significantly.
Using barriers to synchronize the inference threads.
