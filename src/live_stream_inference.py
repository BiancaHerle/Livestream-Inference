import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from threading import Thread, Barrier


class LiveStreamInference():
    def __init__(self, src, model, transforms):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        print(self.frame_height, self.frame_width)

        # Start the thread to read frames from the video stream
        self.status = False
        self.frame = None
        self.inference_result = None

        self.model = model
        self.transforms = transforms

    def update(self, barrier, scale, thread_name):
        # Read the next frame from the stream
        if self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
            dim = (int(self.frame_width * scale), int(self.frame_height * scale))
            self.frame = cv2.resize(self.frame, dim)
            self.run_inference()
            print(f"Thread {thread_name} done")
            barrier.wait()  

    def run_inference(self):
        self.frame = np.moveaxis(self.frame, -1, 0)
        image = self.transforms(torch.tensor(self.frame))
        outputs = self.model([image])[0]
        score_threshold = .8
        cars_with_boxes = [
            draw_bounding_boxes(torch.tensor(self.frame, dtype=torch.uint8), 
                                boxes=outputs['boxes'][outputs['scores'] > score_threshold], 
                                width=2)
        ]
        img = cars_with_boxes[0].numpy()
        img = np.moveaxis(img, 0, -1)
        self.inference_result = img

def parallel_inference(rtsp_stream_link):
    weights_resnet = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms_resnet = weights_resnet.transforms()
    model_resnet = fasterrcnn_resnet50_fpn(weights=weights_resnet, progress=False)  # GFLOPS=134.38
    model_resnet = model_resnet.eval()
    video_stream_widget_resnet = LiveStreamInference(src=rtsp_stream_link, model=model_resnet, transforms=transforms_resnet)

    weights_mbnet = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    transforms_mbnet= weights_mbnet.transforms()
    model_mbnet = fasterrcnn_mobilenet_v3_large_fpn(weights=weights_mbnet, progress=False)  # GFLOPS=4.49
    model_mbnet = model_mbnet.eval()
    video_stream_widget_mbnet = LiveStreamInference(src=rtsp_stream_link, model=model_mbnet, transforms=transforms_mbnet)
    
    while True:
        # Start the thread to read frames from the video stream
        barrier = Barrier(2 + 1)
        thread1 = Thread(target=video_stream_widget_resnet.update, args=(barrier, 0.7, "RN50"))
        thread2 = Thread(target=video_stream_widget_mbnet.update, args=(barrier, 0.7, "MBNETV3"))
        thread1.start()
        thread2.start()

        barrier.wait()
        result_stack = np.hstack((video_stream_widget_resnet.inference_result,
                                  video_stream_widget_mbnet.inference_result))
        cv2.imshow("ResNet50 vs MobilenetV3", result_stack)

        key = cv2.waitKey(1)
        if key == ord('q'):
            video_stream_widget_resnet.capture.release()
            video_stream_widget_mbnet.capture.release()
            cv2.destroyAllWindows()
            exit(1)

def main():
    rtsp_stream_link = 'http://82.78.94.9:82/cgi-bin/faststream.jpg?stream=full&fps=25&rand=COUNTER'
    parallel_inference(rtsp_stream_link)

if __name__ == '__main__':
    main()
