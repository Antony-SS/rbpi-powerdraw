from experiments.core.experiment import BaseExperiment
import os
import time
import cv2
import numpy as np
import picamera2

class ReadAndSave(BaseExperiment):

    NAME = "read_and_save"

    def __init__(self, camera: picamera2.Picamera2, fps: int, size: tuple[int, int], output_dir: str, experiment_duration: int = 60):
        self.fps = fps
        self.size = size
        self.output_dir = output_dir
        self.experiment_duration = experiment_duration
        super().__init__(self.NAME, camera, experiment_duration)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def configure_camera(self):
        time_per_frame = 1000000 // self.fps
        config = self.camera.create_video_configuration(
            main={
                "size": self.size,
                "format": "RGB888",
        
            },
            controls={
                "FrameDurationLimits": (time_per_frame, time_per_frame),
            },
        )
        self.camera.configure(config)
        self.camera.start()


    def process_image(self, image: np.ndarray):
       # save image to output directory
       image_path = os.path.join(self.output_dir, f"{time.time()}.jpg")
       cv2.imwrite(image_path, image)