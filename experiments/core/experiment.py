import picamera2
import time
import numpy as np

class BaseExperiment:
    def __init__(self, name: str, camera: picamera2.Picamera2, experiment_duration: int):
        self.name = name
        self.camera = camera
        self.experiment_duration = experiment_duration # in seconds
        self.configure_camera()

    def run_experiment(self):
        start_time = time.time()
        while time.time() - start_time < self.experiment_duration:
            image = self.camera.capture_array("main", wait=True) # wait = True blocks until image is available
            if image is not None:
                self.process_image(image)
                print(f"Processed image {time.time() - start_time} seconds")
                
        self.unconfigure_camera()

    def process_image(self, image: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")

    def configure_camera(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def unconfigure_camera(self):
        self.camera.stop()
        self.camera.close()
        print(f"Unconfigured camera {self.name}")
    