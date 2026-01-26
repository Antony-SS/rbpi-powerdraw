import picamera2
import system


class BaseExperiment:
    def __init__(self, name: str, camera: picamera2.Picamera2):
        self.name = name
        self.camera = camera

    def run_experiment(self):
        while True:

    def process_image

    