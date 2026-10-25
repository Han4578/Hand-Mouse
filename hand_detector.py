import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, HandLandmarksConnections, drawing_utils
import time
import numpy as np
import configparser

# !wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

config = configparser.ConfigParser()
config.read("config.ini")

class HandDetector:
    def __init__(self, callback, hands = 2, flip = False):
        """
        def callback(result, image, timestamp):
        """
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint("Camera", "WIDTH_PIXELS"))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint("Camera", "HEIGHT_PIXELS"))

        if not self.capture.isOpened():
            raise Exception("Camera not found")

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, 
            num_hands=hands,
            result_callback=callback
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.flip = flip

    def detect_async(self):
        success, frame = self.capture.read()
        if self.flip: frame = cv2.flip(frame, 1)

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            self.detector.detect_async(mp_image, int(time.time() * 1000))
        else:
            print("Couldn't capture frame")

    def draw_landmarks(self, mp_image, hand_landmarks):
        result = np.copy(mp_image.numpy_view())

        for hand in hand_landmarks.hand_landmarks:
            drawing_utils.draw_landmarks(
                result,
                hand,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )

        return result




    