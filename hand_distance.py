from hand_detector import HandDetector
import time
import math
from tools import landmark_to_coords
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

FOCAL_LENGTH = config.getint("Camera", "FOCAL_LENGTH", fallback=900)
KNUCKLE_WIDTH = config.getfloat("Hand", "KNUCKLE_WIDTH")

def calculate_distance(result, mp_image, point_a = 5, point_b = 17, length = KNUCKLE_WIDTH):
    a = landmark_to_coords(result.hand_landmarks[0][point_a])
    b = landmark_to_coords(result.hand_landmarks[0][point_b])

    vector = (a - b) * 100
    
    dist_cm = math.hypot(*(vector / np.linalg.norm(vector) * length)[:2])
    dist_px = math.hypot((a[0] - b[0]) * mp_image.numpy_view().shape[1], (a[1] - b[1]) * mp_image.numpy_view().shape[0])

    return dist_cm * FOCAL_LENGTH / dist_px

if __name__ == "__main__":
    def callback(result, image, timestamp):
        if result.hand_landmarks: 
            print(f"{calculate_distance(result, image):.02f} cm")

            # pyautogui.moveTo(x, y)
        time.sleep(0.01)
        detector.detect_async()
            
    detector = HandDetector(callback, hands=1)

    detector.detect_async()

    while True: pass