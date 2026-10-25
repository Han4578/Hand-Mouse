from hand_detector import HandDetector
from hand_distance import calculate_distance
from tools import landmark_to_coords
import configparser
import numpy as np
import pyautogui
import tkinter as tk
import time
import math
from collections import deque

SCREEN_W, SCREEN_H = pyautogui.size()

config = configparser.ConfigParser()
config.read("config.ini")

REFERENCE_VECTOR = np.array([float(n) for n in config.get("Screen", "VECTOR").split(", ")])
REFERENCE_POSITION = np.array([float(n) for n in config.get("Screen", "POSITION").split(", ")])
REFERENCE_DISTANCE = config.getfloat("Screen", "DISTANCE")
EXTRA_DISTANCE = config.getfloat("Camera", "EXTRA_DISTANCE")
FOCAL_LENGTH = config.getint("Camera", "FOCAL_LENGTH", fallback=900)
PPCM = config.getint("Screen", "WIDTH_PIXELS", fallback=1920) / config.getfloat("Screen", "WIDTH_CM", fallback=40)
INDEX_LENGTH = config.getfloat("Hand", "INDEX_LENGTH")

INDEX_TIP = 8
INDEX_KNUCKLE = 5

REFERENCE_COORDS = -(REFERENCE_VECTOR / abs(REFERENCE_VECTOR[2]))[:2] * (REFERENCE_DISTANCE + EXTRA_DISTANCE) * PPCM
print(REFERENCE_COORDS)
pointers = deque(maxlen=5)

def callback(result, mp_image, timestamp):

    if result.hand_landmarks:
        a = landmark_to_coords(result.hand_landmarks[0][INDEX_KNUCKLE])
        b = landmark_to_coords(result.hand_landmarks[0][INDEX_TIP])
        c = landmark_to_coords(result.hand_landmarks[0][INDEX_KNUCKLE])

        matrix = np.array([
            [mp_image.numpy_view().shape[1], 0],
            [0, mp_image.numpy_view().shape[0]],
            [0, 0]
        ])

        position = c @ matrix
        distance = calculate_distance(result, mp_image)

        v_px = (b - a) @ matrix

        vector = v_px * distance / FOCAL_LENGTH
        z = math.sqrt(max(0.1, INDEX_LENGTH * INDEX_LENGTH - vector[0] ** 2 - vector[1] ** 2))

        pointers.append((vector / z)[:2] * (distance + EXTRA_DISTANCE) * PPCM)

        screen_coords = np.copy(REFERENCE_COORDS)

        screen_coords += ((position - REFERENCE_POSITION) * REFERENCE_DISTANCE) / FOCAL_LENGTH * PPCM

        # screen_coords -= (REFERENCE_VECTOR / abs(REFERENCE_VECTOR[2]))[:2] * (REFERENCE_DISTANCE + EXTRA_DISTANCE) * PPCM
        screen_coords += np.sum(pointers, axis=0) / len(pointers)
        if 0 <= screen_coords[0] <= SCREEN_W and 0 <= screen_coords[1] <= SCREEN_H:
            pyautogui.moveTo(int(screen_coords[0]), int(screen_coords[1]))
        else:
            print(screen_coords)
            pointers.clear()

    root.after(max(0, 33 - int(time.time() * 1000 - timestamp)), detector.detect_async)
    
if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    
    detector = HandDetector(callback, hands=1, flip=True)

    root = tk.Tk()
    root.withdraw()

    detector.detect_async()

    root.mainloop() 