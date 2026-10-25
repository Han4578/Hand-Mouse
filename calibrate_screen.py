import tkinter as tk
from PIL import Image, ImageTk
from hand_detector import HandDetector
from hand_distance import calculate_distance
from tools import landmark_to_coords
import time
import cv2
import numpy as np
import math
import configparser

calibrating = False
vectors = []
positions = []
distances = []

config = configparser.ConfigParser()
config.read("config.ini")

FOCAL_LENGTH = config.getint("Camera", "FOCAL_LENGTH", fallback=900)
INDEX_LENGTH = config.getfloat("Hand", "INDEX_LENGTH")

INDEX_TIP = 8
INDEX_KNUCKLE = 5

def detector_callback(result, mp_image, timestamp):

    update_display(mp_image)
    calibrate(result, mp_image)

    root.after(max(0, 33 - int(time.time() * 1000 - timestamp)), detector.detect_async)

def update_display(image):
    img_np = np.copy(image.numpy_view())

    cv2.circle(img_np, (0, 0), 20, (0, 0, 255), -1)

    img = Image.fromarray(img_np)
    
    w = display.winfo_width()
    h = display.winfo_height()
    img = img.resize((w, w * img.size[1] // img.size[0]))
    img.thumbnail((w, h))
    imgtk = ImageTk.PhotoImage(img)
    display.config(image=imgtk)
    display.img = imgtk

def calibrate(result, mp_image):
    global calibrating

    if not calibrating or not result.hand_landmarks: return

    a = landmark_to_coords(result.hand_landmarks[0][INDEX_KNUCKLE])
    b = landmark_to_coords(result.hand_landmarks[0][INDEX_TIP])


    matrix = np.array([
        [mp_image.numpy_view().shape[1], 0],
        [0, mp_image.numpy_view().shape[0]],
        [0, 0]
    ])

    d = calculate_distance(result, mp_image)
    v_px = (b - a) @ matrix

    v_cm = v_px * d / FOCAL_LENGTH
    v_cm = np.append(v_cm, math.sqrt(INDEX_LENGTH * INDEX_LENGTH - v_cm[0] ** 2 - v_cm[1] ** 2))

    vectors.append(v_cm)
    positions.append(a @ matrix)
    distances.append(d)

    if len(vectors) >= 10:
        v = np.average(vectors, axis=0)
        pos = np.average(positions, axis=0)
        dist = sum(distances) / 10

        config.set("Screen", "VECTOR", ', '.join(map(str, v.tolist())))
        config.set("Screen", "POSITION", ', '.join(map(str, pos.tolist())))
        config.set("Screen", "DISTANCE", str(dist))

        with open("config.ini", "w") as f:
            config.write(f)

        calibrating = False
        btn.config(state="normal", text="Start")

def start_calibrate():
    global calibrating

    btn.config(state="disabled")
    wait = tk.BooleanVar(value=False)

    for countdown in range(3, 0, -1):
        wait.set(False)
        btn.config(text=str(countdown))
        root.after(1000, wait.set, True)
        root.wait_variable(wait)

    calibrating = True
    btn.config(text="Calibrating...")
    
detector = HandDetector(detector_callback, hands=1, flip=True)

root = tk.Tk()
root.title("Point at the Top Left Corner of the Screen")

root.geometry(f"700x700+{(root.winfo_screenwidth() - 700) // 2}+{(root.winfo_screenheight() - 700) // 2}")
root.resizable(False, False)

display = tk.Label(root)
display.pack(fill="both", expand=True)

btn = tk.Button(root, text="Start", command=start_calibrate)
btn.pack(side="bottom", anchor="center")

root.after(20, detector.detect_async)

root.mainloop()