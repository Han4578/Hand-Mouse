import tkinter as tk
from PIL import Image, ImageTk
from hand_detector import HandDetector
import time


def update_display(result, image, timestamp):
    img = Image.fromarray(detector.draw_landmarks(image, result))
    
    w = display.winfo_width()
    h = display.winfo_height()
    img = img.resize((w, w * img.size[1] // img.size[0]))
    img.thumbnail((w, h))
    imgtk = ImageTk.PhotoImage(img)
    display.config(image=imgtk)
    display.img = imgtk
    
    display.after(max(0, 33 - int(time.time() * 1000 - timestamp)), next_frame)

def next_frame():
    detector.detect_async()

detector = HandDetector(update_display)

root = tk.Tk()

root.geometry(f"500x500+{(root.winfo_screenwidth() - 500) // 2}+{(root.winfo_screenheight() - 500) // 2}")

display = tk.Label(root)
display.pack(fill="both", expand=True)

display.after(20, next_frame)

root.mainloop()