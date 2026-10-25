import numpy as np

def landmark_to_coords(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])
