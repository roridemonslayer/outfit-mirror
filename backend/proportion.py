import mediapipe as mp 
import numpy as np 
from PIL import Image 
from segment import segment_image 

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)
pose_landmarker = PoseLandmarker.create_from_options(options) 

image = Image.open("images.jpeg")
image_array = np.array(image) #opens photo and converts it to numpy array u
print(image_array.shape)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
results = pose_landmarker.detect(mp_image)
print(results.pose_landmarks)