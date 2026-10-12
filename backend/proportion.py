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
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
results = pose_landmarker.detect(mp_image)
landmarks = results.pose_landmarks[0]
hip_row = int(landmarks[23].y * image_array.shape[0])
mask = segment_image("images.jpeg")
rows_with_clothing = np.any(mask.numpy(),axis =1)


clothing_rows = np.nonzero(rows_with_clothing)[0]
outfit_top = clothing_rows[0]
outfit_bottom = clothing_rows[-1]
print(outfit_top, outfit_bottom, hip_row)
mask_array = mask.numpy()
mask_pic = (mask_array * 225).astype(np.uint8)
Image.fromarray(mask_pic).save("mask_check.png")