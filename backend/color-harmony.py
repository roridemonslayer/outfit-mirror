import numpy as np 
from PIL import Image
from segment import segment_image


mask = segment_image("images.jpeg")
image = Image.open("images.jpeg")#opens images
image_array = np.array(image) #this is the image in array form

print(image_array.shape)