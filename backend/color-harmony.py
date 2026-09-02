import numpy as np 
from PIL import Image
from segment import segment_image
from sklearn.cluster import KMeans
import colorsys


mask = segment_image("images.jpeg")
image = Image.open("images.jpeg")#opens images
image_array = np.array(image) #this is the image in array form

print(image_array.shape)

clothing_pixels = image_array[mask] #this is the pixels of the outfit
print(clothing_pixels.shape)

kmeans = KMeans(n_clusters = 5, random_state = 42)#this means that we want to find 5 clusters in the data, and the random state is set to 42 for reproducibility
kmeans.fit(clothing_pixels) #this means that we are fitting the kmeans model to the clothing pixels
print(kmeans.cluster_centers_) #this is the center of the clusters, which is the average color of the pixels in each cluster

for color in kmeans.cluster_centers_: 
    r,g ,b = color / 225
    h,s,v = colorsys.rgb_to_hsv(r,g,b) #wehre the conversin happens
    hue_degrees = h * 360
    print(hue_degrees)