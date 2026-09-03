import numpy as np 
from PIL import Image
from segment import segment_image
from sklearn.cluster import KMeans
import colorsys
from itertools import combinations
from collections import Counter
from collections import defaultdict

mask = segment_image("images.jpeg")
image = Image.open("images.jpeg")#opens images
image_array = np.array(image) #this is the image in array form

print(image_array.shape)

clothing_pixels = image_array[mask] #this is the pixels of the outfit
print(clothing_pixels.shape)

kmeans = KMeans(n_clusters = 5, random_state = 42)#this means that we want to find 5 clusters in the data, and the random state is set to 42 for reproducibility
kmeans.fit(clothing_pixels) #this means that we are fitting the kmeans model to the clothing pixels
print(kmeans.cluster_centers_) #this is the center of the clusters, which is the average color of the pixels in each cluster

hue = [] 

for color in kmeans.cluster_centers_: 
    r,g ,b = color / 255
    h,s,v = colorsys.rgb_to_hsv(r,g,b) #wehre the conversin happens
    hue_degrees = h * 360
    hue.append(hue_degrees)


def hue_distance(h1, h2): #this function calculates the distance between two hues, taking into account the circular nature of the hue space
    diff = abs(h1-h2)
    if diff > 180:
        return 360 - diff
    return diff





def classify_distance(distance):
    ideas = {"monochrome":0, "analogous":45, "triadic":120, "complementary":180}
    best_cat = None
    small_gap = 999 #this means that we are setting the initial value of the smallest gap to a large number, so that any distance will be smaller than it
    for name, val in ideas.items():
        gap = abs(distance - val ) 
        if gap <small_gap: 
            small_gap = gap 
            best_cat = name
    return best_cat  

categories = [] 
category_distances = defaultdict(list)

for h1, h2 in combinations(hue,2):
    distance = hue_distance(h1, h2)
    category = classify_distance(distance)
    print(distance,category)
    categories.append(category)
    category_distances[category].append(distance)
counts = Counter(categories)
print(counts)
print(category_distances)