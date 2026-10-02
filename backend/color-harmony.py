import numpy as np 
from PIL import Image
from segment import segment_image
from sklearn.cluster import KMeans
import colorsys
from itertools import combinations
from collections import Counter
from collections import defaultdict
IDEAL_HUE_GAPS = {"monochrome": 0, "analogous": 45, "triadic": 120, "complementary": 180}


def hue_distance(h1, h2): #this function calculates the distance between two hues, taking into account the circular nature of the hue space
    diff = abs(h1 - h2)
    if diff > 180:
        return 360 - diff
    return diff


def classify_distance(distance):
    best_cat = None
    small_gap = 999
    for name, val in IDEAL_HUE_GAPS.items():
        gap = abs(distance - val)
        if gap < small_gap:
            small_gap = gap
            best_cat = name
    return best_cat

def get_color_harmony(image_path):
    mask = segment_image(image_path)
    image = Image.open(image_path)
    image_array = np.array(image)

    clothing_pixels = image_array[mask]

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(clothing_pixels)

    hue = []
    for color in kmeans.cluster_centers_:
        r, g, b = color / 255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hue_degrees = h * 360
        hue.append(hue_degrees)

    categories = []
    category_distances = defaultdict(list)

    for h1, h2 in combinations(hue, 2):
        distance = hue_distance(h1, h2)
        category = classify_distance(distance)
        categories.append(category)
        category_distances[category].append(distance)

    counts = Counter(categories)

    top_count = counts.most_common(1)[0][1]
    tied_categories = [name for name, count in counts.items() if count == top_count]

    if len(tied_categories) == 1:
        winner = tied_categories[0]
    else:
        best_gap = 999
        winner = None
        for category in tied_categories:
            avg = sum(category_distances[category]) / len(category_distances[category])
            gap = abs(avg - IDEAL_HUE_GAPS[category])
            if gap < best_gap:
                best_gap = gap
                winner = category

    return winner

if __name__ == "__main__":
    result = get_color_harmony("images.jpeg")
    print(result)