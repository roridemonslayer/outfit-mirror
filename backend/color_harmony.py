import numpy as np
from PIL import Image
from segment import segment_labels, NOT_OUTFIT
from sklearn.cluster import KMeans
import colorsys
from itertools import combinations
from collections import Counter
from collections import defaultdict
IDEAL_HUE_GAPS = {"monochrome": 0, "analogous": 45, "triadic": 120, "complementary": 180}

HARMONY_READS = {
    "monochrome": "one color family in different shades, which reads calm and intentional",
    "analogous": "neighbors on the color wheel, which blend smoothly",
    "triadic": "spaced about a third of the wheel apart, which reads bold but balanced",
    "complementary": "opposites on the color wheel, which makes each one pop",
}

# upper hue limits (degrees) for naming colors in the explanation
HUE_NAMES = [(15, "red"), (40, "orange"), (65, "yellow"), (160, "green"), (195, "teal"),
             (255, "blue"), (290, "purple"), (335, "pink"), (361, "red")]


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


def is_neutral(s, v):
    """black, white, gray and very muted colors have no real hue, so they don't vote on harmony"""
    return s < 0.2 or v < 0.2 or (s < 0.3 and v > 0.85)


def color_name(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if v < 0.2:
        return "black"
    if s < 0.2:
        return "white" if v > 0.85 else "gray" if v > 0.35 else "charcoal"
    if s < 0.3 and v > 0.85:
        return "cream"
    name = next(n for limit, n in HUE_NAMES if h * 360 < limit)
    if name == "orange" and v < 0.6:
        return "brown"
    if name == "yellow" and v < 0.6:
        return "olive"
    if name in ("red", "pink") and v < 0.5:
        return "burgundy"
    if name == "blue" and s < 0.6:
        return "denim blue"
    if name == "blue" and v < 0.4:
        return "navy"
    if v < 0.45:
        return "dark " + name
    if s < 0.45 and v > 0.7:
        return "pastel " + name
    return name


def pick_harmony(categories, category_distances):
    """majority vote; on a tie, the category whose average distance is closest to its ideal gap wins"""
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


def analyze_colors(image_array, labels):
    """dominant outfit colors, the harmony between them, a 0-100 score and a plain-language reason"""
    mask = ~np.isin(labels, list(NOT_OUTFIT)) #outfit pixels only, so skin tone never counts
    clothing_pixels = image_array[mask].astype(float)
    if len(clothing_pixels) < 500:
        return None
    if len(clothing_pixels) > 20000: #a sample is plenty for 5 clusters and much faster
        rng = np.random.default_rng(42)
        clothing_pixels = clothing_pixels[rng.choice(len(clothing_pixels), 20000, replace=False)]

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=4)
    kmeans.fit(clothing_pixels)
    shares = np.bincount(kmeans.labels_, minlength=5) / len(kmeans.labels_)

    swatches = []
    for color, share in sorted(zip(kmeans.cluster_centers_, shares), key=lambda c: -c[1]):
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        swatches.append({"hex": "#%02x%02x%02x" % (int(r), int(g), int(b)), "share": round(float(share), 3),
                         "name": color_name(r, g, b), "hue": round(float(h) * 360, 1), "neutral": bool(is_neutral(s, v))})

    chromatic = [c for c in swatches if not c["neutral"] and c["share"] >= 0.04]
    neutral_share = sum(c["share"] for c in swatches if c["neutral"])
    neutral_names = list(dict.fromkeys(c["name"] for c in swatches if c["neutral"]))

    families = [] #distinct hue families, named by their biggest swatch; more than 3 starts to feel busy
    colors = []
    for c in chromatic:
        if all(hue_distance(c["hue"], f) >= 25 for f in families):
            families.append(c["hue"])
            colors.append(c["name"])

    if len(families) < 2:
        if chromatic:
            harmony, score = "monochrome", 90
            reason = f"One color family, {colors[0]}, anchored by {', '.join(neutral_names) or 'its own shades'}. Hard to get wrong."
        else:
            harmony, score = "neutral", 92
            reason = f"An all-neutral palette ({', '.join(neutral_names)}). Clean and easy to wear, so texture and shape do the talking."
        return {"harmony": harmony, "score": score, "reason": reason, "swatches": swatches}

    categories = []
    category_distances = defaultdict(list)
    fit, weight = 0.0, 0.0
    for c1, c2 in combinations(chromatic, 2):
        distance = hue_distance(c1["hue"], c2["hue"])
        category = classify_distance(distance)
        categories.append(category)
        category_distances[category].append(distance)
        w = c1["share"] * c2["share"] #bigger color areas matter more
        fit += w * max(0.0, 1 - abs(distance - IDEAL_HUE_GAPS[category]) / 40) #how cleanly the pair lands on a harmony
        weight += w
    harmony = pick_harmony(categories, category_distances)

    score = 55 + 40 * fit / weight + min(neutral_share, 0.5) * 10 - max(0, len(families) - 3) * 8
    score = int(max(30, min(98, score)))

    pair = " and ".join(colors[:3])
    reason = f"{pair[0].upper() + pair[1:]}: {HARMONY_READS[harmony]}."
    if neutral_names:
        reason += f" The {' and '.join(neutral_names[:2])} give the eye somewhere to rest."
    if score < 60:
        reason += " The colors land between the classic pairings, so they can read a little unplanned."
    return {"harmony": harmony, "score": score, "reason": reason, "swatches": swatches}


def get_color_harmony(image_path):
    image = Image.open(image_path).convert("RGB")
    return analyze_colors(np.array(image), segment_labels(image))["harmony"]

if __name__ == "__main__":
    image = Image.open("images.jpeg").convert("RGB")
    result = analyze_colors(np.array(image), segment_labels(image))
    print(result["harmony"], result["score"], result["reason"])
    for s in result["swatches"]:
        print(" ", s)
