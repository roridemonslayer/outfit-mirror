"""outfit mirror pipeline: photo in, explainable outfit feedback out.

    python main.py photo.jpg                      detect the aesthetic and score the outfit
    python main.py photo.jpg --target "old money"  also compare against a target aesthetic
"""
import time
import numpy as np
from PIL import Image, ImageOps

from segment import segment_labels
from color_harmony import analyze_colors
from proportion import detect_pose, keep_main_person, analyze_proportion

MAX_SIDE = 768  # big enough for the models, small enough to stay fast on a laptop


def prepare(image):
    """fix phone rotation, drop transparency, and shrink big photos"""
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = MAX_SIDE / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def crop_to_person(image, landmarks):
    """crop to the main person so other people's clothes in the background don't count.
    returns the crop, the crop box, and the landmarks moved into the crop's coordinates"""
    if landmarks is None:
        return image, (0, 0, image.width, image.height), None
    visible = [l for l in landmarks if l["visibility"] > 0.3] or landmarks
    xs = [l["x"] * image.width for l in visible]
    ys = [l["y"] * image.height for l in visible]
    body_h = max(ys) - min(ys)
    pad_x = max(body_h * 0.18, (max(xs) - min(xs)) * 0.35) #room for sleeves, wide legs and bags
    box = (int(max(0, min(xs) - pad_x)), int(max(0, min(ys) - body_h * 0.15)),
           int(min(image.width, max(xs) + pad_x)), int(min(image.height, max(ys) + body_h * 0.08)))
    crop = image.crop(box)
    moved = [{"x": (l["x"] * image.width - box[0]) / crop.width,
              "y": (l["y"] * image.height - box[1]) / crop.height,
              "visibility": l["visibility"]} for l in landmarks]
    return crop, box, moved


def analyze(image, target=None, with_vibe=True):
    timings = {}
    t = time.time()
    image = prepare(image)
    landmarks = detect_pose(np.array(image))
    crop, box, crop_landmarks = crop_to_person(image, landmarks)
    crop_array = np.array(crop)
    timings["pose"] = time.time() - t

    t = time.time()
    labels = keep_main_person(segment_labels(crop), crop_landmarks) #drop clothing that belongs to anyone else in the shot
    timings["segment"] = time.time() - t

    t = time.time()
    colors = analyze_colors(crop_array, labels)
    proportion = analyze_proportion(crop_array, labels, crop_landmarks)
    timings["scores"] = time.time() - t

    vibe = None
    if with_vibe:
        from vibe_match import analyze_vibe #CLIP is the slowest model to load, so only load it when needed
        t = time.time()
        vibe = analyze_vibe(crop, labels, target)
        timings["vibe"] = time.time() - t

    from feedback import build_feedback
    return {
        "image": image,
        "box": list(box),
        "landmarks": landmarks,
        "labels": labels,
        "colors": colors,
        "proportion": proportion,
        "vibe": vibe,
        "feedback": build_feedback(colors, proportion, vibe),
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--target", help="aesthetic to compare against, e.g. 'old money'")
    args = ap.parse_args()
    result = analyze(Image.open(args.photo), args.target)
    f = result["feedback"]
    print(f"\n{f['headline']}\n")
    for section in f["sections"]:
        print(f"{section['title']}: {section['score'] if section['score'] is not None else '-'}")
        print(f"  {section['reason']}")
    for tip in f["tips"]:
        print(f"  tip: {tip}")
    print(f"\ntimings: {result['timings']}")
