"""outfit mirror API + mirror UI.

    uvicorn server:app --port 8000     then open http://localhost:8000
"""
import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from main import analyze
from segment import UPPER, LOWER, ONE_PIECE, SHOE, NOT_OUTFIT
from proportion import UPPER_PARTS
from vibe_match import NAMES  # importing loads CLIP now, so the first scan isn't slow

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
MAX_UPLOAD = 15 * 1024 * 1024

# overlay colors for the garment mask (RGBA)
GROUP_COLORS = [  # kept see-through so the outfit still shows underneath
    (UPPER | UPPER_PARTS, (200, 255, 61, 95)),   # tops: lime
    (LOWER, (140, 155, 255, 105)),                # bottoms: periwinkle
    (ONE_PIECE, (255, 122, 200, 105)),            # dresses and jumpsuits: pink
    ({SHOE}, (255, 190, 80, 140)),                # shoes: amber
]

app = FastAPI(title="outfit mirror")


def overlay_png(labels):
    """garment mask as a transparent PNG, sized to the person crop"""
    rgba = np.zeros((*labels.shape, 4), np.uint8)
    accessories = ~np.isin(labels, list(NOT_OUTFIT))
    rgba[accessories] = (255, 255, 255, 70)
    for ids, color in GROUP_COLORS:
        rgba[np.isin(labels, list(ids))] = color
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/api/aesthetics")
def aesthetics():
    return {"aesthetics": NAMES}


@app.post("/api/analyze")
def analyze_photo(photo: UploadFile = File(...), target: str = Form("")):
    data = photo.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That photo is over 15 MB. Try a smaller one.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't an image I can read. Try a JPG or PNG.")
    if target and target not in NAMES:
        raise HTTPException(400, f"Unknown aesthetic. Pick one of: {', '.join(NAMES)}")

    result = analyze(image, target or None)
    if result["landmarks"] is None:
        raise HTTPException(422, "Couldn't find a person in the photo. Try a full-length shot with the whole outfit in frame.")

    image = result["image"]
    return {
        "size": [image.width, image.height],
        "box": result["box"],
        "landmarks": result["landmarks"],
        "overlay": overlay_png(result["labels"]),
        "swatches": result["colors"]["swatches"] if result["colors"] else [],
        "feedback": result["feedback"],
        "vibe": result["vibe"],
        "timings": result["timings"],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")
