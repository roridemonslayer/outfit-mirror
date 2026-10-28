import os
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from segment import segment_labels, UPPER, LOWER, ONE_PIECE, BELT, SHOE

UPPER_PARTS = {28, 29, 30, 31, 32, 34}  # hood, collar, lapel, epaulette, sleeve, neckline belong to the top
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 11, 12, 23, 24, 27, 28

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), "pose_landmarker.task")),
    running_mode=VisionRunningMode.IMAGE
)
pose_landmarker = PoseLandmarker.create_from_options(options)


# skeleton segments used to outline the main person: shoulders, arms, torso sides, hips, legs
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


def detect_pose(image_array):
    """the main person's 33 body landmarks as dicts with x, y in 0-1 and a visibility score, or None if nobody is found"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_array)) #opens photo as a mediapipe image
    results = pose_landmarker.detect(mp_image)
    if not results.pose_landmarks:
        return None
    return [{"x": l.x, "y": l.y, "visibility": l.visibility} for l in results.pose_landmarks[0]]


def keep_main_person(labels, landmarks):
    """zero out clothing that belongs to other people in the shot.
    keeps garment blobs that touch the main person's skeleton, trimmed to a corridor around it"""
    if landmarks is None:
        return labels
    height, width = labels.shape
    pts = [(l["x"] * width, l["y"] * height) for l in landmarks]
    shoulder_width = np.hypot(pts[11][0] - pts[12][0], pts[11][1] - pts[12][1])
    torso = np.hypot((pts[11][0] + pts[12][0] - pts[23][0] - pts[24][0]) / 2, (pts[11][1] + pts[12][1] - pts[23][1] - pts[24][1]) / 2)
    reach = max(shoulder_width, torso * 0.6)

    bones = Image.new("L", (width, height), 0) #thin skeleton, used to pick which blobs are theirs
    corridor = Image.new("L", (width, height), 0) #wide band around it, so loose clothes still fit
    thin, wide = ImageDraw.Draw(bones), ImageDraw.Draw(corridor)
    torso_poly = [pts[11], pts[12], pts[24], pts[23]]
    thin.polygon(torso_poly, fill=1)
    wide.polygon(torso_poly, fill=1)
    for a, b in BONES:
        if min(landmarks[a]["visibility"], landmarks[b]["visibility"]) > 0.2:
            thin.line([pts[a], pts[b]], fill=1, width=max(3, int(reach * 0.15)))
            wide.line([pts[a], pts[b]], fill=1, width=int(reach * 1.6))
    for p in (pts[15], pts[16], pts[27], pts[28]): #round ends at hands and feet
        wide.ellipse([p[0] - reach * 0.8, p[1] - reach * 0.8, p[0] + reach * 0.8, p[1] + reach * 0.8], fill=1)
    bones, corridor = np.array(bones, bool), np.array(corridor, bool)

    blobs, _ = ndimage.label(labels != 0)
    theirs = np.unique(blobs[bones & (blobs > 0)])
    keep = np.isin(blobs, theirs) & corridor
    return np.where(keep, labels, 0)


def extent(mask):
    """top and bottom row of a garment, ignoring the odd stray pixel"""
    ys = np.nonzero(mask)[0]
    if len(ys) < 200:
        return None
    return np.percentile(ys, 2), np.percentile(ys, 98)


def median_width(mask, row_from, row_to):
    """typical left-to-right width of a garment between two rows"""
    widths = []
    for row in mask[max(0, int(row_from)):int(row_to)]:
        xs = np.nonzero(row)[0]
        if len(xs) > 5:
            widths.append(xs[-1] - xs[0])
    return float(np.median(widths)) if widths else None


def line_reading(ratio, to_hip, source):
    """turn the break position into a score and a garment-focused explanation.
    ratio: break height as a share of shoulder-to-ankle. to_hip: the same break measured shoulder-to-hip (1 = at the hip)"""
    distance = min(abs(ratio - 0.30), abs(ratio - 0.62)) #the 1:2 and 2:1 splits read longest
    score = int(max(45, min(100, 100 - 200 * distance)))
    what = {"belt": "The belt", "tuck": "The tuck", "hem": "The top's hem"}[source]
    tip = None
    if to_hip < 0.55:
        text = f"{what} sets the break high, giving the bottom half most of the line. It reads cropped and leggy."
    elif to_hip < 0.95:
        text = f"{what} sets the break near the waist, so the bottom half gets about two-thirds of the line: the classic 1:2 split that lengthens the leg line."
    elif to_hip < 1.25:
        text = f"{what} lands around the hip, splitting the outfit close to half and half, which reads a little blocky."
        tip = "Try a front tuck, a belt, or a cropped layer to move the break up toward the waist."
    else:
        text = f"{what} runs long over a shorter bottom, a 2:1 split that works best when the bottom half is streamlined."
    return score, text, tip


def volume_reading(top_fit, bottom_fit, defined_waist):
    if top_fit != bottom_fit:
        loose, fitted = ("top", "bottom") if top_fit == "relaxed" else ("bottom", "top")
        return 95, f"A relaxed {loose} with a fitted {fitted}: one piece carries the volume and the other keeps it grounded.", None
    if top_fit == "fitted":
        return 85, "Fitted top and fitted bottom make a streamlined, column-like silhouette.", None
    if defined_waist:
        return 88, "Relaxed on top and bottom, and the defined waist keeps all that volume looking intentional.", None
    return 72, "Relaxed on top and bottom with nothing defining the middle, so the shape can get lost.", \
        "A belt or a tuck would give the volume a center point."


def analyze_proportion(image_array, labels, landmarks):
    """proportion score from where the outfit breaks and how its volume is balanced"""
    if landmarks is None or abs(landmarks[L_HIP]["y"] - landmarks[L_SHOULDER]["y"]) < 0.05:
        return {"score": None, "reason": "Couldn't find a full body in the frame. Step back so the whole outfit is visible.", "tips": [], "notes": []}
    height, width = labels.shape
    y = lambda i: landmarks[i]["y"] * height
    x = lambda i: landmarks[i]["x"] * width
    notes = []

    upper = np.isin(labels, list(UPPER | UPPER_PARTS))
    lower = np.isin(labels, list(LOWER))
    one_piece = np.isin(labels, list(ONE_PIECE))
    belt = labels == BELT

    shoulder_row = (y(L_SHOULDER) + y(R_SHOULDER)) / 2
    hip_row = (y(L_HIP) + y(R_HIP)) / 2
    feet_visible = min(landmarks[L_ANKLE]["visibility"], landmarks[R_ANKLE]["visibility"]) > 0.5
    bottoms = [hip_row]
    if feet_visible:
        bottoms.append(min(height, (y(L_ANKLE) + y(R_ANKLE)) / 2))
    else:
        notes.append("Feet are out of frame, so the line estimate uses the bottom of the outfit.")
    for garment in (lower, one_piece):
        e = extent(garment)
        if e:
            bottoms.append(e[1])
    outfit_bottom = max(bottoms)
    length = outfit_bottom - shoulder_row
    if length < height * 0.25:
        return {"score": None, "reason": "The outfit is too small in the frame to judge its proportions. Step closer or crop the photo.", "tips": [], "notes": notes}

    upper_extent, lower_extent, belt_extent = extent(upper), extent(lower), extent(belt)
    tips = []

    #where does the eye see the top half end?
    if belt_extent and belt.sum() > 0.0004 * height * width:
        break_row, source = (belt_extent[0] + belt_extent[1]) / 2, "belt"
    elif one_piece.sum() > lower.sum():
        break_row, source = None, None
    elif lower_extent:
        break_row = lower_extent[0] #segmentation gives each pixel one garment, so the bottoms start where the top stops covering them
        source = "tuck" if break_row < hip_row - length * 0.03 else "hem"
    elif upper_extent:
        break_row, source = upper_extent[1], "hem"
    else:
        return {"score": None, "reason": "Couldn't tell the top and bottom apart in this photo.", "tips": [], "notes": notes}

    if break_row is None:
        ratio = None
        line_score, line_text = 88, "A one-piece makes an unbroken vertical line from top to hem, which reads long and simple."
    else:
        ratio = (break_row - shoulder_row) / length
        line_score, line_text, tip = line_reading(ratio, (break_row - shoulder_row) / (hip_row - shoulder_row), source)
        if tip:
            tips.append(tip)

    #volume: garment width compared to the frame of the shoulders and hips
    shoulder_width = abs(x(L_SHOULDER) - x(R_SHOULDER))
    hip_width = abs(x(L_HIP) - x(R_HIP))
    split = break_row or hip_row
    top_width = median_width(upper | one_piece, shoulder_row, split)
    bottom_width = median_width(lower | one_piece, split + (outfit_bottom - split) * 0.5, outfit_bottom - length * 0.05)
    top_ratio = top_width / shoulder_width if top_width and shoulder_width > 5 else None
    bottom_ratio = bottom_width / hip_width if bottom_width and hip_width > 5 else None

    top_fit = bottom_fit = None
    volume_text = ""
    score = line_score
    if top_ratio and bottom_ratio and source is not None: #a one-piece's shape is judged by its line alone
        top_fit = "relaxed" if top_ratio > 1.55 else "fitted"
        bottom_fit = "relaxed" if bottom_ratio > 1.8 else "fitted"
        volume_score, volume_text, tip = volume_reading(top_fit, bottom_fit, source in ("belt", "tuck"))
        if tip:
            tips.append(tip)
        score = round(0.65 * line_score + 0.35 * volume_score)

    return {
        "score": int(score),
        "reason": " ".join(t for t in [line_text, volume_text] if t),
        "tips": tips,
        "notes": notes,
        "break_ratio": None if ratio is None else round(float(ratio), 3),
        "top_fit": top_fit,
        "bottom_fit": bottom_fit,
        "debug": {"top_ratio": top_ratio and round(top_ratio, 2), "bottom_ratio": bottom_ratio and round(bottom_ratio, 2),
                  "break_row": break_row and int(break_row), "shoulder_row": int(shoulder_row), "hip_row": int(hip_row),
                  "outfit_bottom": int(outfit_bottom)},
    }


if __name__ == "__main__":
    import sys
    image = Image.open(sys.argv[1] if len(sys.argv) > 1 else "images.jpeg").convert("RGB")
    image_array = np.array(image) #opens photo and converts it to numpy array
    result = analyze_proportion(image_array, segment_labels(image), detect_pose(image_array))
    for key, value in result.items():
        print(f"{key}: {value}")
