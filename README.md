# outfit-mirror

**A mirror, not a store.** Outfit Mirror is a hands-free computer vision project that gives instant, explainable feedback on how well an outfit works. It looks at what you're already wearing and reads its aesthetic, color harmony, and proportion. It doesn't recommend things to buy.

I built it because I'm always putting fits together and wanted to see whether I could build something that actually understands outfit composition.

> **Status: working locally.** Photo and live-camera modes, aesthetic detection and compare, color harmony and proportion scoring all run end to end on a laptop (about 3 seconds per photo on CPU). Deployment is next. See the [roadmap](#roadmap).

---

## How it works

1. **Pick an input.** Upload a photo, or go live with your webcam.
2. **Trigger the scan.** In live mode there's no button: a hand gesture starts the scan.
3. **Watch the scan.** Landmark dots move down the body from top to bottom.
4. **See your aesthetic.** The outfit's vibe (e.g. *"brunch"*) is detected automatically.
5. **Compare it (optional).** Pick a different target aesthetic to get a % match and see what's pulling the outfit away from it.
6. **Read the sub-scores.** You always get color harmony and proportion scores, each with an explanation.

### Input modes

| Mode | Description |
|---|---|
| **Photo** | Upload any image (stock or personal). This is the main mode during development, so testing doesn't require wearing every outfit. |
| **Live video** | Stand in front of the camera. There's no upload and no button; a hand gesture triggers the scan. |

## Features

- **Aesthetic detection:** automatically identifies the outfit's vibe.
- **Aesthetic compare** (optional): shows a percentage match against a target aesthetic and explains what's pulling it away.
- **Color harmony score:** measures how well the outfit's own colors work together (monochrome, analogous, triadic, complementary). It compares outfit colors only, never skin tone.
- **Proportion score:** measures overall silhouette balance, using body-neutral language (see below).
- **Explainable output:** gives sub-scores with reasons, never just a single number.

## Design rule: body-neutral feedback

Proportion feedback always describes **the garment and its styling effect**, never the body underneath it.

| ❌ Avoid | ✅ Use instead |
|---|---|
| "Your waistline breaks the balance." | "This top's hem sits below your hip, which visually shortens the leg line." |

- No "flattering," "slimming," or "bulky."
- Frame feedback as *styling cause → visual effect*, never as a verdict.
- No score should ever read as "your body scored low." Scores apply only to outfit choices.

## Tech stack

| Component | Choice |
|---|---|
| Segmentation | Pretrained SegFormer fashion model ([`sayeed99/segformer-b2-fashion`](https://huggingface.co/sayeed99/segformer-b2-fashion)) |
| Pose / gesture | MediaPipe Pose (Python) and Gesture Recognizer (in the browser) |
| Color analysis | scikit-learn k-means, HSV hue math |
| Aesthetic embeddings | CLIP ([`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)) |
| Backend | Python, FastAPI |
| Frontend | Single page, plain HTML/CSS/JS, no login, no saved history |
| Deployment | AWS (planned) |

## Project structure

```
outfit-mirror/
├── backend/
│   ├── segment.py         # SegFormer: label every pixel with a garment (shirt, pants, dress, belt, shoe, ...)
│   ├── proportion.py      # MediaPipe pose, main-person isolation, proportion scoring
│   ├── color_harmony.py   # dominant colors, hue relationships, color score
│   ├── vibe_match.py      # CLIP aesthetic detection and compare
│   ├── feedback.py        # combines sub-scores into explainable output
│   ├── main.py            # pipeline: photo in, feedback out (also a CLI)
│   ├── server.py          # FastAPI: /api/analyze, /api/aesthetics, serves the UI
│   └── test_images/       # street-style photos used for calibration (see SOURCES.md)
└── frontend/              # mirror UI: index.html, style.css, app.js (no build step)
```

### How the pipeline works

1. **Find the person.** MediaPipe Pose finds 33 body landmarks for the main person, and the photo is cropped to them.
2. **Segment the outfit.** SegFormer labels every pixel with a garment class. Clothing blobs that don't touch the main person's skeleton are dropped, so people in the background never count.
3. **Color harmony.** K-means finds the 5 dominant outfit colors. Neutrals (black, white, gray) are set aside because they have no meaningful hue. The rest are compared pairwise by circular hue distance and classified as monochrome, analogous, triadic or complementary (majority vote, with a tiebreaker on closeness to the ideal gap). The score rewards pairs that land cleanly on a harmony, and penalizes more than 3 hue families.
4. **Proportion.** The visual break (belt, tuck or the top's hem) is located along the shoulder-to-ankle line. A 1:2 or 2:1 split reads longest; an even 1:1 split reads blockier. Garment widths relative to shoulders and hips classify the top and bottom as relaxed or fitted, and the balance between them is scored.
5. **Aesthetic.** CLIP compares the outfit to written descriptions of 16 aesthetics. It also rates plain traits (polished, patterned, sporty, vintage, soft); palette and fit traits come from the measured colors and proportions instead of CLIP. Each aesthetic leans a certain way on those traits, so compare mode can name exactly what's pulling an outfit away from a target.
6. **Feedback.** Sub-scores, each with a reason, plus tips. No single overall number.

### Live mode

The browser runs MediaPipe's gesture recognizer on the webcam feed. Hold up an open palm ✋ for under a second, lower it during the 3-2-1 countdown, and the frame is captured and analyzed. Space bar works as a fallback trigger.

## Running it

Requires Python 3.12.

```bash
git clone https://github.com/roridemonslayer/outfit-mirror.git
cd outfit-mirror/backend

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn server:app --port 8000
```

Then open http://localhost:8000. The SegFormer and CLIP weights (about 700 MB) download from Hugging Face on the first run, so that start takes a few minutes. After that, startup takes about 15 seconds.

From the command line:

```bash
python main.py test_images/t7.jpg --target "old money"
```

## Roadmap

- [x] **Phase 0 — Setup:** project folders, virtual environment, core packages
- [x] **Phase 1 — Segmentation:** SegFormer outfit mask via `segment_image()`
- [x] **Phase 2 — Color harmony scoring:** dominant colors, neutral handling, harmony vote with tiebreaker, score and explanation
- [x] **Phase 3 — Pose / proportion scoring:** break position, relaxed vs fitted volume, main-person isolation
- [x] **Phase 4 — Vibe / aesthetic matching:** CLIP detect mode plus compare mode with "what's pulling it away"
- [x] **Phase 5 — Feedback synthesis:** sub-scores with reasons, tips, body-neutral language
- [x] **Phase 6 — Live video + gesture capture:** webcam mode, open-palm trigger, countdown
- [x] **Phase 7 — API layer:** FastAPI `/api/analyze`
- [x] **Phase 8 — Frontend / mirror UI:** photo upload + live video, mask overlay, color swatches, pose skeleton scan, scores
- [ ] **Phase 9 — Deploy**

### Open questions

- Reference images per aesthetic, to sharpen detection beyond text descriptions
- A larger CLIP checkpoint, if hosting has the memory for it
- A shareable result card (to revisit once real results exist)
- Which AWS services to use for backend hosting (depends on final model weight)
