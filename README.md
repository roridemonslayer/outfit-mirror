# outfit-mirror

**A mirror, not a store.** Outfit Mirror is a hands-free computer vision project that gives instant, explainable feedback on how well an outfit works. It looks at what you're already wearing and reads its aesthetic, color harmony, and proportion. It doesn't recommend things to buy.

I built it because I'm always putting fits together and wanted to see whether I could build something that actually understands outfit composition.

> 🚧 **Status: early development.** Segmentation works, and color harmony scoring is in progress. See the [roadmap](#roadmap).

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
| Capture / gesture | MediaPipe Hands + Pose |
| Color analysis | OpenCV / scikit-learn (k-means) |
| Aesthetic embeddings | CLIP (open-source checkpoint) |
| Backend | Python |
| Frontend | Single page, no login, no saved history |
| Deployment | AWS |

## Project structure

```
outfit-mirror/
├── backend/
│   ├── segment.py         # segment_image(path) → boolean outfit mask
│   ├── color-harmony.py   # dominant colors + hue-relationship classification
│   ├── porportion.py      # proportion scoring (planned)
│   ├── vibe-match.py      # aesthetic detection / compare (planned)
│   ├── feedback.py        # combines sub-scores into explainable output (planned)
│   ├── main.py            # pipeline entry point (planned)
│   └── images.jpeg        # sample test image
└── frontend/              # mirror UI (planned)
```

### Pipeline so far

- **Segmentation (`segment.py`):** runs SegFormer on the image, upsamples the logits to the original size, and collapses the model's labels into a single outfit mask. Everything that isn't background counts as outfit; the mask isn't split per garment.
- **Color harmony (`color-harmony.py`):** pulls the clothing pixels using the mask and runs k-means with 5 clusters to find the dominant colors. It converts each color to HSV hue degrees, then computes the circular hue distance for every pair. Each pair is classified as monochrome (0°), analogous (45°), triadic (120°), or complementary (180°) by nearest match, and a majority vote picks the overall harmony.

## Getting started

Requires Python 3.12.

```bash
git clone https://github.com/roridemonslayer/outfit-mirror.git
cd outfit-mirror/backend

python3 -m venv venv
source venv/bin/activate
pip install torch transformers pillow numpy scikit-learn

# Run segmentation on the sample image
python segment.py

# Run color harmony analysis on the sample image
python color-harmony.py
```

The SegFormer weights download from Hugging Face on the first run. Scripts currently read `images.jpeg` from the `backend/` directory, so run them from there.

## Roadmap

- [x] **Phase 0 — Setup:** project folders, virtual environment, core packages
- [x] **Phase 1 — Segmentation:** SegFormer outfit mask via `segment_image()`
- [ ] **Phase 2 — Color harmony scoring:** *in progress*; finishing the tiebreaker (averaging distances within tied categories)
- [ ] **Phase 3 — Pose / proportion scoring:** MediaPipe pose landmarks, outfit silhouette relative to body joints
- [ ] **Phase 4 — Vibe / aesthetic matching:** CLIP embeddings; detect mode plus optional compare mode
- [ ] **Phase 5 — Feedback synthesis:** combine sub-scores into one structured, explainable result
- [ ] **Phase 6 — Live video + gesture capture:** webcam mode, hand-gesture trigger, frame-quality gating
- [ ] **Phase 7 — API layer:** wrap the pipeline so a server can call it
- [ ] **Phase 8 — Frontend / mirror UI:** photo upload + live video, mask overlay, color swatches, pose skeleton, scores
- [ ] **Phase 9 — Deploy:** backend and frontend live on AWS

### Open questions

- Final list of aesthetic presets and their reference images
- A composite score formula, or sub-scores only
- A shareable result card (to revisit once real results exist)
- Which AWS services to use for backend hosting (depends on final model weight)
