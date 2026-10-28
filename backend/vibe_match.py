"""aesthetic detection and compare, using CLIP.

CLIP scores how well an image matches a piece of text. Each aesthetic gets a few written descriptions,
and the closest one wins. To explain the result, CLIP also rates the outfit on a handful of plain traits
(neutral vs bold colors, relaxed vs fitted, ...). Each aesthetic says which way it leans on those traits,
so compare mode can point at exactly what's pulling an outfit away from a target.
"""
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

CHECKPOINT = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(CHECKPOINT).eval()
processor = CLIPProcessor.from_pretrained(CHECKPOINT)

# trait: (prompt for the high end, prompt for the low end, how the high end reads, how the low end reads)
TRAITS = {
    "neutral": ("clothes in muted neutral colors like black, white, beige and gray",
                "clothes in bright, bold, saturated colors", "a muted, neutral palette", "bold, saturated color"),
    "dark": ("dark, moody clothing in deep colors", "light, airy clothing in pale colors",
             "dark, deep tones", "light, airy tones"),
    "relaxed": ("baggy, oversized, loose-fitting clothes", "fitted, tailored, close-fitting clothes",
                "a relaxed, oversized fit", "a fitted, tailored fit"),
    "polished": ("a polished, dressy, put-together outfit", "a casual, laid-back, everyday outfit",
                 "polished and dressed up", "casual and laid-back"),
    "patterned": ("clothes with bold prints, stripes or patterns", "plain solid-colored clothes",
                  "prints and patterns", "solid, plain pieces"),
    "sporty": ("sporty athletic sportswear", "non-athletic fashion clothing",
               "sporty, athletic pieces", "non-athletic pieces"),
    "vintage": ("vintage retro style clothing", "modern contemporary style clothing",
                "a vintage, retro feel", "a modern, current feel"),
    "soft": ("soft, romantic, delicate clothing with frills, lace or bows", "sharp, edgy, utilitarian clothing",
             "soft, romantic details", "sharp, edgy lines"),
}

# name: (descriptions, trait leanings (+1 high end, -1 low end), pieces that push an outfit toward it)
AESTHETICS = {
    "streetwear": (["a streetwear outfit with a hoodie, graphic tee, baggy pants and sneakers",
                    "hypebeast street style with oversized layers and chunky sneakers"],
                   {"relaxed": 1, "polished": -1, "sporty": 1}, "an oversized hoodie or graphic tee, baggy cargos, chunky sneakers"),
    "minimalist": (["a minimalist outfit of clean, simple pieces in neutral colors",
                    "a capsule wardrobe look with plain basics and clean lines"],
                   {"neutral": 1, "patterned": -1, "soft": -1}, "clean basics in black, white or beige with no logos or prints"),
    "old money": (["an old money outfit with a cashmere sweater, pressed trousers and loafers",
                   "quiet luxury: tailored neutral classics that look expensive"],
                  {"neutral": 1, "polished": 1, "relaxed": -1, "sporty": -1}, "a knit or crisp button-down, tailored trousers, loafers, muted tones"),
    "y2k": (["a y2k outfit with low-rise jeans, a baby tee and bright early 2000s colors",
             "an early 2000s pop star look with butterfly clips, rhinestones and a mini skirt"],
            {"neutral": -1, "soft": 1, "polished": -1}, "a baby tee, low-rise denim, tinted glasses, a pop of pink or metallic"),
    "cottagecore": (["a cottagecore outfit with a floral prairie dress, puff sleeves and soft pastels",
                     "a romantic countryside look with gingham, lace and linen"],
                    {"soft": 1, "patterned": 1, "vintage": 1, "dark": -1}, "a floral or gingham dress, puff sleeves, linen, soft pastels"),
    "dark academia": (["a dark academia outfit with a tweed blazer, turtleneck and plaid in brown and black",
                       "a scholarly vintage look with wool, oxfords and earthy dark tones"],
                      {"dark": 1, "polished": 1, "vintage": 1, "sporty": -1}, "a tweed blazer or cardigan, a turtleneck, plaid trousers, oxfords"),
    "athleisure": (["an athleisure outfit with leggings, a sports bra, a zip-up and sneakers",
                    "a matching workout set worn as everyday wear"],
                   {"sporty": 1, "polished": -1, "relaxed": -1}, "leggings or joggers, a matching set, clean sneakers"),
    "business casual": (["a business casual outfit with a blazer, button-down shirt and chinos",
                         "a smart office look with tailored trousers and a crisp shirt"],
                        {"polished": 1, "neutral": 1, "relaxed": -1, "sporty": -1}, "a blazer, a crisp shirt, tailored trousers, loafers or clean flats"),
    "brunch": (["a cute brunch outfit with a sundress or a pretty top in light colors",
                "a breezy weekend daytime look, effortless and put together"],
               {"dark": -1, "soft": 1, "sporty": -1}, "a sundress or pretty blouse, light colors, sandals or ballet flats"),
    "grunge": (["a grunge outfit with flannel, ripped jeans, combat boots and a band tee",
                "a 90s grunge look, messy and dark with layered plaid"],
               {"dark": 1, "relaxed": 1, "polished": -1, "soft": -1}, "flannel, a band tee, ripped denim, combat boots"),
    "boho": (["a boho outfit with a flowy maxi skirt, earthy tones, fringe and layered jewelry",
              "a bohemian festival look with suede, crochet and flowing fabrics"],
             {"relaxed": 1, "patterned": 1, "vintage": 1, "soft": 1}, "a flowy maxi skirt, crochet or suede, earthy prints, layered jewelry"),
    "preppy": (["a preppy outfit with a polo, a pleated skirt or chinos and a cardigan",
                "an ivy league look with stripes, cable knits and loafers"],
               {"polished": 1, "patterned": 1, "sporty": -1}, "a polo or cable knit, pleats, stripes, loafers"),
    "coquette": (["a coquette outfit with bows, lace, ribbons and pink feminine details",
                  "a dainty girly look with satin, pearls and ballet flats"],
                 {"soft": 1, "dark": -1, "neutral": -1}, "bows, lace or satin, pink or cream, ballet flats"),
    "gorpcore": (["a gorpcore outfit with technical outdoor fleece, cargo pants and hiking shoes",
                  "a functional outdoor look with a windbreaker, utility vest and trail runners"],
                 {"sporty": 1, "relaxed": 1, "polished": -1, "soft": -1}, "a fleece or shell jacket, cargos, trail shoes"),
    "goth": (["a goth outfit, all black with lace, leather and platform boots",
              "a dark gothic look with black velvet, chains and heavy boots"],
             {"dark": 1, "neutral": 1, "soft": -1}, "all black, leather or lace, silver hardware, platform boots"),
    "night out": (["a going-out outfit, a sleek dress or going-out top for a night out",
                   "a glam evening party look with heels"],
                  {"polished": 1, "sporty": -1, "relaxed": -1}, "a sleek dress or going-out top, heels, one statement piece"),
}


def _features(output):
    """newer transformers versions wrap the embedding in an output object"""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


@torch.no_grad()
def _embed_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    return torch.nn.functional.normalize(_features(model.get_text_features(**inputs)), dim=-1)


@torch.no_grad()
def _embed_image(image):
    inputs = processor(images=image, return_tensors="pt")
    return torch.nn.functional.normalize(_features(model.get_image_features(**inputs)), dim=-1)[0]


#text never changes, so embed it once at startup. each aesthetic is the average of its descriptions
NAMES = list(AESTHETICS)
_aesthetic_emb = torch.nn.functional.normalize(torch.stack([
    _embed_text([f"a photo of a person wearing {d}" for d in AESTHETICS[n][0]]).mean(0) for n in NAMES]), dim=-1)
_trait_emb = {t: _embed_text([f"a photo of a person wearing {hi}", f"a photo of a person wearing {lo}"])
              for t, (hi, lo, _, _) in TRAITS.items()}


def outfit_crop(image, labels):
    """crop to the outfit itself so the street behind it doesn't steer the vibe"""
    ys, xs = np.nonzero(labels)
    if len(ys) < 500:
        return image
    pad = int(0.06 * (ys.max() - ys.min()))
    return image.crop((max(0, xs.min() - pad), max(0, ys.min() - pad * 2),
                       min(image.width, xs.max() + pad), min(image.height, ys.max() + pad)))


FASHION_NEUTRALS = {"cream", "brown", "navy", "denim blue", "olive"} #have a hue, but style like neutrals


def measured_traits(colors, proportion):
    """traits we can measure directly instead of asking CLIP to guess: palette from the color swatches,
    fit from the proportion geometry"""
    out = {}
    if colors:
        swatches = colors["swatches"]
        out["neutral"] = sum(s["share"] * (1 if s["neutral"] else 0.6 if s["name"] in FASHION_NEUTRALS else 0) for s in swatches)
        brightness = sum(s["share"] * s["value"] for s in swatches)
        out["dark"] = float(np.clip(1.5 - 2 * brightness, 0, 1))
    if proportion and proportion.get("top_fit"):
        out["relaxed"] = [proportion["top_fit"], proportion["bottom_fit"]].count("relaxed") / 2
    return out


def read_traits(image_emb, measured=None):
    """0-1 for each trait: how much the outfit leans to its high end"""
    traits = {t: float(torch.softmax(100 * emb @ image_emb, 0)[0]) for t, emb in _trait_emb.items()}
    traits.update(measured or {})
    return traits


def trait_agreement(traits, leanings):
    """per trait, how much the outfit agrees with an aesthetic's leaning (0-1)"""
    return {t: traits[t] if lean > 0 else 1 - traits[t] for t, lean in leanings.items()}


def compare(name, sims, traits):
    """% match against a target aesthetic, and the traits pulling the outfit away from it"""
    _, leanings, pieces = AESTHETICS[name]
    s = sims[NAMES.index(name)]
    closeness = (s - sims.min()) / max(1e-6, sims.max() - sims.min())
    agree = trait_agreement(traits, leanings)
    match = round(100 * (0.5 * closeness + 0.5 * np.mean(list(agree.values()))))

    pulling = []
    for t, a in sorted(agree.items(), key=lambda kv: kv[1]):
        if a < 0.45:
            hi, lo = TRAITS[t][2], TRAITS[t][3]
            has, wants = (hi, lo) if leanings[t] < 0 else (lo, hi)
            pulling.append(f"The outfit reads {has}, while {name} leans on {wants}.")
    return {"aesthetic": name, "match": int(max(5, min(99, match))), "pulling_away": pulling[:3],
            "lean_in": f"To lean {name}: {pieces}."}


def analyze_vibe(image, labels, target=None, colors=None, proportion=None):
    image_emb = _embed_image(outfit_crop(image, labels))
    sims = (_aesthetic_emb @ image_emb).numpy()
    traits = read_traits(image_emb, measured_traits(colors, proportion))
    #CLIP's pick, nudged by how well the outfit's traits agree with each aesthetic's leanings
    agreement = np.array([np.mean(list(trait_agreement(traits, AESTHETICS[n][1]).values())) for n in NAMES])
    probs = torch.softmax(torch.tensor(sims * 100 + agreement * 3), 0).numpy()

    order = np.argsort(-probs)
    top = NAMES[order[0]]
    leanings = AESTHETICS[top][1]
    because = [TRAITS[t][2] if leanings[t] > 0 else TRAITS[t][3]
               for t, a in sorted(trait_agreement(traits, leanings).items(), key=lambda kv: -kv[1]) if a > 0.55][:3]
    result = {
        "aesthetic": top,
        "confidence": round(float(probs[order[0]]), 3),
        "runners_up": [{"aesthetic": NAMES[i], "confidence": round(float(probs[i]), 3)} for i in order[1:4]],
        "because": because,
        "traits": {t: round(v, 3) for t, v in traits.items()},
        "compare": None,
    }
    if target:
        if target not in AESTHETICS:
            raise ValueError(f"unknown aesthetic {target!r}; pick one of: {', '.join(NAMES)}")
        result["compare"] = compare(target, sims, traits)
    return result


if __name__ == "__main__":
    import sys
    from PIL import Image
    from segment import segment_labels
    image = Image.open(sys.argv[1] if len(sys.argv) > 1 else "images.jpeg").convert("RGB")
    print(analyze_vibe(image, segment_labels(image), sys.argv[2] if len(sys.argv) > 2 else "old money"))
