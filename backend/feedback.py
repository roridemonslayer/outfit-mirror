"""combine the sub-scores into one explainable result.

there's deliberately no single overall number: each score comes with the reason behind it.
every sentence describes the garments and how they're styled, never the body wearing them.
"""


def build_feedback(colors, proportion, vibe):
    sections = []
    tips = []

    if vibe:
        because = ", ".join(vibe["because"]) if vibe["because"] else "the overall mix of pieces"
        sections.append({"key": "aesthetic", "title": "Aesthetic", "score": None, "label": vibe["aesthetic"],
                         "reason": f"Reads {vibe['aesthetic']}: {because}."})
        compare = vibe.get("compare")
        if compare:
            if compare["pulling_away"]:
                reason = " ".join(compare["pulling_away"])
            else:
                reason = f"Nothing is pulling it away from {compare['aesthetic']}. It's already there."
            sections.append({"key": "compare", "title": f"vs {compare['aesthetic']}", "score": compare["match"],
                             "label": f"{compare['match']}% match", "reason": reason})
            if compare["pulling_away"]:
                tips.append(compare["lean_in"])

    if colors:
        sections.append({"key": "color", "title": "Color harmony", "score": colors["score"],
                         "label": colors["harmony"], "reason": colors["reason"]})
    else:
        sections.append({"key": "color", "title": "Color harmony", "score": None, "label": "not enough outfit",
                         "reason": "Couldn't find enough of the outfit to read its colors. Try a clearer, full-length shot."})

    sections.append({"key": "proportion", "title": "Proportion", "score": proportion["score"],
                     "label": _proportion_label(proportion), "reason": proportion["reason"]})
    tips += proportion.get("tips", [])

    return {
        "headline": _headline(vibe, colors, proportion),
        "sections": sections,
        "tips": tips,
        "notes": proportion.get("notes", []),
    }


def _proportion_label(p):
    if p["score"] is None:
        return "needs a full-length shot"
    if p.get("break_ratio") is None:
        return "one-piece line"
    split = "1:2 split" if p["break_ratio"] < 0.4 else "even split" if p["break_ratio"] < 0.56 else "2:1 split"
    if p.get("top_fit"):
        return f"{split} · {p['top_fit']} top, {p['bottom_fit']} bottom"
    return split


def _headline(vibe, colors, proportion):
    scores = [s for s in (colors and colors["score"], proportion["score"]) if s is not None]
    strong = scores and min(scores) >= 85
    if vibe:
        name = vibe["aesthetic"]
        article = "an" if name[0] in "aeiou" else "a"
        if strong:
            return f"This is {article} {name} fit, and it's working."
        return f"This reads {name}. A couple of tweaks would sharpen it."
    return "Here's how the outfit reads." if not strong else "This outfit is working."
