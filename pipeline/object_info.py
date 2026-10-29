"""Real-world properties of an object from its photos: NVIDIA Nemotron identifies it, Tavily finds its specs online.

    props = object_properties("celsius_can", [frame paths], "objects.json", use_ai=True)

Order: objects.json (if the object is already there) -> optionally the AI lookup (use_ai=True /
make_asset.py --nemotron), on Nebius Token Factory:
  1. Nemotron 3.5 Super VL looks at the photos: what the object is (brand, product, variant), full/empty, material
  2. Tavily searches the web for its dimensions and weight
  3. Nemotron 3 Super reads only those search results and returns size, mass, material, friction + the URLs used
  4. a check with the AprilTag in the photos as a ruler: the object's width in the photos vs the found size
-> otherwise you are asked in the terminal (the AI's values, if any, as defaults). Nothing is invented silently:
every number keeps a "source" line, a failed size check or a not-found product asks you, and the result is saved
to objects.json. Keys: NEBIUS_API_KEY and TAVILY_API_KEY (environment or the repo .env).
"""
import base64
import json
import os
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

NEBIUS_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"
VISION_MODEL = "nvidia/Nemotron-3.5-Super-VL"
TEXT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
SIZE_TOLERANCE = 0.2  # the width in the photos (tag ruler) may differ this much from the found size
TAG_MM = 21.57  # printed tag's black square (robot/config.json "tag_mm" when measured, else this)

IDENTIFY_PROMPT = """These photos show one object that a small robot arm (SO-100) will pick up in a physics simulation.
Identify it as precisely as the photos allow (read the label: brand, product, flavour, size variant such as
"12 fl oz slim can"). Reply with JSON only:
{"is_product": true if it is a recognisable commercial product (brand/label readable), else false,
 "name": "brand + product + size variant, or a plain description",
 "search_queries": ["this exact product's dimensions", "this exact product's weight",
                    "the dimensions of its standard container/package type, e.g. '12 fl oz slim aluminum can
                     dimensions height diameter' (empty list item if it has none)"],
 "state": "e.g. full / empty / opened / not applicable (when it changes the weight)",
 "material": "main surface material", "visible_text": "label text you can read"}"""

EXTRACT_PROMPT = """Object: {name} (state in the photos: {state}; material seen: {material}).
Web search results (numbered, with URLs):
{results}

Using ONLY numbers that appear in these results, give the object's real size and mass. Different sources may
describe different variants (e.g. slim vs standard can): use the variant that matches the object, and say which
sources you used. Standard container/package specs in the results count when the object is that container. If the photos say opened/empty, give the empty mass (and the full mass in mass_note).
Friction: a typical sliding friction coefficient of that surface material on a cloth or wood table.
If the results do not give the size or mass of this exact object, set found=false and give rough estimates.
Reply with JSON only:
{{"found": true/false, "name": "...", "size_m": [length, width, height] in metres (height = up when it stands),
 "mass_kg": number, "mass_note": "...", "material": "...", "friction": number,
 "sources": ["URLs actually used"], "notes": "which variant / any disagreement between sources"}}"""


def _key(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (put it in the repo .env)")
    return value


def _post(url, key, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _json_from(text):
    """The JSON object in a model's reply (reasoning models may think first or wrap it in ```json)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON in the reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def nemotron(model, content, max_tokens=4000):
    """One Nemotron chat call on Nebius Token Factory -> the reply parsed as JSON."""
    out = _post(NEBIUS_URL, _key("NEBIUS_API_KEY"), dict(
        model=model, max_tokens=max_tokens, temperature=0.2, messages=[{"role": "user", "content": content}]))
    return _json_from(out["choices"][0]["message"]["content"] or "")


def _image_part(path, max_side=1024):
    img = cv2.imread(str(path))
    s = min(1.0, max_side / max(img.shape[:2]))
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.standard_b64encode(buf.tobytes()).decode()}}


def tavily(query, max_results=5):
    """Web search results [{title, url, content}] from Tavily."""
    out = _post(TAVILY_URL, _key("TAVILY_API_KEY"), dict(query=query, max_results=max_results,
                                                          search_depth="advanced"), timeout=60)
    return [dict(title=r["title"], url=r["url"], content=r["content"]) for r in out["results"]]


def look_up(frames):
    """Nemotron VL identifies the object -> Tavily searches -> Nemotron reads the results. Returns a dict."""
    seen = nemotron(VISION_MODEL, [_image_part(f) for f in frames[:4]] + [{"type": "text", "text": IDENTIFY_PROMPT}])
    print(f"   Nemotron VL: {seen['name']} | product: {seen['is_product']} | {seen['state']} | {seen['material']}")
    results, urls = [], set()
    for query in [q for q in seen["search_queries"] if q.strip()][:3]:
        for r in tavily(query):
            if r["url"] not in urls:
                urls.add(r["url"])
                results.append(r)
    print(f"   Tavily: {len(results)} web results")
    listing = "\n".join(f"[{i + 1}] {r['title']} ({r['url']})\n{r['content'][:1200]}" for i, r in enumerate(results))
    found = nemotron(TEXT_MODEL, EXTRACT_PROMPT.format(name=seen["name"], state=seen["state"],
                                                       material=seen["material"], results=listing))
    found["sources"] = [s for s in found.get("sources", []) if s in urls]  # only URLs Tavily really returned
    found["identified"] = bool(seen["is_product"] and found.get("found") and found["sources"])
    return found


def measure_with_tags(frames, tag_mm=None):
    """The object measured in the photos with an AprilTag on it as the ruler: its width at the tag's level (the
    narrowest over the photos, i.e. the most face-on view; a box seen across a corner looks up to 1.4x wider) and
    the tag centre's height above the object's bottom. {"width_m", "tag_height_m", "photos"} or None if no photo
    shows a tag. Width is used, not height: a camera looking down sees the top face too, which inflates height."""
    from cutout import cut_out
    tag_mm = tag_mm or TAG_MM
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                       cv2.aruco.DetectorParameters())
    widths, heights = [], []
    for f in frames:
        img = cv2.imread(str(f))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None:
            continue
        c = corners[0][0]
        sides = np.linalg.norm(c - np.roll(c, 1, 0), axis=1)
        if sides.min() < 0.75 * sides.max():  # seen too slanted: the sides are foreshortened
            continue
        mm = tag_mm / sides.mean()  # millimetres per pixel at the tag
        solid = np.asarray(cut_out(f))[:, :, 3] > 0
        x, y = c.mean(0).round().astype(int)
        row, col = solid[y], solid[:, x]
        if not row[x]:
            continue
        left = x - np.argmin(row[x::-1]) if not row[:x + 1].all() else 0
        right = x + np.argmin(row[x:]) if not row[x:].all() else len(row)
        bottom = y + np.argmin(col[y:]) if not col[y:].all() else len(col)
        widths.append((right - left) * mm / 1000)
        heights.append((bottom - y) * mm / 1000)
    if not widths:
        return None
    return dict(width_m=float(min(widths)), tag_height_m=float(np.median(heights)), photos=len(widths))


def _ask(question, default):
    answer = input(f"   {question} [{default}]: ").strip()
    return answer or default


def ask_user(name, guess=None, measured=None):
    """Terminal questions, with the AI's (or no) values as defaults; the width measured with the AprilTag replaces
    a footprint that was not found online."""
    g = guess or {}
    print(f"   Please give the real properties of '{name}' (Enter keeps the value in brackets).")
    size = list(g.get("size_m") or [0.05, 0.05, 0.10])
    if measured and not g.get("identified"):  # the AI's guessed shape, scaled to the measured width
        scale = measured["width_m"] / min(size[:2])
        size = [measured["width_m"], measured["width_m"], size[2] * scale]
    if g and not g.get("identified"):
        print("   (not found online: the weight below is only a guess - weigh it)")
    size_cm = _ask("size in cm: length width height", " ".join(f"{v * 100:.1f}" for v in size))
    mass_g = _ask("weight in grams (as it will be picked up)", f"{g.get('mass_kg', 0.05) * 1000:.0f}")
    material = _ask("material", g.get("material") or "plastic")
    friction = _ask("friction on the table (0.3 slippery ... 0.8 grippy)", f"{g.get('friction', 0.5):.2f}")
    return dict(size_m=[float(v) / 100 for v in size_cm.split()], mass_kg=float(mass_g) / 1000,
                material=material, friction=float(friction))


def _save(objects, objects_path, name, props, source):
    entry = dict(source=source, size_m=[round(v, 4) for v in props["size_m"]], mass_kg=round(props["mass_kg"], 4),
                 material=props["material"], friction=[float(props["friction"]), 0.005, 0.0001])
    objects[name] = entry
    Path(objects_path).write_text(json.dumps(objects, indent=2))
    return entry


def object_properties(name, frames, objects_path, use_ai=False, tag_mm=None):
    """Properties for objects.json: already there, from Nemotron + Tavily (use_ai; found and size-checked) or
    from you; saved."""
    objects = json.loads(Path(objects_path).read_text()) if Path(objects_path).exists() else {}
    if name in objects:
        print(f"   '{name}' already in {Path(objects_path).name}: using it")
        return objects[name]
    if not use_ai:
        return _save(objects, objects_path, name, ask_user(name), "entered by the user")
    guess = None
    try:
        print(f"   looking it up: {VISION_MODEL} (photos) -> Tavily (web) -> {TEXT_MODEL} (specs)")
        guess = look_up(frames)
        print(f"   Nemotron: {guess['name']} | found: {guess['identified']} | size {guess['size_m']} m | "
              f"{guess['mass_kg'] * 1000:.0f} g ({guess.get('mass_note', '')}) | {guess['material']} | "
              f"friction {guess['friction']}")
        if guess.get("notes"):
            print(f"     notes: {guess['notes']}")
        for s in guess["sources"][:5]:
            print(f"     source: {s}")
    except Exception as err:  # no keys, no network, API error, unreadable reply: fall back to asking
        print(f"   (lookup not available: {type(err).__name__}: {err}) -> asking you instead")

    check, measured = "", measure_with_tags(frames, tag_mm) if guess else None
    if measured:
        print(f"   AprilTag as ruler ({measured['photos']} photo(s)): {measured['width_m'] * 100:.1f} cm wide, tag centre "
              f"{measured['tag_height_m'] * 100:.1f} cm above the bottom")
    if guess and guess["identified"]:
        if measured is None:
            print("   size check: no AprilTag seen face-on in the photos -> please confirm the values")
            guess["identified"] = False
        else:  # the narrowest view lies between the footprint's short side and its diagonal
            short, diagonal = min(guess["size_m"][:2]), float(np.hypot(*guess["size_m"][:2]))
            ok = (1 - SIZE_TOLERANCE) * short <= measured["width_m"] <= (1 + SIZE_TOLERANCE) * diagonal
            print(f"   size check: {measured['width_m'] * 100:.1f} cm wide in the photos vs {short * 100:.1f}-"
                  f"{diagonal * 100:.1f} cm from the found size -> {'OK' if ok else 'MISMATCH'}")
            check = f"; width checked with the AprilTag: {measured['width_m'] * 100:.1f} cm in the photos"
            guess["identified"] = ok

    if guess and guess["identified"]:
        props = dict(size_m=guess["size_m"], mass_kg=guess["mass_kg"], material=guess["material"],
                     friction=guess["friction"])
        if ask_yes(f"use these values (weight {props['mass_kg'] * 1000:.0f} g, {guess.get('mass_note', '')})?"):
            source = (f"Nemotron ({VISION_MODEL} + {TEXT_MODEL}) with Tavily search identified '{guess['name']}': "
                      + "; ".join(guess["sources"]) + check + (f"; {guess['notes']}" if guess.get("notes") else ""))
            return _save(objects, objects_path, name, props, source)
    props = ask_user(name, guess, measured)
    source = (f"user, with Nemotron + Tavily's values for '{guess['name']}' as defaults" if guess
              else "entered by the user")
    return _save(objects, objects_path, name, props, source)


def ask_yes(question):
    return input(f"   {question} [Y/n]: ").strip().lower() in ("", "y", "yes")
