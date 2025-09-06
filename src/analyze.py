import cv2, numpy as np, os, argparse
from skimage import measure, morphology
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

def load_img(path):
    bgr = cv2.imread(path)
    if bgr is None: raise ValueError(f"Cannot read image: {path}")
    return bgr

def skin_mask(bgr):
    # HSV skin segmentation (simple, not perfect)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60]); upper = np.array([25, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([160, 30, 60]); upper2 = np.array([179, 255, 255])
    mask = cv2.bitwise_or(mask1, cv2.inRange(hsv, lower2, upper2))
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=500)
    mask = morphology.remove_small_objects(mask, min_size=500)
    return (mask.astype(np.uint8) * 255)

def redness_map(bgr):
    # Use LAB a* channel (red-green) to highlight redness
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:,:,1].astype(np.float32)
    # Normalize
    a_norm = (a - a.min()) / (a.max()-a.min()+1e-6)
    return (a_norm*255).astype(np.uint8)

def detect_blemishes(bgr, skin):
    red = redness_map(bgr)
    red_skin = cv2.bitwise_and(red, red, mask=skin)
    # Threshold: top 15% redness on skin
    thr = np.percentile(red_skin[skin>0], 85) if np.any(skin>0) else 255
    hot = (red_skin >= thr).astype(np.uint8)*255
    hot = cv2.medianBlur(hot, 5)
    hot = morphology.remove_small_objects(hot.astype(bool), min_size=80)
    hot = morphology.remove_small_holes(hot, area_threshold=80)
    hot = (hot.astype(np.uint8)*255)
    return hot

def overlay_heatmap(bgr, hot):
    heat = cv2.applyColorMap(cv2.normalize(hot,None,0,255,cv2.NORM_MINMAX), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.7, heat, 0.3, 0)
    return overlay

def score_acne(hot, skin):
    skin_px = max(int(np.sum(skin>0)), 1)
    hot_px  = int(np.sum(hot>0))
    pct = 100.0 * hot_px / skin_px
    if   pct < 1: level = "clear"
    elif pct < 3: level = "mild"
    elif pct < 7: level = "moderate"
    else:         level = "pronounced"
    return pct, level

def recommend(level):
    tips = {
        "clear":[
            "Gentle cleanse 1–2×/day; non-comedogenic moisturizer.",
            "Daily broad-spectrum SPF 30+."
        ],
        "mild":[
            "Introduce 2–3×/week salicylic acid (BHA) or low-% benzoyl peroxide.",
            "Avoid heavy oils; keep hands off face."
        ],
        "moderate":[
            "Daily BHA + spot treat benzoyl peroxide 2.5%.",
            "Consider niacinamide serum; change pillowcases often."
        ],
        "pronounced":[
            "Use BHA + benzoyl peroxide routine; simplify other products.",
            "If persistent/irritated, consult a dermatologist (this is not medical advice)."
        ],
    }
    return tips[level]

def main(img_path):
    os.makedirs("out", exist_ok=True)
    bgr = load_img(img_path)
    skin = skin_mask(bgr)
    hot  = detect_blemishes(bgr, skin)
    overlay = overlay_heatmap(bgr, hot)

    out_img = "out/overlay.png"
    cv2.imwrite(out_img, overlay)

    pct, level = score_acne(hot, skin)

    env = Environment(loader=FileSystemLoader("templates"))
    tpl = env.get_template("report.html")
    html = tpl.render(
        generated=datetime.utcnow().isoformat()+"Z",
        image=os.path.basename(out_img), severity=level, pct=f"{pct:.2f}%"
    )
    with open("out/report.html","w") as f: f.write(html)
    print(f"Saved: {out_img} and out/report.html  (severity={level}, area={pct:.2f}%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to face image")
    args = ap.parse_args()
    main(args.image)

