from flask import Flask, render_template, request, redirect, jsonify
from jinja2 import Environment, FileSystemLoader
import os, cv2, numpy as np
from skimage import morphology
from datetime import datetime, timezone
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB upload cap

# Folders
TEMPLATE_DIR = "templates"
STATIC_DIR   = "static"
UPLOAD_DIR   = os.path.join(STATIC_DIR, "uploads")
OUT_DIR      = os.path.join(STATIC_DIR, "out")
for d in (STATIC_DIR, UPLOAD_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

ALLOWED = {"jpg", "jpeg", "png"}

# ---------- helpers ----------
def load_img(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return bgr

def preprocess(bgr):
    """
    Quick lighting normalization:
    - Gray-world white balance
    - CLAHE on L channel to stabilize contrast
    """
    f = bgr.astype(np.float32)
    avg = f.mean(axis=(0, 1), keepdims=True)
    scale = avg.mean() / (avg + 1e-6)
    balanced = np.clip(f * scale, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge([L, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def safe_resize(bgr, max_side=1280):
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    s = max_side / float(m)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def skin_mask(bgr):
    # Simple HSV skin segmentation
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 60]); upper1 = np.array([25, 255, 255])
    lower2 = np.array([160, 30, 60]); upper2 = np.array([179, 255, 255])
    m = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    m = morphology.remove_small_holes(m.astype(bool), area_threshold=500)
    m = morphology.remove_small_objects(m, min_size=500)
    return (m.astype(np.uint8) * 255)

def face_mask_ellipse(bgr, exclude_lips=True):
    """
    Largest-face ellipse + optional lip/moustache exclusions.
    Falls back to full mask if no face is found.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
    )
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(faces) == 0:
        mask[:] = 255
        return mask

    # Pick largest face
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    # Main face ellipse
    center = (x + fw // 2, y + fh // 2)
    axes   = (int(fw * 0.48), int(fh * 0.60))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    if exclude_lips:
        # Mouth box (lower-center); tweak factors if needed
        mx1 = x + int(0.18 * fw);  mx2 = x + int(0.82 * fw)
        my1 = y + int(0.58 * fh);  my2 = y + int(0.82 * fh)
        cv2.rectangle(mask, (mx1, my1), (mx2, my2), 0, -1)

        # Small moustache band just above lips
        sx1 = x + int(0.25 * fw);  sx2 = x + int(0.75 * fw)
        sy1 = y + int(0.48 * fh);  sy2 = y + int(0.60 * fh)
        cv2.rectangle(mask, (sx1, sy1), (sx2, sy2), 0, -1)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask

def redness_map(bgr):
    # LAB a* channel captures red-green
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return (a * 255).astype(np.uint8)

def detect_blemishes(bgr, mask, percentile=85):
    red = redness_map(bgr)
    red_skin = cv2.bitwise_and(red, red, mask=mask)
    thr = np.percentile(red_skin[mask > 0], percentile) if np.any(mask > 0) else 255
    hot = (red_skin >= thr).astype(np.uint8) * 255
    hot = cv2.medianBlur(hot, 5)
    hot = morphology.remove_small_objects(hot.astype(bool), min_size=80)
    hot = morphology.remove_small_holes(hot, area_threshold=80)
    return (hot.astype(np.uint8) * 255)

def overlay_heatmap(bgr, hot):
    heat = cv2.applyColorMap(
        cv2.normalize(hot, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(bgr, 0.7, heat, 0.3, 0)

def score_acne(hot, mask):
    skin_px = max(int(np.sum(mask > 0)), 1)
    hot_px = int(np.sum(hot > 0))
    pct = 100.0 * hot_px / skin_px
    if pct < 1: level = "clear"
    elif pct < 3: level = "mild"
    elif pct < 7: level = "moderate"
    else: level = "pronounced"
    return pct, level

def compute_confidence(mask, bgr):
    """
    Confidence [0–1]:
    - larger analyzed area = higher confidence
    - more even lighting (lower stdev) = higher confidence
    """
    total_px = mask.size
    used_px  = np.sum(mask > 0)
    area_ratio = used_px / total_px  # 0–1

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    std_brightness = np.std(gray[mask > 0]) if np.any(mask > 0) else 0.0
    lighting_score = max(0.0, 1.0 - (std_brightness / 64.0))  # cap

    confidence = 0.5 * area_ratio + 0.5 * lighting_score
    return round(float(min(max(confidence, 0.0), 1.0)), 2)

def generate_report(overlay_path, severity, pct, confidence):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    tpl = env.get_template("report.html")
    html = tpl.render(
        generated=datetime.now(timezone.utc).isoformat(),
        image=os.path.basename(overlay_path),
        severity=severity,
        pct=f"{pct:.2f}%",
        confidence=confidence
    )
    out_html = os.path.join(OUT_DIR, "report.html")
    with open(out_html, "w") as f:
        f.write(html)
    return out_html

# ---------- routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("photo")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select an image.")
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED:
            return render_template("index.html", error="Only JPG/PNG images are allowed.")

        fname = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_DIR, fname)
        file.save(upload_path)

        percentile = int(request.form.get("sensitivity", 85))

        # --- analysis ---
        bgr = load_img(upload_path)
        bgr = preprocess(bgr)
        bgr = safe_resize(bgr)

        skin = skin_mask(bgr)
        face = face_mask_ellipse(bgr)
        analysis_mask = cv2.bitwise_and(skin, face)

        hot = detect_blemishes(bgr, analysis_mask, percentile=percentile)
        overlay = overlay_heatmap(bgr, hot)

        # outputs
        overlay_path = os.path.join(OUT_DIR, "overlay.png")
        cv2.imwrite(overlay_path, overlay)

        # DEBUG: visualize mask (dim excluded zones)
        dbg = bgr.copy()
        dbg[analysis_mask == 0] = (dbg[analysis_mask == 0] * 0.3).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, "debug_mask.png"), dbg)

        pct, level = score_acne(hot, analysis_mask)
        confidence = compute_confidence(analysis_mask, bgr)
        generate_report(overlay_path, level, pct, confidence)

        # privacy: delete uploaded original
        try:
            os.remove(upload_path)
        except Exception:
            pass

        return redirect("/report")

    return render_template("index.html", error=None)

@app.route("/report")
def report():
    return render_template("report_link.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)

