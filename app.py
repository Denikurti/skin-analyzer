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

# ---------- face detection / masks ----------
def detect_face_box(bgr):
    """Return largest face box (x, y, w, h) or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])

def face_mask_ellipse(bgr, exclude_lips=True, box=None):
    """
    Largest-face ellipse + optional lip/moustache exclusions.
    Falls back to full mask if no face is found.
    """
    if box is None:
        box = detect_face_box(bgr)

    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if box is None:
        mask[:] = 255
        return mask

    x, y, fw, fh = box
    center = (x + fw // 2, y + fh // 2)
    axes   = (int(fw * 0.48), int(fh * 0.60))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    if exclude_lips:
        # Mouth box (lower-center)
        mx1 = x + int(0.18 * fw);  mx2 = x + int(0.82 * fw)
        my1 = y + int(0.58 * fh);  my2 = y + int(0.82 * fh)
        cv2.rectangle(mask, (mx1, my1), (mx2, my2), 0, -1)

        # Small moustache band just above lips
        sx1 = x + int(0.25 * fw);  sx2 = x + int(0.75 * fw)
        sy1 = y + int(0.48 * fh);  sy2 = y + int(0.60 * fh)
        cv2.rectangle(mask, (sx1, sy1), (sx2, sy2), 0, -1)

    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask

# ---------- redness features ----------
def cheek_baseline_a(bgr, box):
    """
    Estimate LAB 'a*' baseline from cheek patches inside face box.
    Returns (mean, std).
    """
    x, y, fw, fh = box
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)

    # Cheek windows relative to face box (tweakable)
    lx1 = x + int(0.15 * fw); lx2 = x + int(0.40 * fw)
    rx1 = x + int(0.60 * fw); rx2 = x + int(0.85 * fw)
    cy1 = y + int(0.45 * fh); cy2 = y + int(0.70 * fh)

    left  = a[cy1:cy2, lx1:lx2]
    right = a[cy1:cy2, rx1:rx2]

    def _safe_stats(arr, fallback):
        if arr.size == 0: return fallback
        mean = float(np.mean(arr)); std = float(np.std(arr))
        if std < 1e-6: std = 1.0
        return mean, std

    meanL, stdL = _safe_stats(left,  (float(np.mean(a)), float(np.std(a)) + 1e-6))
    meanR, stdR = _safe_stats(right, (float(np.mean(a)), float(np.std(a)) + 1e-6))

    mean = (meanL + meanR) / 2.0
    std  = (stdL + stdR) / 2.0
    if std < 1e-6: std = 1.0
    return mean, std

def detect_blemishes(bgr, mask, percentile=85, face_box=None):
    """
    Δa*-aware hotspot detection:
    - If face_box provided, compute z-score relative to cheek baseline (mean/std).
    - Threshold by percentile inside the analysis mask.
    - Clean with median filter + small object/holes removal.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)

    if face_box is not None:
        mean, std = cheek_baseline_a(bgr, face_box)
        metric = (a - mean) / (std + 1e-6)   # z-score (relative redness)
    else:
        # Fallback: use absolute a* (normalized)
        amin, amax = a.min(), a.max()
        metric = (a - amin) / (amax - amin + 1e-6)

    # Percentile threshold inside the mask
    vals = metric[mask > 0]
    thr = np.percentile(vals, percentile) if vals.size else np.inf
    hot = (metric >= thr) & (mask > 0)

    # Denoise/cleanup
    hot = cv2.medianBlur(hot.astype(np.uint8) * 255, 5)
    hot_bool = hot.astype(bool)
    hot_bool = morphology.remove_small_objects(hot_bool, min_size=80)
    hot_bool = morphology.remove_small_holes(hot_bool, area_threshold=80)
    return (hot_bool.astype(np.uint8) * 255)

# ---------- scoring / rendering ----------
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
    lighting_score = max(0.0, 1.0 - (std_brightness / 64.0))

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

        # face + masks
        face_box = detect_face_box(bgr)
        face = face_mask_ellipse(bgr, exclude_lips=True, box=face_box)
        skin = skin_mask(bgr)
        analysis_mask = cv2.bitwise_and(skin, face)

        hot = detect_blemishes(bgr, analysis_mask, percentile=percentile, face_box=face_box)
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

