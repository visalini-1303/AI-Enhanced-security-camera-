"""
AUTHENCAM - multi-person face enrollment + live authentication dashboard.
Enrollment ADDS people to a database folder (it never overwrites older ones).
Authentication detects every face in frame, gives each one its own colour,
and shows one status card per person in the right-hand sidebar.

Files created:
    faces/<name>/<name>_1.jpg, <name>_2.jpg ...      enrolled photos
    (legacy my_face.jpg + person_name.txt are imported automatically)

Controls while authenticating:
    q  quit          d  toggle distance numbers
    +  more lenient  -  stricter
"""

import cv2
import numpy as np
from deepface import DeepFace
from collections import deque
from datetime import datetime
import os
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_DIR = "faces"
LEGACY_IMG = "my_face.jpg"
LEGACY_NAME = "person_name.txt"
TEMP_FACE = "_temp_auth_face.jpg"

DISTANCE_THRESHOLD = 0.35
HISTORY_LEN = 12
REQUIRED_MATCHES = 7
PROCESS_EVERY_N = 3

# Dashboard palette (BGR) - deco blueprint look
BG        = (30, 38, 24)
PANEL     = (38, 50, 32)
GOLD      = (150, 205, 224)
GOLD_DIM  = (105, 145, 160)
GREEN     = (150, 200, 150)
RED       = (110, 110, 235)
GRID      = (44, 58, 38)

# Per-person box colours (BGR)
COLORS = [
    (120, 220, 120),   # green
    (60, 170, 255),    # orange
    (255, 190, 90),    # cyan-blue
    (220, 120, 255),   # pink
    (230, 230, 100),   # aqua
    (140, 140, 255),   # salmon
]

CANVAS_W, CANVAS_H = 1280, 720
VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H = 40, 100, 780, 560
SIDE_X, SIDE_W = 850, 390

# Restricted zone, in video-panel coordinates
ZONE = (520, 40, 240, 180)  # x, y, w, h


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))


def embed_crop(face_crop_bgr):
    """Normalized embedding for an already-cropped face, or None."""
    try:
        cv2.imwrite(TEMP_FACE, face_crop_bgr)
        reps = DeepFace.represent(
            img_path=TEMP_FACE,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="skip",
            align=True,
        )
        if not reps:
            return None
        emb = np.array(reps[0]["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        return None if n < 1e-6 else emb / n
    except Exception:
        return None


def embed_file(path):
    """Normalized embedding for a photo on disk (runs face detection)."""
    for backend in ("opencv", "skip"):
        try:
            rep = DeepFace.represent(
                img_path=path,
                model_name="Facenet",
                enforce_detection=(backend == "opencv"),
                detector_backend=backend,
                align=True,
            )[0]["embedding"]
            emb = np.array(rep, dtype=np.float32)
            n = np.linalg.norm(emb)
            if n > 1e-6:
                return emb / n
        except Exception:
            continue
    return None


def cleanup():
    if os.path.exists(TEMP_FACE):
        os.remove(TEMP_FACE)


def open_camera():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap


def cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


# ---------------------------------------------------------------------------
# Face database (multi-person, additive)
# ---------------------------------------------------------------------------
def import_legacy():
    """Move an old single-face enrollment into the database folder."""
    if os.path.exists(LEGACY_IMG) and os.path.exists(LEGACY_NAME):
        with open(LEGACY_NAME) as f:
            name = f.read().strip()
        if name:
            person_dir = os.path.join(DB_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            target = os.path.join(person_dir, f"{name}_legacy.jpg")
            if not os.path.exists(target):
                img = cv2.imread(LEGACY_IMG)
                if img is not None:
                    cv2.imwrite(target, img)
                    print(f"Imported previous enrollment '{name}' into the database.")


def list_people():
    if not os.path.isdir(DB_DIR):
        return []
    return sorted(
        d for d in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, d))
    )


def save_enrollment(name, frame):
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    idx = len(glob.glob(os.path.join(person_dir, "*.jpg"))) + 1
    path = os.path.join(person_dir, f"{name}_{idx}.jpg")
    cv2.imwrite(path, frame)
    return path


def load_database():
    """{name: [embedding, ...]} for everyone ever enrolled."""
    import_legacy()
    db = {}
    for name in list_people():
        embs = []
        for path in sorted(glob.glob(os.path.join(DB_DIR, name, "*.jpg"))):
            e = embed_file(path)
            if e is not None:
                embs.append(e)
        if embs:
            db[name] = embs
            print(f"  {name}: {len(embs)} photo(s) loaded")
        else:
            print(f"  {name}: no usable photo (re-enroll this person)")
    return db


def identify(emb, db, threshold):
    """Return (name, distance) for the closest enrolled person."""
    best_name, best_dist = None, 1.0
    for name, embs in db.items():
        for ref in embs:
            d = cosine_distance(ref, emb)
            if d < best_dist:
                best_dist, best_name = d, name
    if best_name is not None and best_dist < threshold:
        return best_name, best_dist
    return None, best_dist


# ---------------------------------------------------------------------------
# Enrollment (additive)
# ---------------------------------------------------------------------------
def enroll():
    cap = open_camera()
    if not cap.isOpened():
        print("Camera not found!")
        return False

    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty!")
        cap.release()
        cv2.destroyAllWindows()
        return False

    existing = os.path.isdir(os.path.join(DB_DIR, name))
    print(f"\n{'Adding another photo for' if existing else 'Enrolling'}: {name}")
    print("Press 's' when the green box is visible, 'q' to finish.\n")

    det = cascade()
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.3, 5)
        display = frame.copy()

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            msg, col = "FACE DETECTED - press 's' to save", (0, 255, 0)
        else:
            msg, col = "No face detected - come closer", (0, 0, 255)

        cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(display, f"{name}  |  saved this session: {saved}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Enroll Face", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(faces) > 0:
                path = save_enrollment(name, frame)
                saved += 1
                print(f"Saved {path}  (take a few from different angles)")
            else:
                print("No face detected. Position your face properly.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved > 0 or existing


# ---------------------------------------------------------------------------
# Dashboard drawing
# ---------------------------------------------------------------------------
def draw_grid(canvas):
    for x in range(0, CANVAS_W, 40):
        cv2.line(canvas, (x, 0), (x, CANVAS_H), GRID, 1)
    for y in range(0, CANVAS_H, 40):
        cv2.line(canvas, (0, y), (CANVAS_W, y), GRID, 1)


def draw_corners(canvas):
    m, L = 14, 40
    pts = [(m, m, 1, 1), (CANVAS_W - m, m, -1, 1),
           (m, CANVAS_H - m, 1, -1), (CANVAS_W - m, CANVAS_H - m, -1, -1)]
    for (x, y, sx, sy) in pts:
        cv2.line(canvas, (x, y), (x + sx * L, y), GOLD, 2)
        cv2.line(canvas, (x, y), (x, y + sy * L), GOLD, 2)
    cv2.rectangle(canvas, (m, m), (CANVAS_W - m, CANVAS_H - m), GOLD_DIM, 1)


def dashed_rect(img, x, y, w, h, color, dash=10, thick=1):
    for i in range(x, x + w, dash * 2):
        cv2.line(img, (i, y), (min(i + dash, x + w), y), color, thick)
        cv2.line(img, (i, y + h), (min(i + dash, x + w), y + h), color, thick)
    for i in range(y, y + h, dash * 2):
        cv2.line(img, (x, i), (x, min(i + dash, y + h)), color, thick)
        cv2.line(img, (x + w, i), (x + w, min(i + dash, y + h)), color, thick)


def draw_card(canvas, x, y, w, h, label, value, note, accent):
    sub = canvas[y:y + h, x:x + w]
    overlay = np.full(sub.shape, PANEL, dtype=np.uint8)
    canvas[y:y + h, x:x + w] = cv2.addWeighted(sub, 0.35, overlay, 0.65, 0)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), GOLD_DIM, 1)
    cv2.line(canvas, (x, y), (x + w, y), accent, 2)
    cv2.line(canvas, (x, y), (x, y + h), accent, 3)

    cv2.putText(canvas, label.lower(), (x + 14, y + 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, GOLD_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, value.upper(), (x + 14, y + 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, accent, 2, cv2.LINE_AA)
    if note:
        cv2.putText(canvas, note, (x + 14, y + h - 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.95, GOLD_DIM, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Simple IoU tracker so each person keeps a stable colour + history
# ---------------------------------------------------------------------------
class Tracker:
    def __init__(self):
        self.tracks = {}   # id -> dict(box, missed, history, name, dist, color)
        self.next_id = 0

    @staticmethod
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def update(self, boxes):
        """Match detections to tracks; returns list of (track_id, box)."""
        assigned, used = [], set()
        for box in boxes:
            best_id, best_iou = None, 0.25
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                score = self.iou(box, t["box"])
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "box": box, "missed": 0,
                    "history": deque([False] * HISTORY_LEN, maxlen=HISTORY_LEN),
                    "names": deque(maxlen=HISTORY_LEN),
                    "name": None, "dist": 1.0,
                    "color": COLORS[best_id % len(COLORS)],
                }
            self.tracks[best_id]["box"] = box
            self.tracks[best_id]["missed"] = 0
            used.add(best_id)
            assigned.append((best_id, box))

        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > 15:
                    del self.tracks[tid]
        return assigned


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def authenticate():
    global DISTANCE_THRESHOLD

    print("\nLoading enrolled faces...")
    db = load_database()
    if not db:
        print("No enrolled faces found. Enroll someone first.")
        return
    print(f"Database ready: {len(db)} person(s).\n")

    cap = open_camera()
    if not cap.isOpened():
        print("No camera found!")
        return

    det = cascade()
    tracker = Tracker()
    frame_counter = 0
    show_distance = False
    breach_log = "system standing by"
    window = "AUTHENCAM"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, CANVAS_W, CANVAS_H)

    print("Controls:  q quit   d distances   + lenient   - stricter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1

        video = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        boxes = [tuple(int(v) for v in b) for b in
                 det.detectMultiScale(gray, 1.2, 5, minSize=(70, 70))]
        tracked = tracker.update(boxes)

        run_heavy = frame_counter % PROCESS_EVERY_N == 0
        zx, zy, zw, zh = ZONE
        breach = False

        for tid, (x, y, w, h) in tracked:
            t = tracker.tracks[tid]
            if run_heavy:
                pad = int(0.2 * min(w, h))
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2 = min(video.shape[1], x + w + pad)
                y2 = min(video.shape[0], y + h + pad)
                emb = embed_crop(video[y1:y2, x1:x2])
                if emb is None:
                    t["history"].append(False)
                else:
                    name, dist = identify(emb, db, DISTANCE_THRESHOLD)
                    t["dist"] = dist
                    t["history"].append(name is not None)
                    if name:
                        t["names"].append(name)
            # smoothed identity
            ok = sum(t["history"]) >= REQUIRED_MATCHES
            if ok and t["names"]:
                t["name"] = max(set(t["names"]), key=list(t["names"]).count)
            elif not ok:
                t["name"] = None

            if (x < zx + zw and x + w > zx and y < zy + zh and y + h > zy):
                breach = True

        # ---------------- canvas ----------------
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        draw_grid(canvas)
        draw_corners(canvas)

        cv2.putText(canvas, "AUTHENCAM", (44, 66),
                    cv2.FONT_HERSHEY_DUPLEX, 1.25, GOLD, 2, cv2.LINE_AA)
        now = datetime.now()
        cv2.putText(canvas, now.strftime("%d %b %Y").upper(), (CANVAS_W - 220, 44),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, GOLD_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, now.strftime("%I:%M:%S %p"), (CANVAS_W - 220, 68),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, GOLD_DIM, 1, cv2.LINE_AA)
        cv2.line(canvas, (40, 84), (CANVAS_W - 40, 84), GOLD_DIM, 1)

        # boxes on the video panel
        dashed_rect(video, zx, zy, zw, zh, RED, 10, 1)
        cv2.putText(video, "restricted zone", (zx + 18, zy + zh // 2),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, RED, 1, cv2.LINE_AA)

        for tid, (x, y, w, h) in tracked:
            t = tracker.tracks[tid]
            col = t["color"]
            cv2.rectangle(video, (x, y), (x + w, y + h), col, 2)
            label = t["name"] if t["name"] else "unknown"
            cv2.putText(video, label.lower(), (x, max(y - 10, 16)),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, col, 1, cv2.LINE_AA)
            if show_distance:
                cv2.putText(video, f"{t['dist']:.3f}", (x, y + h + 18),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, col, 1, cv2.LINE_AA)

        canvas[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = video
        cv2.rectangle(canvas, (VIDEO_X - 2, VIDEO_Y - 2),
                      (VIDEO_X + VIDEO_W + 2, VIDEO_Y + VIDEO_H + 2), GOLD_DIM, 1)

        # ---------------- sidebar ----------------
        y_pos = VIDEO_Y
        draw_card(canvas, SIDE_X, y_pos, SIDE_W, 78, "people detected",
                  str(len(tracked)), "", GOLD)
        y_pos += 94

        if breach:
            breach_log = f"breach at {now.strftime('%I:%M:%S %p')}"
        draw_card(canvas, SIDE_X, y_pos, SIDE_W, 92, "perimeter status",
                  "BREACH" if breach else "SECURE",
                  "subject inside restricted bounds" if breach
                  else "no breach of restricted bounds recorded",
                  RED if breach else GOLD)
        y_pos += 108

        if not tracked:
            draw_card(canvas, SIDE_X, y_pos, SIDE_W, 92, "identity",
                      "NO SUBJECT", "camera field is clear", GOLD_DIM)
        else:
            for tid, _ in tracked:
                if y_pos + 92 > VIDEO_Y + VIDEO_H:
                    break
                t = tracker.tracks[tid]
                if t["name"]:
                    value = t["name"]
                    note = "identity confirmed - access granted"
                else:
                    value = "UNAUTHENTICATED"
                    note = "subject not matched to any known record"
                draw_card(canvas, SIDE_X, y_pos, SIDE_W, 92,
                          f"identity / subject {tid + 1}", value, note, t["color"])
                y_pos += 104

        # footer
        cv2.line(canvas, (40, CANVAS_H - 60), (CANVAS_W - 40, CANVAS_H - 60), GOLD_DIM, 1)
        cv2.putText(canvas, f"log - {breach_log}", (44, CANVAS_H - 34),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, GOLD_DIM, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"threshold {DISTANCE_THRESHOLD:.2f}  |  system nominal",
                    (CANVAS_W - 330, CANVAS_H - 34),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, GOLD_DIM, 1, cv2.LINE_AA)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_distance = not show_distance
        elif key in (ord('+'), ord('=')):
            DISTANCE_THRESHOLD = min(0.8, DISTANCE_THRESHOLD + 0.02)
        elif key == ord('-'):
            DISTANCE_THRESHOLD = max(0.1, DISTANCE_THRESHOLD - 0.02)

    cap.release()
    cv2.destroyAllWindows()
    cleanup()


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DB_DIR, exist_ok=True)
    import_legacy()
    people = list_people()

    if people:
        print("Enrolled people: " + ", ".join(people))
    else:
        print("No one enrolled yet.")

    print("\nOptions:")
    print("  1 = Authenticate")
    print("  2 = Add / enroll another person (keeps everyone already enrolled)")
    print("  q = Quit")
    choice = input("Choose: ").strip().lower()

    if choice == 'q':
        return
    if choice == '2' or not people:
        if not enroll():
            return
        print("\nEnrollment saved. Starting authentication...")
    authenticate()


if __name__ == "__main__":
    main()
