# server.py
import threading
import time
import sqlite3
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np

from plyer import notification
from flask import Flask, jsonify

app = Flask(__name__)
DB_PATH = "blinks.db"

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS blinks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL
                  )""")
    conn.commit()
    conn.close()

def record_blink(ts: datetime):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO blinks (ts) VALUES (?)", (ts.isoformat(),))
    conn.commit()
    conn.close()

def get_blinks_since(dt: datetime):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ts FROM blinks WHERE ts >= ?", (dt.isoformat(),))
    rows = cur.fetchall()
    conn.close()
    return [datetime.fromisoformat(r[0]) for r in rows]

# ---------- Blink detection ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 385, 263, 373, 380]

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio(pts, eye):
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in eye]
    v1 = dist(p2, p5)
    v2 = dist(p3, p6)
    h = dist(p1, p4)
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)

class BlinkMonitor(threading.Thread):
    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.running = False
        self.capture = None

        self.blink_count = 0
        self.eye_closed = False
        self.last_blink_time = None

        self.window_seconds = 60
        self.lock = threading.Lock()

        self.ear_threshold = None
        self.calibration_frames = 50
        self.calibration_values = []

        self.min_blinks_per_min = 12

    def auto_calibrate_threshold(self):
        if len(self.calibration_values) < 10:
            return
        avg_open_ear = np.mean(self.calibration_values)
        self.ear_threshold = avg_open_ear * 0.7
        print(f"[INFO] EAR threshold calibrated: {self.ear_threshold:.3f}")

    def run(self):
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            print(f"[ERROR] Cannot open webcam with source {self.source}")
            self.running = False
            return

        self.running = True
        print("[INFO] Blink monitor started...")

        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("[WARNING] Frame capture failed, retrying...")
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            lm = results.multi_face_landmarks[0]
            pts = [(p.x * w, p.y * h) for p in lm.landmark]

            left_ear = eye_aspect_ratio(pts, LEFT_EYE)
            right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            if self.ear_threshold is None:
                self.calibration_values.append(ear)
                if len(self.calibration_values) >= self.calibration_frames:
                    self.auto_calibrate_threshold()
                continue

            if ear < self.ear_threshold:
                if not self.eye_closed:
                    self.eye_closed = True
            else:
                if self.eye_closed:
                    self.eye_closed = False
                    blink_time = datetime.utcnow()
                    record_blink(blink_time)

                    with self.lock:
                        self.blink_count += 1

                    self.last_blink_time = blink_time
                    self.check_notify()

            time.sleep(0.01)

        if self.capture:
            self.capture.release()
        print("[INFO] Blink monitor stopped.")

    def check_notify(self):
        window_start = datetime.utcnow() - timedelta(seconds=self.window_seconds)
        blinks = get_blinks_since(window_start)
        bpm = len(blinks) * (60 / self.window_seconds)

        if bpm < self.min_blinks_per_min:
            try:
                notification.notify(
                    title="Just-Blink Reminder",
                    message=f"Blink rate {bpm:.1f} < threshold ({self.min_blinks_per_min})",
                    timeout=5
                )
            except Exception as e:
                print(f"[WARNING] Notification failed: {e}")

    def get_stats(self):
        window_start = datetime.utcnow() - timedelta(seconds=self.window_seconds)
        blinks = get_blinks_since(window_start)
        return {
            "blinks_per_min": len(blinks) * (60 / self.window_seconds),
            "last_window_count": len(blinks),
            "total_recorded": self.blink_count,
            "ear_threshold": self.ear_threshold,
            "min_blinks_per_min": self.min_blinks_per_min
        }

    def stop(self):
        self.running = False

monitor = BlinkMonitor(source=0)

# ---------- API ----------
@app.route('/start', methods=['POST'])
def start():
    global monitor
    if not monitor.running:
        monitor = BlinkMonitor(source=0)
        monitor.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})

@app.route('/stop', methods=['POST'])
def stop():
    global monitor
    if monitor.running:
        monitor.stop()
        return jsonify({"status": "stopping"})
    return jsonify({"status": "not_running"})

@app.route('/status', methods=['GET'])
def status():
    global monitor
    if not monitor.running:
        return jsonify({"running": False})

    stats = monitor.get_stats()
    return jsonify({
        "running": True,
        **stats
    })

@app.route('/stat', methods=['GET'])
def stat():
    global monitor
    stats = monitor.get_stats()
    return jsonify({
        "running": monitor.running,
        "blinks": stats["total_recorded"],
        "bpm": stats["blinks_per_min"]
    })

if __name__ == '__main__':
    init_db()
    print("[INFO] Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=False)
