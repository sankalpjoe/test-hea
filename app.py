from flask import Flask, render_template, Response
import cv2
import numpy as np
from threading import Thread
from queue import Queue, Empty
from model import Model, CATEGORY_META
from ws_stream import WebSocketStream

app = Flask(__name__)
model = Model()

INFER_EVERY = model.settings.get('video-settings', {}).get('inference-every-n-frames', 5)

# Load video sources from settings.yaml
_raw_sources = model.settings.get('video-sources', [{'id': 0, 'name': 'Main Webcam', 'source': 0}])
video_sources = [{'id': s['id'], 'name': s['name']} for s in _raw_sources]
# id → raw source value (int, RTSP/HTTP URL, or ws:// URL)
_source_map: dict[int, any] = {s['id']: s['source'] for s in _raw_sources}


# ── Overlay rendering ─────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    h, w     = frame.shape[:2]
    category = result['category']
    conf     = result['confidence']
    label    = result['label']
    is_alert = result['is_alert']
    count    = result.get('headcount', 0)
    boxes    = result.get('person_boxes', [])
    crowd    = result.get('crowd_alert', False)
    meta     = CATEGORY_META.get(category, CATEGORY_META['normal'])
    color    = meta['color']

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 220, 80), 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    cv2.putText(frame, f"{meta['alert']}  ({conf:.0%})", (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.95, 1)[0]
    cv2.putText(frame, label, (w - lbl_size[0] - 10, 20),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (160, 160, 160), 1, cv2.LINE_AA)

    badge_color = (0, 60, 220) if crowd else (40, 160, 255)
    badge_text  = f'People: {count}' + (' ⚠ CROWD' if crowd else '')
    bw, bh = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    bx, by = w - bw - 18, h - 14
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bx - 8, by - bh - 6), (w - 6, h - 4), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, badge_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2, cv2.LINE_AA)

    if is_alert or crowd:
        cv2.rectangle(frame, (2, 2), (w - 3, h - 3),
                      (0, 60, 220) if crowd and not is_alert else color, 5)
    return frame


# ── VideoProcessor — works for webcam, HTTP/RTSP, AND WebSocket ───────────────
class VideoProcessor:
    """
    Unified processor that auto-detects stream type from the source value:
      - int              → local webcam via cv2.VideoCapture
      - http:// https:// → IP camera MJPEG/HLS via cv2.VideoCapture
      - rtsp://          → RTSP via cv2.VideoCapture
      - ws:// wss://     → WebSocket via WebSocketStream (custom class)
                           cv2.VideoCapture cannot open WebSocket URLs —
                           they use a different protocol than HTTP/RTSP.
    """

    def __init__(self, source):
        self._stopped    = False
        self._frame_q    = Queue(maxsize=2)
        self._result_q   = Queue(maxsize=2)
        self._last_result = {
            'label': '', 'category': 'Unknown', 'confidence': 0.0,
            'is_alert': False, 'headcount': 0, 'person_boxes': [], 'crowd_alert': False
        }
        self._frame_id = 0

        # ── Choose capture backend based on source type ────────────────────
        src_str = str(source).lower()
        if src_str.startswith('ws://') or src_str.startswith('wss://'):
            # WebSocket stream — cv2.VideoCapture cannot handle this protocol
            token = None  # set if your camera needs auth: "Bearer your-token"
            self._capture = WebSocketStream(source, token=token).start()
            self._use_ws  = True
        else:
            # Webcam int, HTTP/HTTPS MJPEG, RTSP — all handled by OpenCV/FFmpeg
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f'Unable to open video source: {source}')
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self._capture = cap
            self._use_ws  = False

        Thread(target=self._capture_loop,   daemon=True).start()
        Thread(target=self._inference_loop, daemon=True).start()

    def _capture_loop(self):
        while not self._stopped:
            if self._use_ws:
                # WebSocketStream.read() returns latest decoded frame or None
                frame = self._capture.read()
                if self._capture.stopped:
                    self._stopped = True
                    break
            else:
                ok, frame = self._capture.read()
                if not ok:
                    self._stopped = True
                    break

            if frame is None:
                continue

            self._frame    = frame
            self._frame_id += 1
            if self._frame_id % INFER_EVERY == 0 and not self._frame_q.full():
                self._frame_q.put(frame.copy())

    def _inference_loop(self):
        while True:
            frame = self._frame_q.get()
            if frame is None:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = model.predict(image=rgb)
            while not self._result_q.empty():
                try: self._result_q.get_nowait()
                except Empty: break
            self._result_q.put(result)

    def get_frame(self):
        try:
            self._last_result = self._result_q.get_nowait()
        except Empty:
            pass
        return getattr(self, '_frame', None), self._last_result

    def stop(self):
        self._stopped = True
        self._frame_q.put(None)
        if self._use_ws:
            self._capture.stop()
        else:
            self._capture.release()


_processors: dict[int, VideoProcessor] = {}

def get_processor(source_id: int) -> VideoProcessor:
    if source_id not in _processors:
        raw_source = _source_map.get(source_id, source_id)
        _processors[source_id] = VideoProcessor(raw_source)
    return _processors[source_id]


def generate_frames(source_id: int):
    processor = get_processor(source_id)
    while not processor._stopped:
        frame, result = processor.get_frame()
        if frame is None:
            continue
        display = draw_overlay(frame.copy(), result)
        _, buffer = cv2.imencode('.jpg', display)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', video_sources=video_sources)

@app.route('/video_feed/<int:source_id>')
def video_feed(source_id):
    try:
        return Response(generate_frames(source_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except ValueError as e:
        return str(e), 404

if __name__ == '__main__':
    app.run(debug=True)
