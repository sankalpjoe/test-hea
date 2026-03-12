from flask import Flask, render_template, Response
import cv2
import numpy as np
from threading import Thread
from queue import Queue, Empty
from model import Model, CATEGORY_META

app = Flask(__name__)
model = Model()

# Load sources from settings.yaml
video_sources = model.settings.get('video-sources', [])
if not video_sources:
    video_sources = [{'id': '0', 'name': 'Main Webcam', 'source': 0}]
else:
    # Standardize sources to ensure they have a 'source' key (index or URL)
    for s in video_sources:
        s['id'] = str(s['id'])
        if 'source' not in s:
            s['source'] = int(s['id']) if s['id'].isdigit() else s['id']

INFER_EVERY = model.settings.get('video-settings', {}).get('inference-every-n-frames', 5)


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

    # ── Person bounding boxes (from YOLOv8) ──────────────────────────────────
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 220, 80), 2)

    # ── Top banner (semi-transparent) ─────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    # Alert / status text
    status = f"{meta['alert']}  ({conf:.0%})"
    cv2.putText(frame, status, (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Matched CLIP label (smaller, right-aligned)
    lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.95, 1)[0]
    cv2.putText(frame, label, (w - lbl_size[0] - 10, 20),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Headcount badge (bottom-right) ────────────────────────────────────────
    badge_color = (0, 60, 220) if crowd else (40, 160, 255)
    badge_text  = f'People: {count}'
    if crowd:
        badge_text += ' ⚠ CROWD'
    bw, bh = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    bx, by = w - bw - 18, h - 14
    # Pill background
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bx - 8, by - bh - 6), (w - 6, h - 4), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, badge_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2, cv2.LINE_AA)

    # Alert border flash
    if is_alert or crowd:
        border_color = (0, 60, 220) if crowd and not is_alert else color
        cv2.rectangle(frame, (2, 2), (w - 3, h - 3), border_color, 5)

    return frame


# ── Threaded capture + async inference ───────────────────────────────────────
class VideoProcessor:
    """
    Two background threads eliminate the two main sources of lag:
      Thread 1 (capture): cv2.read() is blocking I/O — running it in a thread
        means the Flask generator never waits for frame decoding.
        CAP_PROP_BUFFERSIZE=2 prevents stale OS frame queue accumulation.
      Thread 2 (inference): CLIP + YOLOv8 take 80-300ms on CPU. Without a
        thread, each inference call would stall the frame generator, dropping
        the effective stream FPS to 3-12fps. With threading, display runs at
        full camera FPS; inference publishes results as fast as it can.
    """

    def __init__(self, source):
        self._source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f'Unable to open video source: {source}')
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self._frame      = None
        self._stopped    = False
        self._frame_q    = Queue(maxsize=2)
        self._result_q   = Queue(maxsize=2)
        self._last_result = {
            'label': '', 'category': 'Unknown', 'confidence': 0.0,
            'is_alert': False, 'headcount': 0, 'person_boxes': [], 'crowd_alert': False
        }
        self._frame_id = 0

        Thread(target=self._capture_loop,   daemon=True).start()
        Thread(target=self._inference_loop, daemon=True).start()

    def _capture_loop(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                # Add "Signal Lost" indicator
                h, w = 480, 640  # Default fallback size
                f = self._frame
                if f is not None:
                    h, w = f.shape[:2]
                lost_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(lost_frame, 'SIGNAL LOST', (w // 4, h // 2), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 50, 200), 3)
                self._frame = lost_frame
                # Try to reconnect every 5 seconds if ok is false
                import time
                time.sleep(5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self._source)
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
        return self._frame, self._last_result

    def stop(self):
        self._stopped = True
        self._frame_q.put(None)
        self.cap.release()


_processors: dict[int, VideoProcessor] = {}

def get_processor(source_id: str) -> VideoProcessor:
    if source_id not in _processors:
        # Find the actual source (index or URL) from our list
        source_config = next((s for s in video_sources if s['id'] == source_id), None)
        if not source_config:
            raise ValueError(f"Source ID {source_id} not found")
        
        _processors[source_id] = VideoProcessor(source_config['source'])
    return _processors[source_id]


def generate_frames(source_id: str):
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

@app.route('/video_feed/<source_id>')
def video_feed(source_id):
    try:
        return Response(generate_frames(str(source_id)),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return str(e), 404

@app.route('/add_source', methods=['POST'])
def add_source():
    from flask import request, redirect, url_for
    name = request.form.get('name')
    source = request.form.get('source')
    
    if not name or not source:
        return "Missing name or source", 400
    
    # Try to convert source to int if it's a digit (webcam index)
    actual_source = int(source) if source.isdigit() else source
    
    source_id = str(len(video_sources))
    video_sources.append({
        'id': source_id,
        'name': name,
        'source': actual_source
    })
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)