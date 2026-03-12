from flask import Flask, render_template, Response
import cv2
import numpy as np
from threading import Thread
from queue import Queue, Empty
from model import Model, CATEGORY_META
from ws_stream import WebSocketStream
from logger import SafetyLogger

app = Flask(__name__)
# Show real exception messages instead of "stream error after headers sent"
app.config['PROPAGATE_EXCEPTIONS'] = True
model = Model()

# Start logger — reads interval from settings.yaml
_log_interval = model.settings.get('logging-settings', {}).get('interval-minutes', 5)
logger = SafetyLogger(interval_minutes=_log_interval).start()

INFER_EVERY = model.settings.get('video-settings', {}).get('inference-every-n-frames', 5)

# Load video sources from settings.yaml
_raw_sources = model.settings.get('video-sources', [{'id': 0, 'name': 'Main Webcam', 'source': 0}])
video_sources = [{'id': s['id'], 'name': s['name']} for s in _raw_sources]
# id → raw source value (int, RTSP/HTTP URL, or ws:// URL)
_source_map: dict[int, any]  = {s['id']: s['source'] for s in _raw_sources}
_name_map:   dict[int, str]  = {s['id']: s['name']   for s in _raw_sources}


# ── Overlay rendering ─────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    h, w          = frame.shape[:2]
    category      = result['category']
    conf          = result['confidence']
    label         = result['label']
    is_alert      = result['is_alert']
    count         = result.get('headcount', 0)
    boxes         = result.get('person_boxes', [])
    rejected      = result.get('rejected_boxes', [])
    crowd         = result.get('crowd_alert', False)
    crush_zones   = result.get('crush_zones', [])
    panic         = result.get('panic_detected', False)
    flow_vectors  = result.get('flow_vectors', [])
    zone_alerts   = result.get('zone_alerts', [])
    meta          = CATEGORY_META.get(category, CATEGORY_META['normal'])
    color         = meta['color']

    # Rejected boxes (filtered out) — draw in grey so operator can see what
    # was detected but suppressed by pose/temporal/static filters
    for (x1, y1, x2, y2) in rejected:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
        cv2.putText(frame, 'x', (x1+2, y1+12),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (80,80,80), 1)

    # Confirmed person boxes — green
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)

    # Crush zone hotspots — pulsing red circle
    for (cx, cy, density) in crush_zones:
        cv2.circle(frame, (cx, cy), 40, (0, 0, 255), 3)
        cv2.putText(frame, f'CRUSH {density}p', (cx-30, cy-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

    # Flow vectors — show crowd movement arrows (hall mode)
    for (fx, fy, vx, vy) in flow_vectors:
        ex, ey = fx + vx*5, fy + vy*5
        cv2.arrowedLine(frame, (fx,fy), (int(ex),int(ey)),
                        (0,200,255), 1, tipLength=0.4)

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    status = meta['alert']
    if panic:    status = '⚠ PANIC / STAMPEDE'
    elif zone_alerts: status = f'⚠ ZONE BREACH: {zone_alerts[0]}'
    cv2.putText(frame, f'{status}  ({conf:.0%})', (12, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.95, 1)[0]
    cv2.putText(frame, label, (w - lbl_size[0] - 10, 20),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (160,160,160), 1, cv2.LINE_AA)

    badge_color = (0, 60, 220) if (crowd or panic) else (40, 160, 255)
    badge_text  = f'People: {count}'
    if panic:        badge_text += ' ⚠ PANIC'
    elif crowd:      badge_text += ' ⚠ CROWD'
    elif crush_zones: badge_text += ' ⚠ CRUSH'
    bw, bh = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    bx, by = w - bw - 18, h - 14
    ov2 = frame.copy()
    cv2.rectangle(ov2, (bx-8, by-bh-6), (w-6, h-4), (15,15,15), -1)
    cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, badge_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2, cv2.LINE_AA)

    if is_alert or crowd or panic:
        border = (0,0,255) if panic else ((0,60,220) if crowd and not is_alert else color)
        cv2.rectangle(frame, (2,2), (w-3,h-3), border, 5)
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

    def __init__(self, source, source_name: str = "Camera"):
        self._source_name = source_name
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
            # For HTTPS streams with self-signed certs (common on IP cameras),
            # pass FFmpeg flags to disable SSL verification and set a timeout.
            # Without these, OpenCV rejects the connection before even trying auth.
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not cap.isOpened():
                raise ValueError(f'Unable to open video source: {source}')
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
            # Log every result — logger handles buffering and intervals
            logger.record(self._source_name, result)

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
        name = _name_map.get(source_id, f'Camera-{source_id}')
        _processors[source_id] = VideoProcessor(raw_source, source_name=name)
    return _processors[source_id]


def generate_frames(source_id: int):
    try:
        processor = get_processor(source_id)
    except ValueError as e:
        print(f'[ERROR] Could not open source {source_id}: {e}')
        return

    while not processor._stopped:
        try:
            frame, result = processor.get_frame()
            if frame is None:
                continue
            display = draw_overlay(frame.copy(), result)
            _, buffer = cv2.imencode('.jpg', display)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f'[ERROR] Frame error: {e}')
            break


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



@app.route('/logs')
def view_logs():
    """
    Browser-viewable log dashboard at http://localhost:5000/logs
    Shows today's headcount log and incident log as HTML tables.
    """
    import csv, os
    from datetime import datetime
    date_str   = datetime.now().strftime('%Y-%m-%d')
    log_dir    = model.settings.get('logging-settings', {}).get('log-dir', 'logs')
    hc_path    = os.path.join(log_dir, f'headcount_{date_str}.csv')
    inc_path   = os.path.join(log_dir, f'incidents_{date_str}.csv')

    def read_csv(path):
        if not os.path.exists(path):
            return [], []
        with open(path, newline='') as f:
            rows = list(csv.reader(f))
        return (rows[0], rows[1:]) if rows else ([], [])

    hc_headers,  hc_rows  = read_csv(hc_path)
    inc_headers, inc_rows = read_csv(inc_path)

    def table(headers, rows, row_color_col=None, alert_val=None):
        if not headers:
            return '<p style="color:#888">No data yet today.</p>'
        cols  = ' '.join(f'<th>{h}</th>' for h in headers)
        html  = f'<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;font-family:monospace;font-size:13px"><tr style="background:#222;color:#eee">{cols}</tr>'
        for row in reversed(rows):   # newest first
            color = '#2a0000' if row_color_col and len(row) > row_color_col and row[row_color_col] != 'normal' else '#111'
            cells = ' '.join(f'<td style="padding:4px 10px">{c}</td>' for c in row)
            html += f'<tr style="background:{color};color:#ddd">{cells}</tr>'
        html += '</table>'
        return html

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Safety Monitor Logs — {date_str}</title>
<style>body{{background:#0d0d0d;color:#ddd;font-family:sans-serif;padding:24px}}
h1{{color:#4af}}h2{{color:#8cf;margin-top:32px}}
</style>
<meta http-equiv="refresh" content="60">
</head><body>
<h1>Safety Monitor Logs — {date_str}</h1>
<p style="color:#888">Auto-refreshes every 60s</p>
<h2>Headcount Log (every {model.settings.get('logging-settings',{{}}).get('interval-minutes',5)} mins)</h2>
{table(hc_headers, hc_rows)}
<h2>Incidents</h2>
{table(inc_headers, inc_rows, row_color_col=2)}
</body></html>"""
    return html

if __name__ == '__main__':
    app.run(debug=True)
