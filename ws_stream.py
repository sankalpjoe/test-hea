"""
ws_stream.py — WebSocket video capture, drop-in replacement for cv2.VideoCapture.

Why cv2.VideoCapture can't open WebSocket streams:
  OpenCV's VideoCapture uses FFmpeg as its backend. FFmpeg supports HTTP, RTSP,
  RTMP, and HLS — but NOT the WebSocket protocol (ws:// or wss://). Attempting
  to pass a ws:// URL to VideoCapture raises "Unable to open video source".

How this works instead:
  1. websocket-client connects to the camera's ws:// or wss:// endpoint.
  2. The camera sends frames as binary messages — either:
       a) Raw JPEG bytes        (most common: IP cameras, NVRs)
       b) H.264 NAL units       (needs PyAV or imageio to decode — see below)
       c) Base64-encoded JPEG   (some cheap cameras)
  3. Each message is decoded into a NumPy BGR frame (same as cv2.read()).
  4. self.frame is updated in-place so the rest of app.py is unchanged.

Usage — replace VideoProcessor's cap with this:
  vs = WebSocketStream("wss://your-ip/stream").start()
  frame = vs.read()   # returns latest BGR frame, never blocks
  vs.stop()
"""

import cv2
import numpy as np
import base64
import ssl
from threading import Thread

try:
    import websocket          # pip install websocket-client
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print('[WARN] websocket-client not installed. Run: pip install websocket-client')


class WebSocketStream:
    """
    Connects to a WebSocket camera stream and exposes the same
    .read() / .stop() / .stopped interface as ThreadedVideoStream,
    so app.py needs zero changes beyond swapping the class.
    """

    def __init__(self, url: str, token: str = None):
        """
        url   : ws:// or wss:// stream URL found in browser DevTools (WS tab)
        token : optional auth token if camera requires a header
                e.g.  "Bearer eyJhbGci..."
        """
        if not WS_AVAILABLE:
            raise ImportError('Install websocket-client:  pip install websocket-client')

        self.url     = url
        self.token   = token
        self.frame   = None
        self.stopped = False
        self._ws     = None

    def start(self):
        t = Thread(target=self._run, daemon=True)
        t.start()
        return self

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self._ws:
            self._ws.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self):
        headers = {}
        if self.token:
            headers['Authorization'] = self.token

        # Disable SSL certificate check for self-signed camera certs
        # (most IP cameras use self-signed — without this you get SSL errors)
        ssl_opt = {'cert_reqs': ssl.CERT_NONE}

        self._ws = websocket.WebSocketApp(
            self.url,
            header=headers,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws.run_forever(sslopt=ssl_opt, ping_interval=20, ping_timeout=10)

    def _on_message(self, ws, message):
        """
        Decode incoming WebSocket message into a BGR NumPy frame.
        Tries three formats in order: raw JPEG bytes, Base64 JPEG, H.264 NAL.
        """
        frame = None

        if isinstance(message, bytes):
            # ── Format A: raw JPEG bytes (most common) ──────────────────────
            frame = self._decode_jpeg(message)

            # ── Format B: H.264 NAL unit (if JPEG decode fails) ─────────────
            if frame is None:
                frame = self._decode_h264(message)

        elif isinstance(message, str):
            # ── Format C: Base64-encoded JPEG string ─────────────────────────
            try:
                raw = base64.b64decode(message)
                frame = self._decode_jpeg(raw)
            except Exception:
                pass

        if frame is not None:
            self.frame = frame

    @staticmethod
    def _decode_jpeg(data: bytes):
        """Decode raw JPEG bytes → BGR NumPy array."""
        try:
            arr   = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame  # None if data wasn't valid JPEG
        except Exception:
            return None

    @staticmethod
    def _decode_h264(data: bytes):
        """
        Decode H.264 NAL unit bytes → BGR NumPy array using PyAV.
        Only used if the camera sends H.264 over WebSocket (less common).
        Install:  pip install av
        """
        try:
            import av
            codec = av.CodecContext.create('h264', 'r')
            packets = codec.parse(data)
            for packet in packets:
                frames = codec.decode(packet)
                for f in frames:
                    bgr = f.to_ndarray(format='bgr24')
                    return bgr
        except Exception:
            return None

    def _on_error(self, ws, error):
        print(f'[WS ERROR] {error}')

    def _on_close(self, ws, code, msg):
        print(f'[WS] Connection closed — code={code}')
        self.stopped = True
