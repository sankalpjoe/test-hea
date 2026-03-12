"""
people_counter.py — Comprehensive people counter with behavioral intelligence.

Pipeline per frame:
  Raw YOLO boxes
      │
      ├─[1] Perspective scale filter   reject boxes too small/big for Y position
      ├─[2] Excluded zone mask         reject centroids in desk/furniture zones
      ├─[3] Static background mask     reject centroids on historically motionless pixels
      ├─[4] Pose skeleton validator    require >=N human keypoints in crop
      └─[5] Temporal consistency gate  centroid must persist >=N frames before counted

Two scenarios (settings.yaml → scenario-settings.mode):

  office  (<=30 people)
    Full pose validation on every box. Tight temporal gate (15 frames).
    Per-person ViT crop available. Detailed fall/fight analysis.

  hall    (<=300 people)
    SAHI slicing — detects tiny people at the back of a large room.
    Pose validation only on boxes < 50 total (skipped when overcrowded).
    Loose temporal gate (5 frames) for fast crowd changes.
    Crush zone + panic detection active.
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id:    int
    cx:          float
    cy:          float
    box:         tuple
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)
    frame_count: int   = 0
    confirmed:   bool  = False
    history:     deque = field(default_factory=lambda: deque(maxlen=30))


@dataclass
class CountResult:
    headcount:       int
    confirmed_boxes: list
    all_boxes:       list
    rejected_boxes:  list
    crowd_alert:     bool
    crush_zones:     list     # [(cx, cy, count)]
    panic_detected:  bool
    flow_vectors:    list     # [(cx, cy, vx, vy)]
    zone_alerts:     list     # [zone_name, ...]
    scenario:        str


# ─────────────────────────────────────────────────────────────────────────────

class PeopleCounter:

    def __init__(self, settings: dict):
        ps  = settings.get('people-counter-settings', {})
        sc  = settings.get('scenario-settings', {})
        sahi_cfg = sc.get('sahi', {})

        self.enabled       = ps.get('enabled', True)
        self.conf          = ps.get('confidence', 0.25)
        self.imgsz         = ps.get('imgsz', 1280)
        self.iou_thresh    = ps.get('iou', 0.35)
        self.crowd_thresh  = ps.get('crowd-alert-threshold', 30)
        self.device        = ps.get('device', 'cpu')

        self.scenario         = sc.get('mode', 'office')
        self.min_keypoints    = sc.get('min-keypoints', 4)
        self.temporal_frames  = sc.get('temporal-frames',
                                       5 if self.scenario == 'hall' else 15)
        self.static_frames_n  = sc.get('static-learn-frames', 150)
        self.crush_density    = sc.get('crush-density-threshold', 8)
        self.panic_velocity   = sc.get('panic-velocity-threshold', 25)
        self.perspective      = sc.get('perspective-scale', {})

        self.use_sahi      = sahi_cfg.get('enabled', self.scenario == 'hall')
        self.sahi_h        = sahi_cfg.get('slice-height', 640)
        self.sahi_w        = sahi_cfg.get('slice-width',  640)
        self.sahi_overlap  = sahi_cfg.get('overlap-ratio', 0.2)

        self.restricted_zones = settings.get('zones', {}).get('restricted', [])
        self.excluded_zones   = settings.get('zones', {}).get('excluded',   [])

        if not self.enabled:
            return

        try:
            from ultralytics import YOLO
            self.yolo_det  = YOLO(ps.get('model',       'yolov8m.pt'))
            self.yolo_pose = YOLO(ps.get('pose-model',  'yolov8n-pose.pt'))
            print(f'[INFO] Counter ready  scenario={self.scenario}  '
                  f'temporal={self.temporal_frames}f  sahi={self.use_sahi}')
        except ImportError:
            print('[WARN] pip install ultralytics')
            self.enabled = False
            return

        self._tracks:        dict  = {}
        self._next_id:       int   = 0
        self._frame_idx:     int   = 0
        self._static_mask:   Optional[np.ndarray] = None
        self._static_accum:  Optional[np.ndarray] = None
        self._static_ready:  bool  = False
        self._static_n:      int   = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def count(self, frame_bgr):
        """Backward-compatible: returns (headcount, confirmed_boxes)."""
        if not self.enabled:
            return 0, []
        r = self.count_full(frame_bgr)
        return r.headcount, r.confirmed_boxes

    def count_full(self, frame_bgr) -> CountResult:
        if not self.enabled:
            return CountResult(0,[],[],[],False,[],False,[],[],'disabled')

        self._frame_idx += 1
        h, w = frame_bgr.shape[:2]
        self._update_static(frame_bgr)

        # 1. Detect
        raw = self._detect_sahi(frame_bgr) if self.use_sahi else self._detect(frame_bgr)

        # 2. Filter
        good, rejected = [], []
        for box in raw:
            x1,y1,x2,y2 = box
            cx,cy = (x1+x2)/2, (y1+y2)/2
            bh = y2-y1
            if not self._perspective_ok(cx,cy,bh,w,h): rejected.append(box); continue
            if self._in_excluded(cx,cy,w,h):            rejected.append(box); continue
            if self._is_static(cx,cy,h,w):              rejected.append(box); continue
            good.append(box)

        # 3. Pose validation
        if self.scenario == 'office' or len(good) < 50:
            validated, rej2 = [], []
            for b in good:
                (validated if self._has_pose(frame_bgr, b) else rej2).append(b)
            rejected += rej2
        else:
            validated = good   # too many to validate individually in hall mode

        # 4. Temporal tracking
        self._update_tracks(validated)
        confirmed = [t.box for t in self._tracks.values() if t.confirmed]

        # 5. Zone alerts
        zone_alerts = self._zone_breach(confirmed, w, h)

        # 6. Crowd analytics
        crush, panic, flow = [], False, []
        if self.scenario == 'hall' and len(confirmed) > 10:
            crush        = self._crush_zones(confirmed, w, h)
            panic, flow  = self._panic(confirmed)

        headcount   = len(confirmed)
        crowd_alert = (self.crowd_thresh > 0 and headcount > self.crowd_thresh)

        return CountResult(
            headcount       = headcount,
            confirmed_boxes = confirmed,
            all_boxes       = raw,
            rejected_boxes  = rejected,
            crowd_alert     = crowd_alert or bool(crush),
            crush_zones     = crush,
            panic_detected  = panic,
            flow_vectors    = flow,
            zone_alerts     = zone_alerts,
            scenario        = self.scenario,
        )

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame):
        res = self.yolo_det(frame, classes=[0], conf=self.conf,
                            imgsz=self.imgsz, iou=self.iou_thresh,
                            augment=(self.scenario=='office'),
                            verbose=False, device=self.device)
        return [tuple(map(int, b.xyxy[0].tolist())) for r in res for b in r.boxes]

    def _detect_sahi(self, frame):
        """
        SAHI — Slicing Aided Hyper Inference.
        Cuts frame into overlapping tiles, detects on each, merges with NMS.
        Makes tiny distant people in a hall detectable.
        """
        h, w   = frame.shape[:2]
        step_y = int(self.sahi_h * (1-self.sahi_overlap))
        step_x = int(self.sahi_w * (1-self.sahi_overlap))
        all_b, all_s = [], []

        ys = list(range(0, max(1,h-self.sahi_h+1), step_y))
        if not ys or ys[-1]+self.sahi_h < h: ys.append(max(0,h-self.sahi_h))
        xs = list(range(0, max(1,w-self.sahi_w+1), step_x))
        if not xs or xs[-1]+self.sahi_w < w: xs.append(max(0,w-self.sahi_w))

        for y0 in set(ys):
            for x0 in set(xs):
                y1e,x1e = min(y0+self.sahi_h,h), min(x0+self.sahi_w,w)
                tile = frame[y0:y1e, x0:x1e]
                res  = self.yolo_det(tile, classes=[0], conf=self.conf,
                                     imgsz=640, iou=self.iou_thresh,
                                     verbose=False, device=self.device)
                for r in res:
                    for b in r.boxes:
                        bx1,by1,bx2,by2 = b.xyxy[0].tolist()
                        all_b.append([int(bx1+x0),int(by1+y0),
                                      int(bx2+x0),int(by2+y0)])
                        all_s.append(float(b.conf[0]))

        if not all_b:
            return []
        idx = cv2.dnn.NMSBoxes(all_b, all_s, self.conf, 0.4)
        if len(idx)==0: return []
        return [tuple(all_b[i]) for i in idx.flatten()]

    # ── Pose validation ───────────────────────────────────────────────────────

    def _has_pose(self, frame, box) -> bool:
        """
        Crop box, run YOLOv8-pose, require >=min_keypoints with conf>0.3.
        Desks/chairs never produce a human skeleton — this eliminates them.
        For seated office workers: 4 keypoints (nose + shoulders + elbow)
        is enough to confirm a human even with legs hidden behind desk.
        """
        x1,y1,x2,y2 = box
        h,w = frame.shape[:2]
        pad = 10
        crop = frame[max(0,y1-pad):min(h,y2+pad),
                     max(0,x1-pad):min(w,x2+pad)]
        if crop.size == 0:
            return False
        try:
            res = self.yolo_pose(crop, conf=0.25, verbose=False, device=self.device)
            for r in res:
                if r.keypoints is None or r.keypoints.conf is None:
                    continue
                for kpts in r.keypoints.conf:
                    if int((kpts > 0.3).sum().item()) >= self.min_keypoints:
                        return True
        except Exception:
            pass
        return False

    # ── Temporal tracker ──────────────────────────────────────────────────────

    def _update_tracks(self, boxes):
        now = time.time()
        matched = set()
        for box in boxes:
            x1,y1,x2,y2 = box
            cx,cy = (x1+x2)/2,(y1+y2)/2
            best_id, best_d = None, float('inf')
            for tid,t in self._tracks.items():
                d = ((t.cx-cx)**2+(t.cy-cy)**2)**0.5
                if d < best_d and d < 80:
                    best_d,best_id = d,tid
            if best_id is not None:
                t = self._tracks[best_id]
                t.cx,t.cy,t.box,t.last_seen = cx,cy,box,now
                t.frame_count += 1
                t.history.append((cx,cy,now))
                if t.frame_count >= self.temporal_frames:
                    t.confirmed = True
                matched.add(best_id)
            else:
                tid = self._next_id; self._next_id += 1
                tr  = Track(track_id=tid,cx=cx,cy=cy,box=box,frame_count=1)
                tr.history.append((cx,cy,now))
                self._tracks[tid] = tr
                matched.add(tid)
        # Expire stale tracks
        for tid in [k for k,t in self._tracks.items() if now-t.last_seen>1.0]:
            del self._tracks[tid]

    # ── Static background ─────────────────────────────────────────────────────

    def _update_static(self, frame):
        if self._static_ready: return
        gray = cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(21,21),0)
        if self._static_accum is None:
            self._static_accum = gray.astype(np.float32)
        else:
            diff = cv2.absdiff(gray, self._static_accum.astype(np.uint8))
            self._static_accum = (self._static_accum*0.95 +
                                   gray.astype(np.float32)*0.05)
            self._static_mask = (diff.astype(np.float32) if self._static_mask is None
                                  else np.maximum(self._static_mask, diff.astype(np.float32)))
        self._static_n += 1
        if self._static_n >= self.static_frames_n:
            self._static_ready = True
            print('[INFO] Static background mask ready.')

    def _is_static(self, cx, cy, h, w) -> bool:
        if not self._static_ready or self._static_mask is None: return False
        ix = int(np.clip(cx,0,w-1)); iy = int(np.clip(cy,0,h-1))
        return float(self._static_mask[iy,ix]) < 5.0

    # ── Perspective filter ────────────────────────────────────────────────────

    def _perspective_ok(self, cx, cy, bh, w, h) -> bool:
        if not self.perspective: return True
        yf = cy / max(h,1)
        for band in self.perspective.get('bands',[]):
            if band.get('y-min',0) <= yf <= band.get('y-max',1):
                return band.get('min-height-px',0) <= bh <= band.get('max-height-px',9999)
        return True

    # ── Zone helpers ──────────────────────────────────────────────────────────

    def _in_excluded(self, cx, cy, w, h) -> bool:
        for z in self.excluded_zones:
            if z['x1']*w <= cx <= z['x2']*w and z['y1']*h <= cy <= z['y2']*h:
                return True
        return False

    def _zone_breach(self, boxes, w, h) -> list:
        alerts = []
        for zone in self.restricted_zones:
            for x1,y1,x2,y2 in boxes:
                cx,cy = (x1+x2)/2,(y1+y2)/2
                if (zone['x1']*w <= cx <= zone['x2']*w and
                    zone['y1']*h <= cy <= zone['y2']*h):
                    alerts.append(zone.get('name','Restricted Zone')); break
        return alerts

    # ── Crowd analytics ───────────────────────────────────────────────────────

    def _crush_zones(self, boxes, w, h):
        gr,gc = 4,6
        grid  = np.zeros((gr,gc),dtype=np.int32)
        for x1,y1,x2,y2 in boxes:
            cx,cy = (x1+x2)/2,(y1+y2)/2
            r = int(np.clip(cy/(h/gr),0,gr-1))
            c = int(np.clip(cx/(w/gc),0,gc-1))
            grid[r,c] += 1
        pts = []
        for r in range(gr):
            for c in range(gc):
                if grid[r,c] >= self.crush_density:
                    pts.append((int((c+0.5)*(w/gc)), int((r+0.5)*(h/gr)),
                                int(grid[r,c])))
        return pts

    def _panic(self, boxes):
        now = time.time()
        flow,vels = [],[]
        for t in self._tracks.values():
            if not t.confirmed or len(t.history)<5: continue
            recent = [(cx,cy,ts) for cx,cy,ts in t.history if now-ts<0.5]
            if len(recent)<2: continue
            dx = recent[-1][0]-recent[0][0]; dy = recent[-1][1]-recent[0][1]
            dt = max(recent[-1][2]-recent[0][2],0.01)
            vx,vy = dx/dt, dy/dt
            flow.append((int(t.cx),int(t.cy),int(vx*0.1),int(vy*0.1)))
            vels.append((vx**2+vy**2)**0.5)
        if not vels: return False,flow
        return float(np.mean(vels)) > self.panic_velocity, flow
