import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

# ── Category display metadata ─────────────────────────────────────────────────
CATEGORY_META = {
    'violence':     {'color': (30,   30, 220), 'alert': 'VIOLENCE DETECTED'},
    'fall':         {'color': (30,  140, 255), 'alert': 'FALL DETECTED'},
    'injury':       {'color': (40,   80, 220), 'alert': 'INJURY DETECTED'},
    'weapon-gun':   {'color': (0,    0,  255), 'alert': '⚠ GUN DETECTED'},
    'weapon-knife': {'color': (0,   60,  255), 'alert': '⚠ KNIFE DETECTED'},
    'fire-smoke':   {'color': (0,  200,  255), 'alert': '⚠ FIRE / SMOKE'},
    'vandalism':    {'color': (0,  165,  255), 'alert': 'VANDALISM DETECTED'},
    'suspicious':   {'color': (0,  200,  180), 'alert': 'SUSPICIOUS BEHAVIOUR'},
    'normal':       {'color': (50, 200,   80), 'alert': 'No Threat Detected'},
    'Unknown':      {'color': (120, 120, 120), 'alert': 'Low Confidence'},
}

# ── Priority order for multi-alert tiebreaking ────────────────────────────────
# If two categories score above threshold in the same frame, higher priority wins.
CATEGORY_PRIORITY = [
    'weapon-gun', 'weapon-knife', 'fire-smoke',
    'violence', 'injury', 'fall',
    'vandalism', 'suspicious',
    'normal', 'Unknown',
]


class PeopleCounter:
    """
    Lightweight headcount using YOLOv8 nano (class 0 = person only).

    Why YOLOv8n for counting instead of CLIP:
      CLIP is a scene classifier — it outputs one label for the whole image.
      It cannot count individuals. YOLOv8n is an object detector that draws
      bounding boxes around each person separately, giving an exact count.
      The 'nano' variant is ~6MB and runs ~30ms/frame on CPU — fast enough
      to run alongside CLIP in the same inference thread without adding
      noticeable latency.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.45,
                 device: str = 'cpu'):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(model_path)
            self.conf = confidence
            self.device = device
            self.enabled = True
            print(f'[INFO] YOLOv8 people counter ready (conf={confidence})')
        except ImportError:
            print('[WARN] ultralytics not installed — people counter disabled.')
            print('       Run: pip install ultralytics')
            self.enabled = False

    def count(self, frame_bgr: np.ndarray) -> tuple[int, list[tuple]]:
        """
        Returns (headcount, bboxes).
        bboxes: list of (x1, y1, x2, y2) for each detected person.
        """
        if not self.enabled:
            return 0, []

        results = self.yolo(
            frame_bgr,
            classes=[0],            # 0 = person — skip all other classes
            conf=self.conf,
            verbose=False,
            device=self.device,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
        return len(boxes), boxes


class Model:
    def __init__(self, settings_path: str = './settings.yaml'):
        with open(settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)

        ms = self.settings['model-settings']
        ls = self.settings['label-settings']
        ps = self.settings.get('people-counter-settings', {})

        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = ms['prediction-threshold']

        # ── CLIP setup ────────────────────────────────────────────────────────
        self.model, self.preprocess = clip.load(ms['model-name'], device=self.device)
        self.labels        = ls['labels']
        self.default_label = ls.get('default-label', 'Unknown')

        # 'a photo of X' prefix — shown by CLIP paper to improve accuracy
        self.labels_ = ['a photo of ' + lbl for lbl in self.labels]
        self.text_features = self.vectorize_text(self.labels_)

        # Build label → category lookup
        self._label_to_cat: dict[str, str] = {}
        for cat, cat_labels in ls.get('category-map', {}).items():
            for lbl in cat_labels:
                self._label_to_cat[lbl] = cat

        # EMA smoothing per label — prevents single-frame noise from alerting.
        # alpha=0.4: reacts within ~3 frames. Raise for faster reaction.
        self._ema_alpha  = 0.4
        self._ema_scores = np.zeros(len(self.labels), dtype=np.float32)

        # ── People counter setup ──────────────────────────────────────────────
        counter_enabled    = ps.get('enabled', True)
        self.crowd_thresh  = ps.get('crowd-alert-threshold', 0)
        if counter_enabled:
            self.counter = PeopleCounter(
                model_path=ps.get('model', 'yolov8n.pt'),
                confidence=ps.get('confidence', 0.45),
                device=str(self.device),
            )
        else:
            self.counter = None

        print(f'[INFO] CLIP ready — {len(self.labels)} labels across '
              f'{len(set(self._label_to_cat.values()))} categories')

    # ── Original helper methods (unchanged) ───────────────────────────────────

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        # Pre-resize to 224px: avoids passing large frame through PIL preprocessing
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        pil_image = Image.fromarray(image).convert('RGB')
        return self.preprocess(pil_image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def tokenize(self, text: list):
        return clip.tokenize(text).to(self.device)

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = self.tokenize(text=text)
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def predict_(self, text_features, image_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        return similarity[0].topk(1)

    # ── Main predict — CLIP classification + YOLOv8 headcount ────────────────

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        """
        Runs CLIP scene classification + YOLOv8 people counting on the same frame.

        Returns:
          label       (str)   top CLIP label (or default-label if below threshold)
          category    (str)   mapped category name
          confidence  (float) smoothed cosine similarity of top label
          is_alert    (bool)  True if category is not 'normal' or 'Unknown'
          headcount   (int)   number of people detected by YOLOv8
          person_boxes (list) [(x1,y1,x2,y2), ...] bounding boxes
          crowd_alert (bool)  True if headcount > crowd-alert-threshold
        """
        # ── 1. CLIP classification ────────────────────────────────────────────
        tf_image       = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)

        img_norm   = image_features / image_features.norm(dim=-1, keepdim=True)
        txt_norm   = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        raw_scores = (img_norm @ txt_norm.T)[0].cpu().numpy()

        # EMA smoothing
        self._ema_scores = (self._ema_alpha * raw_scores
                            + (1 - self._ema_alpha) * self._ema_scores)

        top_idx   = int(np.argmax(self._ema_scores))
        top_score = float(self._ema_scores[top_idx])
        confidence = abs(top_score)

        if confidence >= self.threshold:
            label_text = self.labels[top_idx]
            category   = self._label_to_cat.get(label_text, 'normal')
        else:
            label_text = self.default_label
            category   = 'Unknown'

        # ── 2. YOLOv8 headcount ───────────────────────────────────────────────
        headcount, person_boxes = (0, [])
        if self.counter:
            # YOLOv8 needs BGR — image is passed in as RGB from app.py
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            headcount, person_boxes = self.counter.count(bgr)

        crowd_alert = (self.crowd_thresh > 0 and headcount > self.crowd_thresh)

        return {
            'label':        label_text,
            'category':     category,
            'confidence':   confidence,
            'is_alert':     category not in ('normal', 'Unknown'),
            'headcount':    headcount,
            'person_boxes': person_boxes,
            'crowd_alert':  crowd_alert,
        }

    @staticmethod
    def plot_image(image: np.ndarray, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)