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


from people_counter import PeopleCounter, CountResult


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
        # PeopleCounter now receives the full settings dict so it can
        # read scenario, SAHI, zones, pose, and temporal config itself.
        self.counter = PeopleCounter(self.settings) if counter_enabled else None

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

        # ── 2. YOLOv8 headcount + behavioral analytics ──────────────────────
        cr = None
        if self.counter:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cr  = self.counter.count_full(bgr)

        return {
            'label':           label_text,
            'category':        category,
            'confidence':      confidence,
            'is_alert':        category not in ('normal', 'Unknown'),
            # ── People counter fields ────────────────────────────────────────
            'headcount':       cr.headcount       if cr else 0,
            'person_boxes':    cr.confirmed_boxes if cr else [],
            'rejected_boxes':  cr.rejected_boxes  if cr else [],
            'crowd_alert':     cr.crowd_alert      if cr else False,
            'crush_zones':     cr.crush_zones      if cr else [],
            'panic_detected':  cr.panic_detected   if cr else False,
            'flow_vectors':    cr.flow_vectors      if cr else [],
            'zone_alerts':     cr.zone_alerts       if cr else [],
            'scenario':        cr.scenario          if cr else 'office',
        }

    @staticmethod
    def plot_image(image: np.ndarray, title_text: str):
        plt.figure(figsize=[13, 13])
        plt.title(title_text)
        plt.axis('off')
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
