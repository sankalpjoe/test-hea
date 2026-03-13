"""
logger.py — Periodic headcount + incident logger.

Two separate logs are written:

  1. logs/headcount_YYYY-MM-DD.csv
     Written every N minutes (default 5).
     Columns: timestamp, camera, headcount, avg_headcount_since_last_log

     Example:
       2024-03-12 09:00:00, CCTV-Entrance, 23, 21.4
       2024-03-12 09:05:00, CCTV-Entrance, 25, 24.1

  2. logs/incidents_YYYY-MM-DD.csv
     Written immediately whenever an alert fires (violence, weapon, fire etc).
     Columns: timestamp, camera, category, label, confidence, headcount

     Example:
       2024-03-12 09:03:22, CCTV-Entrance, violence, fight on a street, 0.31, 4

Both files roll over at midnight — new file per day, so logs stay manageable.
A plain-text daily summary is also printed to terminal at each interval.
"""

import csv
import os
import threading
import time
from collections import defaultdict
from datetime import datetime


LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)


class SafetyLogger:
    """
    Thread-safe logger. Call:
      logger.record(source_name, result)   — on every inference result
      logger.start()                        — starts the background flush thread
    """

    def __init__(self, interval_minutes: int = 5):
        self.interval   = interval_minutes * 60   # seconds
        self._lock      = threading.Lock()
        self._stopped   = False

        # Per-camera rolling buffer of headcounts since last log flush
        # { camera_name: [count1, count2, ...] }
        self._headcount_buffer: dict[str, list[int]] = defaultdict(list)

        # Track last alert per category to avoid spamming the incident log
        # { (camera, category): last_logged_timestamp }
        self._last_incident: dict[tuple, float] = {}
        self._incident_cooldown = 30   # seconds — don't re-log same category within this

    def start(self):
        t = threading.Thread(target=self._flush_loop, daemon=True)
        t.start()
        return self

    def stop(self):
        self._stopped = True

    # ── Called on every inference result ─────────────────────────────────────

    def record(self, source_name: str, result: dict):
        """
        Called from app.py on every new inference result.
        Buffers headcount for periodic logging.
        Immediately logs incidents (alerts).
        """
        headcount = result.get('headcount', 0)
        category  = result.get('category', 'Unknown')
        is_alert  = result.get('is_alert', False)
        crowd     = result.get('crowd_alert', False)

        # Buffer headcount for the periodic log
        with self._lock:
            self._headcount_buffer[source_name].append(headcount)

        # Incident log — fire immediately, with cooldown per category
        if is_alert or crowd:
            alert_cat = 'crowd' if (crowd and not is_alert) else category
            key       = (source_name, alert_cat)
            now       = time.time()
            last      = self._last_incident.get(key, 0)

            if now - last >= self._incident_cooldown:
                self._last_incident[key] = now
                self._write_incident(source_name, result, crowd)

    # ── Periodic flush (runs every N minutes in background thread) ────────────

    def _flush_loop(self):
        # Align first flush to the next clean interval boundary
        # e.g. if interval=5min and it's 09:03, first flush at 09:05
        now     = time.time()
        wait    = self.interval - (now % self.interval)
        time.sleep(wait)

        while not self._stopped:
            self._flush_headcounts()
            time.sleep(self.interval)

    def _flush_headcounts(self):
        now_dt = datetime.now()
        date_str = now_dt.strftime('%Y-%m-%d')
        path     = os.path.join(LOG_DIR, f'headcount_{date_str}.csv')
        is_new   = not os.path.exists(path)

        with self._lock:
            buffer_snapshot = dict(self._headcount_buffer)
            self._headcount_buffer.clear()

        if not buffer_snapshot:
            return

        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['timestamp', 'camera', 'headcount', 'avg_since_last_log'])

            ts = now_dt.strftime('%Y-%m-%d %H:%M:%S')
            print(f'\n── Headcount Log [{ts}] {"─" * 30}')

            for camera, counts in buffer_snapshot.items():
                latest_count = counts[-1] if counts else 0
                avg_count    = round(sum(counts) / len(counts), 1) if counts else 0
                writer.writerow([ts, camera, latest_count, avg_count])

                print(f'  {camera:<25} people now: {latest_count:>3}   '
                      f'avg over last {len(counts)} readings: {avg_count}')

        print(f'  → Saved to {path}\n')

    # ── Incident writer ───────────────────────────────────────────────────────

    def _write_incident(self, source_name: str, result: dict, crowd: bool):
        now_dt   = datetime.now()
        date_str = now_dt.strftime('%Y-%m-%d')
        path     = os.path.join(LOG_DIR, f'incidents_{date_str}.csv')
        is_new   = not os.path.exists(path)

        category   = 'crowd'            if crowd and not result.get('is_alert') else result.get('category', '')
        label      = 'crowd alert'      if category == 'crowd' else result.get('label', '')
        confidence = result.get('confidence', 0.0)
        headcount  = result.get('headcount', 0)
        ts         = now_dt.strftime('%Y-%m-%d %H:%M:%S')

        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(['timestamp', 'camera', 'category', 'label', 'confidence', 'headcount'])
            writer.writerow([ts, source_name, category, label, f'{confidence:.3f}', headcount])

        print(f'  [INCIDENT] {ts} | {source_name} | {category.upper()} | '
              f'"{label}" | conf={confidence:.2f} | people={headcount}')
        print(f'  → Saved to {path}')
