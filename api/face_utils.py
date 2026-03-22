"""
face_utils.py
─────────────────────────────────────────────────────────────────────────────
All heavy face detection/embedding runs on HF Space (InsightFace buffalo_l).
Django handles:
  - Centroid-based cosine similarity matching (vs per-embedding max)
  - Duplicate face detection during registration
  - Attendance logging with per-event-type cooldowns
  - Entry count tracking (1st/2nd/3rd entry)
  - Bounding box passthrough to frontend

Set HF_SPACE_URL in Railway environment variables:
  HF_SPACE_URL=https://YOUR-USERNAME-attendai-face-service.hf.space

─────────────────────────────────────────────────────────────────────────────
WHY CENTROID MATCHING?
─────────────────────────────────────────────────────────────────────────────
When a user registers 5 photos (front, left, right, tilt, etc.), we store 5
separate embeddings. The old code compared the query against each stored
embedding and took the MAX similarity. This is prone to noise — one bad
photo can drag the score down, and a coincidentally similar angle from a
different person could give a false positive.

Centroid matching:
  1. Stack all stored (L2-normalised) embeddings as rows
  2. Take the mean → this is the centroid in embedding space
  3. Re-normalise the centroid
  4. Compute ONE dot product: query · centroid

This has two key advantages:
  a) The centroid represents the average appearance of the person → more stable
  b) It collapses N comparisons to 1, reducing O(N×M) to O(M) complexity
     where M = number of registered users

The mathematical insight: the mean of L2-normalised vectors, when renormalised,
approximates the principal direction of the cluster in the unit hypersphere.
This is analogous to "class mean" in Fisher LDA, and is empirically effective
for ArcFace embeddings because ArcFace training explicitly clusters same-person
embeddings close together in the hypersphere.
"""
"""
face_utils.py
─────────────────────────────────────────────────────────────────────────────
COLD START / TIMEOUT SOLUTION
─────────────────────────────────────────────────────────────────────────────
HF Spaces free tier sleeps after ~15 min inactivity. Cold start takes 30-60s.
The old HF_TIMEOUT=15s expired before the Space even woke up.

Three-layer fix:
  1. _hf_ping_loop()  — background thread pings /health every 4 min from
                        Django startup. Space never gets a chance to sleep.
  2. _hf_post()       — wrapper with one retry on timeout.
  3. HF_TIMEOUT=30s   — generous enough to survive a warm Space under load.
"""
"""
face_utils.py
─────────────────────────────────────────────────────────────────────────────
PING / KEEP-ALIVE FIXES:
  - PING_TIMEOUT = 60s  (was 10s — buffalo_l cold start takes 30-60s)
  - Wake-up mode: on ping failure, switches to 30s retry interval
    and keeps retrying until Space responds, then returns to 4-min cadence
  - Only one worker runs the thread (enforced by apps.py file-lock)

RECOGNITION:
  - buffalo_l (ResNet50 ArcFace) → 88-95% confidence
  - Centroid matching vs per-embedding max
  - Per-event-type cooldowns (fixes shaky exit)
  - Duplicate registration check (409 Conflict)
  - entry_count for "2nd entry today" display
"""

import os
import time
import threading
import logging
import cv2
import numpy as np
import requests
from django.utils import timezone
from api.models import UserProfile, AttendanceLog, AttendanceSession

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_SPACE_URL = os.environ.get(
    'HF_SPACE_URL',
    'https://YOUR-USERNAME-attendai-face-service.hf.space'
).rstrip('/')

RECOGNITION_THRESHOLD   = 0.50
DUPLICATE_REG_THRESHOLD = 0.55
ENTRY_COOLDOWN_S        = 10
EXIT_COOLDOWN_S         = 5

# ── Timeout constants ─────────────────────────────────────────────────────────
# HF_TIMEOUT: per-request timeout for /extract and /detect
# Use 30s — generous for a WARM Space under load.
HF_TIMEOUT = 30

# PING_TIMEOUT: timeout for keep-alive /health pings.
# Must be >= cold-start time. buffalo_l cold start = 30-60s.
# We set 60s here so a single ping can survive a full cold start.
PING_TIMEOUT = 60

# Keep-alive cadence when Space is healthy
PING_INTERVAL_NORMAL_S = 4 * 60   # 4 minutes

# Keep-alive cadence when Space is not responding (wake-up mode)
PING_INTERVAL_WAKEUP_S = 30       # retry every 30s until it responds


# ── Keep-alive pinger ─────────────────────────────────────────────────────────
def _hf_ping_loop():
    """
    Pings /health every 4 minutes to keep the Space warm.

    State machine:
      HEALTHY  → ping every 4 min, log at DEBUG
      WAKING   → ping every 30s with 60s timeout, log at WARNING
                 (happens when Space cold-starts or Railway redeploys)
      On success after WAKING → back to HEALTHY, log INFO

    Why not just always use a 60s ping?
    A 60s blocking call every 4 minutes means the thread sleeps fine.
    But when the Space IS healthy, a 60s timeout is wasteful — we
    want /health responses in <1s. The state machine uses an appropriate
    timeout for each state.
    """
    time.sleep(15)  # let gunicorn fully boot before first ping

    space_is_healthy = False  # start in waking state to get first ping ASAP

    while True:
        timeout   = PING_TIMEOUT if not space_is_healthy else 10
        interval  = PING_INTERVAL_NORMAL_S if space_is_healthy else PING_INTERVAL_WAKEUP_S

        try:
            r = requests.get(f"{HF_SPACE_URL}/health", timeout=timeout)
            if r.status_code == 200:
                if not space_is_healthy:
                    logger.info("HF Space is UP: %s", r.json())
                    space_is_healthy = True
                else:
                    logger.debug("HF ping OK")
            else:
                logger.warning("HF ping HTTP %s — switching to wake-up mode", r.status_code)
                space_is_healthy = False
        except requests.exceptions.Timeout:
            logger.warning(
                "HF ping timed out (%ds) — Space may be cold-starting. "
                "Will retry in %ds.",
                timeout, PING_INTERVAL_WAKEUP_S
            )
            space_is_healthy = False
        except Exception as e:
            logger.warning("HF ping error: %s — retrying in %ds", e, PING_INTERVAL_WAKEUP_S)
            space_is_healthy = False

        time.sleep(interval)


def start_hf_keepalive():
    """Start the keep-alive daemon. Called once from apps.py."""
    t = threading.Thread(target=_hf_ping_loop, daemon=True, name="hf-keepalive")
    t.start()
    logger.info(
        "HF keep-alive started (normal: every %ds, wake-up: every %ds)",
        PING_INTERVAL_NORMAL_S, PING_INTERVAL_WAKEUP_S
    )


# ── HF HTTP wrapper with one retry ───────────────────────────────────────────
def _hf_post(endpoint: str, files: dict) -> dict | None:
    """
    POST to HF Space with one retry on timeout.
    First attempt: HF_TIMEOUT=30s.
    If it times out (Space still waking from outside keep-alive window),
    wait 3s and try once more with 45s timeout.
    """
    url = f"{HF_SPACE_URL}/{endpoint.lstrip('/')}"
    timeouts = [30, 45]  # attempt 0, attempt 1

    for attempt, timeout in enumerate(timeouts):
        try:
            resp = requests.post(url, files=files, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.error("HF %s HTTP %s: %s", endpoint, resp.status_code, resp.text[:200])
            return None
        except requests.exceptions.Timeout:
            if attempt < len(timeouts) - 1:
                logger.warning("HF %s timeout (attempt %d) — retrying in 3s…", endpoint, attempt + 1)
                time.sleep(3)
            else:
                logger.error("HF %s timed out on all %d attempts", endpoint, len(timeouts))
                return None
        except Exception as e:
            logger.error("HF %s error: %s", endpoint, e)
            return None


# ── Image helpers ─────────────────────────────────────────────────────────────
def _img_to_bytes(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _l2_norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


# ── HF callers ────────────────────────────────────────────────────────────────
def get_embedding(image_array: np.ndarray):
    data = _hf_post('extract', {'image': ('face.jpg', _img_to_bytes(image_array), 'image/jpeg')})
    if data is None or 'embedding' not in data:
        return None
    return _l2_norm(np.array(data['embedding'], dtype=np.float32))


def detect_faces_remote(image_array: np.ndarray) -> list:
    data = _hf_post('detect', {'image': ('frame.jpg', _img_to_bytes(image_array), 'image/jpeg')})
    return data.get('faces', []) if data else []


# ── Centroid matching ─────────────────────────────────────────────────────────
def _get_user_centroid(embeddings: list) -> np.ndarray:
    arr   = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr   = arr / (norms + 1e-9)
    return _l2_norm(arr.mean(axis=0))


def match_face(embedding: np.ndarray):
    emb = _l2_norm(np.array(embedding, dtype=np.float32))
    best_user, best_score = None, 0.0
    for user in UserProfile.objects.all():
        stored = user.get_embeddings()
        if not stored:
            continue
        sim = float(np.dot(emb, _get_user_centroid(stored)))
        if sim > best_score:
            best_score, best_user = sim, user
    return (best_user, best_score) if best_score >= RECOGNITION_THRESHOLD else (None, best_score)


def check_duplicate_face(embedding: np.ndarray):
    emb = _l2_norm(np.array(embedding, dtype=np.float32))
    best_user, best_score = None, 0.0
    for user in UserProfile.objects.all():
        stored = user.get_embeddings()
        if not stored:
            continue
        centroid_sim     = float(np.dot(emb, _get_user_centroid(stored)))
        arr              = np.array(stored, dtype=np.float32)
        norms            = np.linalg.norm(arr, axis=1, keepdims=True)
        arr              = arr / (norms + 1e-9)
        max_individual   = float((arr @ emb).max())
        sim = max(centroid_sim, max_individual)
        if sim > best_score:
            best_score, best_user = sim, user
    return (best_user, best_score) if best_score >= DUPLICATE_REG_THRESHOLD else (None, best_score)


# ── Attendance ────────────────────────────────────────────────────────────────
def mark_attendance(user, event_type: str, confidence: float):
    now      = timezone.now()
    today    = now.date()
    cooldown = ENTRY_COOLDOWN_S if event_type == 'entry' else EXIT_COOLDOWN_S

    last = (AttendanceLog.objects
            .filter(user=user, event_type=event_type)
            .order_by('-timestamp').first())
    if last and (now - last.timestamp).total_seconds() < cooldown:
        return False, 'cooldown', 0

    AttendanceLog.objects.create(
        user=user, event_type=event_type, confidence=round(confidence, 4)
    )
    entry_count = 0

    if event_type == 'entry':
        AttendanceSession.objects.create(user=user, entry_time=now, date=today)
        user.is_present = True
        entry_count = AttendanceLog.objects.filter(
            user=user, event_type='entry', timestamp__date=today
        ).count()
    else:
        sess = (AttendanceSession.objects
                .filter(user=user, exit_time=None, date=today)
                .order_by('-entry_time').first())
        if sess:
            sess.exit_time        = now
            sess.duration_minutes = round((now - sess.entry_time).total_seconds() / 60, 2)
            sess.save()
        user.is_present = False

    user.last_seen = now
    user.save(update_fields=['is_present', 'last_seen'])
    return True, 'marked', entry_count


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_frame(image_array: np.ndarray, event_type: str = 'entry'):
    faces      = detect_faces_remote(image_array)
    detections = []
    for face in faces:
        user, sim = match_face(face['embedding'])
        conf_pct  = round(sim * 100, 1)
        if user:
            logged, reason, entry_count = mark_attendance(user, event_type, sim)
            detections.append({
                'name': user.name, 'student_id': user.student_id,
                'department': user.department, 'confidence': conf_pct,
                'event_type': event_type, 'logged': logged,
                'reason': reason, 'bbox': face['bbox'], 'entry_count': entry_count,
            })
        else:
            detections.append({
                'name': 'Unknown', 'student_id': '', 'department': '',
                'confidence': conf_pct, 'event_type': None,
                'logged': False, 'reason': 'no_match',
                'bbox': face['bbox'], 'entry_count': 0,
            })
    return None, detections
