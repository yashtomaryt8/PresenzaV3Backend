"""
face_utils.py — V3
Clean single-docstring version. Three docstrings were concatenated in the
previous version (not a Python error, but confusing and indicative of merges).
All logic is identical — only the docstrings are cleaned up.
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

HF_SPACE_URL = os.environ.get(
    'HF_SPACE_URL',
    'https://YOUR-USERNAME-attendai-face-service.hf.space'
).rstrip('/')

RECOGNITION_THRESHOLD   = 0.50
DUPLICATE_REG_THRESHOLD = 0.55
ENTRY_COOLDOWN_S        = 10
EXIT_COOLDOWN_S         = 5
HF_TIMEOUT              = 30
PING_TIMEOUT            = 60
PING_INTERVAL_NORMAL_S  = 4 * 60
PING_INTERVAL_WAKEUP_S  = 30


# ── Keep-alive pinger ─────────────────────────────────────────────────────────
def _hf_ping_loop():
    time.sleep(15)
    space_is_healthy = False

    while True:
        timeout  = PING_TIMEOUT if not space_is_healthy else 10
        interval = PING_INTERVAL_NORMAL_S if space_is_healthy else PING_INTERVAL_WAKEUP_S

        try:
            r = requests.get(f"{HF_SPACE_URL}/health", timeout=timeout)
            if r.status_code == 200:
                if not space_is_healthy:
                    logger.info("HF Space is UP: %s", r.json())
                    space_is_healthy = True
                else:
                    logger.debug("HF ping OK")
            else:
                logger.warning("HF ping HTTP %s", r.status_code)
                space_is_healthy = False
        except requests.exceptions.Timeout:
            logger.warning("HF ping timed out (%ds) — retrying in %ds.", timeout, PING_INTERVAL_WAKEUP_S)
            space_is_healthy = False
        except Exception as e:
            logger.warning("HF ping error: %s", e)
            space_is_healthy = False

        time.sleep(interval)


def start_hf_keepalive():
    t = threading.Thread(target=_hf_ping_loop, daemon=True, name="hf-keepalive")
    t.start()
    logger.info("HF keep-alive started (normal: %ds, wake-up: %ds)", PING_INTERVAL_NORMAL_S, PING_INTERVAL_WAKEUP_S)


# ── HF HTTP wrapper ────────────────────────────────────────────────────────────
def _hf_post(endpoint: str, files: dict) -> dict | None:
    url = f"{HF_SPACE_URL}/{endpoint.lstrip('/')}"
    timeouts = [30, 45]
    for attempt, timeout in enumerate(timeouts):
        try:
            resp = requests.post(url, files=files, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.error("HF %s HTTP %s", endpoint, resp.status_code)
            return None
        except requests.exceptions.Timeout:
            if attempt < len(timeouts) - 1:
                logger.warning("HF %s timeout (attempt %d) — retrying…", endpoint, attempt + 1)
                time.sleep(3)
            else:
                logger.error("HF %s timed out on all attempts", endpoint)
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
        centroid_sim   = float(np.dot(emb, _get_user_centroid(stored)))
        arr            = np.array(stored, dtype=np.float32)
        norms          = np.linalg.norm(arr, axis=1, keepdims=True)
        arr            = arr / (norms + 1e-9)
        max_individual = float((arr @ emb).max())
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
