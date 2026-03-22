"""
face_utils.py
─────────────────────────────────────────────────────
All heavy face detection/embedding now runs on HF Space.
Django only does:
  - cosine similarity matching against stored embeddings
  - attendance logging
  - bounding box data passthrough to frontend

Set HF_SPACE_URL in Railway environment variables:
  HF_SPACE_URL=https://YOUR-USERNAME-attendai-face-service.hf.space
"""

import os
import cv2
import numpy as np
import requests
import io
from django.utils import timezone
from api.models import UserProfile, AttendanceLog, AttendanceSession

# ── Config ────────────────────────────────────────────────────────────────────
HF_SPACE_URL = os.environ.get(
    'HF_SPACE_URL',
    'https://YOUR-USERNAME-attendai-face-service.hf.space'
).rstrip('/')

RECOGNITION_THRESHOLD = 0.42
ATTENDANCE_COOLDOWN_S  = 30
HF_TIMEOUT = 15  # seconds


# ── HF Space callers ──────────────────────────────────────────────────────────
def _img_to_bytes(img: np.ndarray) -> bytes:
    """Convert BGR numpy array to JPEG bytes."""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def get_embedding(image_array: np.ndarray):
    """
    Call HF Space /extract to get embedding of largest face.
    Used during registration.
    Returns numpy array or None.
    """
    try:
        resp = requests.post(
            f"{HF_SPACE_URL}/extract",
            files={"image": ("face.jpg", _img_to_bytes(image_array), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return np.array(data["embedding"], dtype=np.float32)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"HF extract failed: {e}")
        return None


def detect_faces_remote(image_array: np.ndarray) -> list:
    """
    Call HF Space /detect to get all faces + embeddings in a frame.
    Returns list of {embedding, bbox, det_score} or empty list on failure.
    """
    try:
        resp = requests.post(
            f"{HF_SPACE_URL}/detect",
            files={"image": ("frame.jpg", _img_to_bytes(image_array), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("faces", [])
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"HF detect failed: {e}")
        return []


# ── Matching ──────────────────────────────────────────────────────────────────
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def match_face(embedding: np.ndarray):
    """Match embedding against all stored users. Returns (user|None, score)."""
    best_user, best_score = None, 0.0
    emb = np.array(embedding, dtype=np.float32)
    for user in UserProfile.objects.all():
        for stored in user.get_embeddings():
            sim = _cosine_sim(emb, np.array(stored, dtype=np.float32))
            if sim > best_score:
                best_score, best_user = sim, user
    return (best_user, best_score) if best_score >= RECOGNITION_THRESHOLD else (None, best_score)


# ── Attendance ────────────────────────────────────────────────────────────────
def mark_attendance(user, event_type: str, confidence: float):
    now  = timezone.now()
    last = AttendanceLog.objects.filter(user=user).order_by('-timestamp').first()
    if last and (now - last.timestamp).total_seconds() < ATTENDANCE_COOLDOWN_S:
        return False, 'cooldown'

    AttendanceLog.objects.create(
        user=user, event_type=event_type, confidence=round(confidence, 4)
    )
    today = now.date()

    if event_type == 'entry':
        AttendanceSession.objects.create(user=user, entry_time=now, date=today)
        user.is_present = True
    else:
        sess = AttendanceSession.objects.filter(
            user=user, exit_time=None, date=today
        ).order_by('-entry_time').first()
        if sess:
            sess.exit_time = now
            sess.duration_minutes = round((now - sess.entry_time).total_seconds() / 60, 2)
            sess.save()
        user.is_present = False

    user.last_seen = now
    user.save(update_fields=['is_present', 'last_seen'])
    return True, 'marked'


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_frame(image_array: np.ndarray, event_type: str = 'entry'):
    """
    1. Send frame to HF Space → get face embeddings + bboxes
    2. Match each embedding against DB locally
    3. Mark attendance
    4. Return detections list (frontend draws boxes on canvas)

    No annotated image returned — frontend canvas handles visualisation.
    """
    faces = detect_faces_remote(image_array)
    detections = []

    for face in faces:
        embedding  = face['embedding']
        bbox       = face['bbox']      # [x1, y1, x2, y2]
        det_score  = face.get('det_score', 1.0)

        user, sim = match_face(embedding)
        conf_pct  = round(sim * 100, 1)

        if user:
            logged, reason = mark_attendance(user, event_type, sim)
            detections.append({
                'name':       user.name,
                'student_id': user.student_id,
                'department': user.department,
                'confidence': conf_pct,
                'event_type': event_type,
                'logged':     logged,
                'reason':     reason,
                'bbox':       bbox,
            })
        else:
            detections.append({
                'name':       'Unknown',
                'student_id': '',
                'department': '',
                'confidence': conf_pct,
                'event_type': None,
                'logged':     False,
                'reason':     'no_match',
                'bbox':       bbox,
            })

    return None, detections  # no annotated image (frontend draws boxes)
