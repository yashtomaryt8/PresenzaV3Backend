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

import os
import cv2
import numpy as np
import requests
from django.utils import timezone
from api.models import UserProfile, AttendanceLog, AttendanceSession

# ── Config ────────────────────────────────────────────────────────────────────
HF_SPACE_URL = os.environ.get(
    'HF_SPACE_URL',
    'https://YOUR-USERNAME-attendai-face-service.hf.space'
).rstrip('/')

# With buffalo_l (ResNet50 ArcFace) typical same-person cosine sim is 0.65-0.92.
# 0.50 is a conservative threshold that still filters impostors well.
# With buffalo_sc (MobileFaceNet) this should be 0.40-0.42.
RECOGNITION_THRESHOLD = 0.50

# Duplicate-registration threshold: how similar must a new face be to an
# existing user's centroid to be considered "already registered".
# Set slightly higher than RECOGNITION_THRESHOLD to avoid false blocks.
DUPLICATE_REG_THRESHOLD = 0.55

# Entry cooldown: minimum seconds between two ENTRY events for the same person.
# 10 seconds is enough to avoid duplicate logs from a single standing person
# while still allowing them to re-enter after genuinely leaving and coming back.
ENTRY_COOLDOWN_S = 10

# Exit cooldown: minimum seconds between two EXIT events for the same person.
# Short because exit is triggered intentionally.
EXIT_COOLDOWN_S = 5

HF_TIMEOUT = 15  # seconds


# ── HF Space callers ──────────────────────────────────────────────────────────
def _img_to_bytes(img: np.ndarray) -> bytes:
    """Convert BGR numpy array to JPEG bytes. Higher quality = better embeddings."""
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def get_embedding(image_array: np.ndarray):
    """
    Call HF Space /extract → L2-normalised embedding of largest face.
    Used during registration.
    Returns numpy array (unit vector) or None.
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
        emb = np.array(data["embedding"], dtype=np.float32)
        # Ensure unit vector (HF Space normalises, but be defensive)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-9)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"HF extract failed: {e}")
        return None


def detect_faces_remote(image_array: np.ndarray) -> list:
    """
    Call HF Space /detect → all faces + L2-normalised embeddings + bboxes.
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


# ── Centroid Matching ──────────────────────────────────────────────────────────
def _get_user_centroid(embeddings: list) -> np.ndarray:
    """
    Compute the normalised centroid of a user's stored embeddings.

    Steps:
      1. Stack embeddings as (N, 512) matrix
      2. Normalise each row to unit vector (defensive, they should already be)
      3. Compute row-mean → (512,) centroid vector
      4. Re-normalise centroid to unit vector

    Why re-normalise step 4?
    The mean of unit vectors is NOT a unit vector. Its magnitude < 1.
    When we renormalise, we project it back onto the unit hypersphere.
    This gives us the "average direction" of the person's face cluster.
    """
    arr = np.array(embeddings, dtype=np.float32)                 # (N, 512)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)           # (N, 1)
    arr = arr / (norms + 1e-9)                                    # normalise rows
    centroid = arr.mean(axis=0)                                   # (512,)
    c_norm = np.linalg.norm(centroid)
    return centroid / (c_norm + 1e-9)                             # unit centroid


def match_face(embedding: np.ndarray):
    """
    Match embedding against the centroid of all stored user embeddings.
    Returns (user|None, score).

    Cosine similarity = dot product (since both vectors are unit vectors).
    Range: -1 to 1, where 1 = identical direction in embedding space.
    """
    emb = np.array(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)   # ensure unit vector

    best_user, best_score = None, 0.0

    for user in UserProfile.objects.all():
        stored = user.get_embeddings()
        if not stored:
            continue
        centroid = _get_user_centroid(stored)
        sim = float(np.dot(emb, centroid))       # cosine sim (unit vecs)
        if sim > best_score:
            best_score, best_user = sim, user

    return (best_user, best_score) if best_score >= RECOGNITION_THRESHOLD else (None, best_score)


def check_duplicate_face(embedding: np.ndarray):
    """
    Check if a NEW registration embedding matches any EXISTING user.
    Used to prevent the same person from registering twice.

    Returns (existing_user | None, similarity_score).
    If the return value is not (None, score), the registration should be BLOCKED
    and the API should return 409 Conflict with the existing user's name.

    Uses a LOWER threshold than RECOGNITION_THRESHOLD so we catch near-duplicates
    even when registration photos are from a different angle/lighting than scan.
    """
    emb = np.array(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    best_user, best_score = None, 0.0

    for user in UserProfile.objects.all():
        stored = user.get_embeddings()
        if not stored:
            continue
        # Check against centroid AND each individual embedding to catch edge cases
        centroid = _get_user_centroid(stored)
        centroid_sim = float(np.dot(emb, centroid))

        # Also check max individual sim (catches very close angle matches)
        arr = np.array(stored, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-9)
        max_individual_sim = float((arr @ emb).max())

        # Take the higher of the two
        sim = max(centroid_sim, max_individual_sim)

        if sim > best_score:
            best_score, best_user = sim, user

    return (best_user, best_score) if best_score >= DUPLICATE_REG_THRESHOLD else (None, best_score)


# ── Attendance ────────────────────────────────────────────────────────────────
def mark_attendance(user, event_type: str, confidence: float):
    """
    Log an attendance event with per-event-type cooldowns.

    Returns: (logged: bool, reason: str, entry_count: int)
      - entry_count is the ordinal number of today's entries for this user
        (1 = first entry, 2 = second re-entry, etc.) — for display in frontend
      - entry_count is 0 for exit events

    ─────────────────────────────────────────────────────────────────────────
    WHY PER-EVENT-TYPE COOLDOWN?
    ─────────────────────────────────────────────────────────────────────────
    The old code used a GLOBAL last-event cooldown:
        last = AttendanceLog.objects.filter(user=user).order_by('-timestamp').first()
        if (now - last.timestamp).total_seconds() < 30: BLOCK

    This was the root cause of "exit logic being shaky":
      1. Person enters (entry logged at T=0)
      2. Scanner switches to EXIT mode
      3. Person stands in front → EXIT blocked because last event < 30s ago

    Fix: separate cooldown per event_type:
      - Last ENTRY is only compared against new ENTRY events
      - Last EXIT is only compared against new EXIT events
      - An EXIT can fire immediately after an ENTRY (different event types)
    """
    now   = timezone.now()
    today = now.date()

    cooldown = ENTRY_COOLDOWN_S if event_type == 'entry' else EXIT_COOLDOWN_S
    last_same_type = (
        AttendanceLog.objects
        .filter(user=user, event_type=event_type)
        .order_by('-timestamp')
        .first()
    )

    if last_same_type and (now - last_same_type.timestamp).total_seconds() < cooldown:
        return False, 'cooldown', 0

    AttendanceLog.objects.create(
        user=user, event_type=event_type, confidence=round(confidence, 4)
    )

    entry_count = 0

    if event_type == 'entry':
        AttendanceSession.objects.create(user=user, entry_time=now, date=today)
        user.is_present = True
        # Count how many times this user has entered TODAY (including the one just created)
        entry_count = AttendanceLog.objects.filter(
            user=user, event_type='entry', timestamp__date=today
        ).count()
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
    return True, 'marked', entry_count


# ── Main pipeline ─────────────────────────────────────────────────────────────
def process_frame(image_array: np.ndarray, event_type: str = 'entry'):
    """
    1. Send frame to HF Space → face embeddings + bboxes (buffalo_l)
    2. Match each embedding against user centroids locally
    3. Mark attendance with per-type cooldowns
    4. Return detections list (frontend draws boxes on canvas)

    No annotated image returned — frontend canvas handles visualisation.
    """
    faces = detect_faces_remote(image_array)
    detections = []

    for face in faces:
        embedding  = face['embedding']
        bbox       = face['bbox']
        det_score  = face.get('det_score', 1.0)

        user, sim = match_face(embedding)
        conf_pct  = round(sim * 100, 1)

        if user:
            logged, reason, entry_count = mark_attendance(user, event_type, sim)
            detections.append({
                'name':        user.name,
                'student_id':  user.student_id,
                'department':  user.department,
                'confidence':  conf_pct,
                'event_type':  event_type,
                'logged':      logged,
                'reason':      reason,
                'bbox':        bbox,
                'entry_count': entry_count,   # 0 for exit, N for entry
            })
        else:
            detections.append({
                'name':        'Unknown',
                'student_id':  '',
                'department':  '',
                'confidence':  conf_pct,
                'event_type':  None,
                'logged':      False,
                'reason':      'no_match',
                'bbox':        bbox,
                'entry_count': 0,
            })

    return None, detections
