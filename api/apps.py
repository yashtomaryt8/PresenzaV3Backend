"""
apps.py
─────────────────────────────────────────────────────────────────────────────
FIXES:
  1. Only ONE worker starts the ping thread (file-lock based, no Redis needed)
  2. Ping uses a 60s timeout (buffalo_l cold start = 30-60s)
  3. Wake-up mode: if ping fails, retry every 30s until Space responds,
     then switch back to the normal 4-min keep-alive interval
"""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        import os

        # ── Only run in the actual server process, not management commands ──
        run_main           = os.environ.get('RUN_MAIN')           # dev reloader child
        railway_env        = os.environ.get('RAILWAY_ENVIRONMENT')
        server_software    = os.environ.get('SERVER_SOFTWARE', '')
        is_gunicorn        = 'gunicorn' in server_software.lower()

        if not (run_main == 'true' or railway_env or is_gunicorn):
            return

        # ── Only ONE worker should run the ping thread ────────────────────
        # Gunicorn spawns N worker processes, each calls ready().
        # We use a lock file: the first worker to create it wins.
        # All others see the file exists and skip silently.
        import fcntl
        lock_path = '/tmp/hf_keepalive.lock'
        try:
            lock_fd = open(lock_path, 'w')
            # LOCK_EX | LOCK_NB: exclusive, non-blocking
            # Raises OSError if another worker already holds the lock
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # This worker holds the lock — start the thread
            from api.face_utils import start_hf_keepalive
            start_hf_keepalive()
        except OSError:
            # Another worker already holds the lock — skip
            pass
