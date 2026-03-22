"""
apps.py
─────────────────────────────────────────────────────────────────────────────
AppConfig.ready() is the correct Django hook for "run once when the server
starts". We use it to kick off the HF Space keep-alive thread.

Why not put this in face_utils.py directly at module import time?
Because Django imports models/utils multiple times during startup (management
commands, migrations, etc.). AppConfig.ready() fires exactly once per process
after all apps are loaded — the safe place for side effects like threads.
"""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        # Only start the keep-alive thread in the actual server process,
        # not during manage.py migrate, collectstatic, shell, etc.
        import os
        # RUN_MAIN=true is set by Django's auto-reloader for the child process
        # In production (gunicorn) there is no RUN_MAIN, so we check for that too
        run_main = os.environ.get('RUN_MAIN')
        server_software = os.environ.get('SERVER_SOFTWARE', '')  # gunicorn sets this

        is_dev_child   = run_main == 'true'
        is_production  = 'gunicorn' in server_software.lower() or os.environ.get('RAILWAY_ENVIRONMENT')

        if is_dev_child or is_production:
            from api.face_utils import start_hf_keepalive
            start_hf_keepalive()
