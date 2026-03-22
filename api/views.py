import cv2
import numpy as np
import base64
import csv
import io
from datetime import date, timedelta, datetime
from collections import defaultdict

from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.conf import settings
from django.db.models import Count, Avg

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import UserProfile, AttendanceLog, AttendanceSession
from .serializers import UserSerializer, AttendanceLogSerializer, AttendanceSessionSerializer
from .face_utils import get_embedding, process_frame
from .ai_utils import query_groq, query_ollama, build_analytics_prompt


# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode_image(file):
    raw = np.frombuffer(file.read(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


# ── Register ──────────────────────────────────────────────────────────────────
class RegisterUser(APIView):
    def post(self, request):
        name       = request.data.get('name', '').strip()
        student_id = request.data.get('student_id', '').strip()
        department = request.data.get('department', '').strip()

        if not name:
            return Response({'error': 'Name is required.'}, status=400)

        images = []
        for key, f in request.FILES.items():
            if key.startswith('image'):
                img = _decode_image(f)
                if img is not None:
                    images.append(img)

        if not images:
            return Response({'error': 'At least one image is required.'}, status=400)

        user = None
        if student_id:
            user = UserProfile.objects.filter(student_id=student_id).first()
        if user is None:
            user = UserProfile(name=name, student_id=student_id, department=department)

        added, failed = 0, 0
        for img in images:
            emb = get_embedding(img)  # calls HF Space
            if emb is not None:
                user.add_embedding(emb)
                added += 1
            else:
                failed += 1

        if added == 0:
            return Response({'error': 'No faces detected in any image. Is HF Space running?'}, status=400)

        user.save()
        return Response({
            'message':     f"Registered '{name}' with {added} photo(s).",
            'failed':      failed,
            'photo_count': user.photo_count,
            'user_id':     user.id,
        }, status=201)


# ── Add Photos ────────────────────────────────────────────────────────────────
class AddPhotos(APIView):
    def post(self, request, pk):
        try:
            user = UserProfile.objects.get(pk=pk)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User not found.'}, status=404)

        added = 0
        for key, f in request.FILES.items():
            if key.startswith('image'):
                img = _decode_image(f)
                if img is not None:
                    emb = get_embedding(img)
                    if emb is not None:
                        user.add_embedding(emb)
                        added += 1
        if added == 0:
            return Response({'error': 'No valid faces found.'}, status=400)
        user.save()
        return Response({'message': f'Added {added} photo(s).', 'photo_count': user.photo_count})


# ── Scan ──────────────────────────────────────────────────────────────────────
class ScanFrame(APIView):
    def post(self, request):
        img = None

        # Option A: file upload
        f = request.FILES.get('image')
        if f:
            img = _decode_image(f)

        # Option B: base64 JSON
        elif 'image' in request.data:
            try:
                image_data = request.data.get('image')
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                img_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                return Response({'error': 'Invalid base64'}, status=400)

        else:
            return Response({'error': 'Image required.'}, status=400)

        if img is None:
            return Response({'error': 'Could not decode image.'}, status=400)

        event_type = request.data.get('event_type', 'entry')
        if event_type not in ('entry', 'exit'):
            event_type = 'entry'

        # process_frame calls HF Space for detection, does matching locally
        _, detections = process_frame(img, event_type)

        return Response({
            'detections':  detections,
            'face_count':  len(detections),
            'known_count': sum(1 for d in detections if d['name'] != 'Unknown'),
        })


# ── Users ─────────────────────────────────────────────────────────────────────
class UserListView(APIView):
    def get(self, request):
        return Response(UserSerializer(UserProfile.objects.all().order_by('name'), many=True).data)


class UserDetailView(APIView):
    def delete(self, request, pk):
        try:
            UserProfile.objects.get(pk=pk).delete()
            return Response({'message': 'Deleted.'}, status=204)
        except UserProfile.DoesNotExist:
            return Response({'error': 'Not found.'}, status=404)


# ── Logs ──────────────────────────────────────────────────────────────────────
class AttendanceLogView(APIView):
    def get(self, request):
        qs = AttendanceLog.objects.select_related('user').all()
        if n := request.query_params.get('name'):
            qs = qs.filter(user__name__icontains=n)
        if e := request.query_params.get('event'):
            if e in ('entry', 'exit'):
                qs = qs.filter(event_type=e)
        if d := request.query_params.get('date'):
            try:
                qs = qs.filter(timestamp__date=datetime.strptime(d, '%Y-%m-%d').date())
            except ValueError:
                pass
        limit = int(request.query_params.get('limit', 200))
        return Response(AttendanceLogSerializer(qs[:limit], many=True).data)


class AttendanceSessionView(APIView):
    def get(self, request):
        qs = AttendanceSession.objects.select_related('user').all()
        if d := request.query_params.get('date'):
            try:
                qs = qs.filter(date=datetime.strptime(d, '%Y-%m-%d').date())
            except ValueError:
                pass
        limit = int(request.query_params.get('limit', 50))
        return Response(AttendanceSessionSerializer(qs[:limit], many=True).data)


# ── Analytics ─────────────────────────────────────────────────────────────────
class AnalyticsView(APIView):
    def get(self, request):
        now   = timezone.now()
        today = now.date()

        total_users   = UserProfile.objects.count()
        present_now   = UserProfile.objects.filter(is_present=True).count()
        today_entries = AttendanceLog.objects.filter(timestamp__date=today, event_type='entry')
        present_today = today_entries.values('user').distinct().count()
        attendance_rate = round(present_today / total_users * 100, 1) if total_users else 0
        late_today = today_entries.filter(timestamp__hour__gte=9).values('user').distinct().count()

        week_data = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            count = AttendanceLog.objects.filter(
                timestamp__date=d, event_type='entry'
            ).values('user').distinct().count()
            week_data.append({'date': str(d), 'label': d.strftime('%a'), 'count': count})

        week_avg   = round(sum(w['count'] for w in week_data) / 7, 1)
        week_total = sum(w['count'] for w in week_data)

        hourly = defaultdict(int)
        for log in today_entries:
            hourly[log.timestamp.hour] += 1
        hourly_data = [{'hour': h, 'label': f'{h:02d}:00', 'count': hourly.get(h, 0)}
                       for h in range(6, 22)]

        peak = max(hourly_data, key=lambda x: x['count'], default=None)
        peak_hour = peak['label'] if peak and peak['count'] > 0 else 'N/A'

        top_attendees = list(
            AttendanceLog.objects
            .filter(event_type='entry')
            .values('user__name', 'user__student_id')
            .annotate(total=Count('id'))
            .order_by('-total')[:5]
        )

        avg_duration = AttendanceSession.objects.filter(
            date=today, duration_minutes__isnull=False
        ).aggregate(avg=Avg('duration_minutes'))['avg']

        present_users = list(UserProfile.objects.filter(is_present=True).values(
            'id', 'name', 'student_id', 'department', 'last_seen'
        ))

        return Response({
            'total_users':      total_users,
            'present_now':      present_now,
            'present_today':    present_today,
            'attendance_rate':  attendance_rate,
            'late_today':       late_today,
            'week_data':        week_data,
            'week_avg':         week_avg,
            'week_total':       week_total,
            'hourly_data':      hourly_data,
            'peak_hour':        peak_hour,
            'top_attendees':    top_attendees,
            'avg_duration_min': round(avg_duration, 1) if avg_duration else None,
            'present_users':    present_users,
        })


# ── Export CSV ────────────────────────────────────────────────────────────────
class ExportCSV(APIView):
    def get(self, request):
        d_str = request.query_params.get('date', str(date.today()))
        try:
            d = datetime.strptime(d_str, '%Y-%m-%d').date()
        except ValueError:
            d = date.today()

        logs = AttendanceLog.objects.filter(timestamp__date=d).select_related('user').order_by('timestamp')
        buf  = io.StringIO()
        w    = csv.writer(buf)
        w.writerow(['Name', 'Student ID', 'Department', 'Event', 'Time', 'Confidence %'])
        for log in logs:
            w.writerow([
                log.user.name, log.user.student_id, log.user.department,
                log.event_type,
                log.timestamp.strftime('%H:%M:%S'),
                round(log.confidence * 100, 1),
            ])

        resp = HttpResponse(buf.getvalue(), content_type='text/csv')
        resp['Content-Disposition'] = f'attachment; filename="attendance_{d_str}.csv"'
        return resp


# ── AI Insight ────────────────────────────────────────────────────────────────
class AIInsightView(APIView):
    def post(self, request):
        mode   = request.data.get('mode', 'groq')
        custom = request.data.get('prompt', '').strip()

        today = timezone.now().date()
        total = UserProfile.objects.count()
        present = AttendanceLog.objects.filter(
            timestamp__date=today, event_type='entry'
        ).values('user').distinct().count()

        stats = {
            'total_users':          total,
            'present_today':        present,
            'attendance_rate_today': round(present / total * 100, 1) if total else 0,
            'late_today':           AttendanceLog.objects.filter(
                                        timestamp__date=today,
                                        event_type='entry',
                                        timestamp__hour__gte=9
                                    ).values('user').distinct().count(),
            'week_total':           AttendanceLog.objects.filter(
                                        timestamp__date__gte=today - timedelta(days=6),
                                        event_type='entry'
                                    ).count(),
            'week_avg':             0,
            'top_attendee':         'N/A',
            'peak_hour':            'unknown',
        }

        top = (AttendanceLog.objects.filter(event_type='entry')
               .values('user__name').annotate(t=Count('id')).order_by('-t').first())
        if top:
            stats['top_attendee'] = top['user__name']

        prompt = custom if custom else build_analytics_prompt(stats)

        if mode == 'ollama':
            result = query_ollama(
                prompt,
                host=getattr(settings, 'OLLAMA_HOST', 'http://localhost:11434'),
                model=getattr(settings, 'OLLAMA_MODEL', 'llama3.2:1b'),
            )
        else:
            result = query_groq(
                prompt,
                api_key=getattr(settings, 'GROQ_API_KEY', ''),
                model=getattr(settings, 'GROQ_MODEL', 'llama-3.1-8b-instant'),
            )

        return Response({'insight': result, 'mode': mode})


# ── Reset ─────────────────────────────────────────────────────────────────────
class ResetPresence(APIView):
    def post(self, request):
        UserProfile.objects.all().update(is_present=False)
        return Response({'message': 'All users marked absent.'})


# ── Health ────────────────────────────────────────────────────────────────────
class HealthCheck(APIView):
    def get(self, request):
        hf_url = getattr(settings, 'HF_SPACE_URL',
                         os.environ.get('HF_SPACE_URL', 'not set'))
        return Response({
            'status': 'ok',
            'users':  UserProfile.objects.count(),
            'hf_space': hf_url,
        })

import os
