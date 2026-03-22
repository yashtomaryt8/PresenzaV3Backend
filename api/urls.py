from django.urls import path
from .views import (
    RegisterUser, AddPhotos, ScanFrame,
    UserListView, UserDetailView,
    AttendanceLogView, AttendanceSessionView,
    AnalyticsView, ExportCSV,
    AIInsightView, ResetPresence, HealthCheck,
)

urlpatterns = [
    # Health
    path('health/',                   HealthCheck.as_view()),

    # User management
    path('register/',                 RegisterUser.as_view()),
    path('users/',                    UserListView.as_view()),
    path('users/<int:pk>/delete/',    UserDetailView.as_view()),
    path('users/<int:pk>/photos/',    AddPhotos.as_view()),

    # Scanning
    path('scan/',                     ScanFrame.as_view()),

    # Logs & Sessions
    path('logs/',                     AttendanceLogView.as_view()),
    path('sessions/',                 AttendanceSessionView.as_view()),

    # Analytics & Export
    path('analytics/',               AnalyticsView.as_view()),
    path('export/',                   ExportCSV.as_view()),

    # AI
    path('ai-insight/',              AIInsightView.as_view()),

    # Utility
    path('reset-presence/',          ResetPresence.as_view()),
]
