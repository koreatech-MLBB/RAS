from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from . import views

urlpatterns = [
    path('sign/up', views.SignUpView.as_view()),
    path('sign/in', views.SignInView.as_view()),
    path('<str:user_id>/running', views.RunningView.as_view()),
    path('<str:user_id>/running/info', views.RunningSaveView.as_view()),
    path('<str:user_id>/running/info/<int:running_id>', views.RunningInfoView.as_view()),
    path('imgstreaming/<str:user_id>', views.StreamingView.as_view()),
    path('audiostreaming/<str:user_id>', views.AudioStreamingView.as_view()),
    path('streaming/<str:user_id>', views.StreamingRunning.as_view()),
    path('get_next_audio/<str:user_id>', views.next_audio.as_view())
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
