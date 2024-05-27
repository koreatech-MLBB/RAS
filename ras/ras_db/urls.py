from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from .views import *

urlpatterns = [
  # NOTE: main & user
  path('', MainView.as_view(), name='main'),
  path('sign/up', SignUpView.as_view(), name='signup'),
  path('sign/in', SignInView.as_view(), name='login'),
  path('<str:user_id>/profile', ProfileView.as_view(), name='profile'),
  # NOTE: running
  path('<str:user_id>/running/main', RunningMainView.as_view()),
  path('<str:user_id>/running', RunningView.as_view()),
  path('<str:user_id>/running/save/<int:running_id>', RunningSaveView.as_view()),
  path('<str:user_id>/running/info/<int:running_id>', RunningInfoView.as_view(), name='run_detail'),
  # path('<str:user_id>/running/log', RunningLogView.as_view()),
  # NOTE: streaming
  path('imgstreaming/<str:user_id>', StreamingView.as_view(), name='video'),
  path('audiostreaming/<str:user_id>', AudioStreamingView.as_view()),
  path('streaming/<str:user_id>', StreamingRunning.as_view()),
  path('get_next_audio/<str:user_id>', get_next_audio, name='get_next_audio'),
  # NOTE: feedback
  path('feedback', FeedbackView.as_view(), name='feedback'),
  path('feedback/<int:feed_id>', FeedbackDetailView.as_view(), name='feedback_detail')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,
                                                                             document_root=settings.MEDIA_ROOT)

