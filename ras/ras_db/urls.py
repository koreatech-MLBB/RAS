from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('sign/up', views.SignUpView.as_view()),
    path('sign/in', views.SignInView.as_view()),
    path('<str:user_id>/running', views.RunningView.as_view()),
    path('<str:user_id>/running/info', views.RunningSaveView.as_view()),
    path('<str:user_id>/running/info/<int:running_id>', views.RunningInfoView.as_view())
]
