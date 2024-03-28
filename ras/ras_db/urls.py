from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('sign/up', views.SignUp.as_view()),
    path('sign/in', views.SignIn.as_view())
]
