from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
    return HttpResponse("ras_db index")

def hello(request):
    return HttpResponse("Hello!! This is ras_db!!")