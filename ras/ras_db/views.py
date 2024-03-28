from django.http import HttpResponse, JsonResponse
from rest_framework import serializers

from .models import *


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'
# Create your views here.


def index(request):
    return HttpResponse("ras_db index")


def get_users(request):
    if request.method == 'GET':
        results = UserSerializer(User.objects.all(), many=True).data
        return JsonResponse(results, safe=False)


def signin(request):
    if request.method == 'POST':
        data = request.POST
        User.objects.create(
            user_id=data['user_id'],
            username=data['username'],
            password=data['password'],
            name=data['name'],
            age=data['age'],
            weight=data['weight'],
            height=data['height'],
            gender=data['gender']
        )
        return JsonResponse({"result": True})
    else:
        return JsonResponse({"result": "이런 요청 안받습니다."})
